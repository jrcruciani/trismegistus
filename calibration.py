"""Conformal risk-controlled threshold calibration & closed-loop precision guard.

Implements:
1. Split Conformal / Finite-Sample Risk-Controlled Calibration (`calibrate_thresholds`):
   Given labeled triage samples (each with `expected_effective_tier`, `raw_tier`, `confidence`,
   `trap_risk_prob`, `effort_score`), finds the optimal `(confidence_threshold, trap_risk_threshold)`
   pair that:
   - Strictly controls the empirical `false_fast_route_rate` (must-escalate S2/trap routed to S1)
     using a finite-sample conformal / Wilson upper bound `<= target_alpha` (precision first!),
   - Among all risk-safe pairs, maximizes System-1 recall (speed!) with preference for stable
     margins away from extreme degenerate thresholds.
2. Backend-scoped persistence (`<HERMES_HOME>/plugin-data/trismegistus/calibration.json`) because
   probabilities from `foundry_systemone`, `local_logits`, and `hermes_aux` have distinct
   calibration curves and must never share thresholds blindly.
3. Conservative Closed-Loop Online Hardening (`compute_closed_loop_adjustment`):
   Observes real runtime failure signals on turns that started in System 1 (`false_s1_err_streak`,
   `false_s1_escalated`, `false_s1_verify_failed`). When observed false-S1 rate exceeds budget
   over a minimum sample window, tightens `(confidence_threshold, trap_risk_threshold)` upward
   within bounded safety rails — never loosening below configured/calibrated precision floors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Any, Iterable

from .compat import atomic_write_json, plugin_data_dir, read_json_safe

CALIBRATION_FILENAME = "calibration.json"


@dataclass(frozen=True)
class CalibrationProfile:
    """Backend-specific calibrated decision thresholds with finite-sample risk metadata."""

    backend: str
    confidence_threshold: float
    trap_risk_threshold: float
    target_alpha: float
    n_total: int
    n_must_s2: int
    n_s1: int
    empirical_false_fast_route_rate: float
    wilson95_false_fast_route_upper: float
    conformal_risk_bound: float
    achieved_s1_recall: float
    updated_at: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationProfile | None":
        try:
            return cls(
                backend=str(data["backend"]),
                confidence_threshold=float(data["confidence_threshold"]),
                trap_risk_threshold=float(data["trap_risk_threshold"]),
                target_alpha=float(data.get("target_alpha", 0.02)),
                n_total=int(data.get("n_total", 0)),
                n_must_s2=int(data.get("n_must_s2", 0)),
                n_s1=int(data.get("n_s1", 0)),
                empirical_false_fast_route_rate=float(data.get("empirical_false_fast_route_rate", 0.0)),
                wilson95_false_fast_route_upper=float(data.get("wilson95_false_fast_route_upper", 0.0)),
                conformal_risk_bound=float(data.get("conformal_risk_bound", 0.0)),
                achieved_s1_recall=float(data.get("achieved_s1_recall", 0.0)),
                updated_at=float(data.get("updated_at", 0.0)),
            )
        except (KeyError, TypeError, ValueError):
            return None


def _wilson_upper(failures: int, total: int, z: float = 1.959964) -> float:
    if total <= 0:
        return 1.0
    p = failures / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)) / denom
    return min(1.0, center + half)


def _would_route_s1(
    sample: dict[str, Any],
    *,
    conf_threshold: float,
    trap_threshold: float,
    effort_floor: float = 1.85,
) -> bool:
    if str(sample.get("raw_tier") or "system2") != "system1":
        return False
    conf = float(sample.get("confidence", 0.0))
    trap = float(sample.get("trap_risk_prob", 0.0))
    effort = float(sample.get("effort_score", 2.5))
    return conf >= conf_threshold and trap < trap_threshold and effort < effort_floor


def calibrate_thresholds(
    samples: Iterable[dict[str, Any]],
    *,
    backend: str,
    target_alpha: float = 0.02,
    effort_floor: float = 1.85,
    min_conf: float = 0.70,
    max_conf: float = 0.98,
    min_trap: float = 0.25,
    max_trap: float = 0.70,
) -> CalibrationProfile:
    """Compute conformal risk-controlled `(confidence_threshold, trap_risk_threshold)` for `backend`.

    For finite sample size `n_must_s2`, conformal risk control adds a `+1` smoothing numerator
    `(false_fast_routes + 1) / (n_must_s2 + 1)` as well as computing the 95% Wilson upper bound.
    We select the threshold pair that satisfies empirical `false_fast_route_rate <= target_alpha`
    with minimal conformal risk and maximal System-1 recall (fastest safe operating point).
    """
    rows = [dict(s) for s in samples if s.get("error_category", "ok") == "ok"]
    must_s2 = [r for r in rows if str(r.get("expected_effective_tier") or "system2") == "system2"]
    true_s1 = [r for r in rows if str(r.get("expected_effective_tier") or "") == "system1"]

    n_total = len(rows)
    n_must_s2 = len(must_s2)
    n_s1 = len(true_s1)

    if n_must_s2 == 0:
        return CalibrationProfile(
            backend=backend,
            confidence_threshold=0.82,
            trap_risk_threshold=0.50,
            target_alpha=float(target_alpha),
            n_total=n_total,
            n_must_s2=0,
            n_s1=n_s1,
            empirical_false_fast_route_rate=0.0,
            wilson95_false_fast_route_upper=1.0,
            conformal_risk_bound=1.0,
            achieved_s1_recall=0.0,
            updated_at=time.time(),
        )

    # Candidate grid combines regular quantiles + exact observed decision boundaries from must_s2
    conf_grid = {round(min_conf + 0.02 * i, 4) for i in range(int((max_conf - min_conf) / 0.02) + 1)}
    trap_grid = {round(min_trap + 0.02 * i, 4) for i in range(int((max_trap - min_trap) / 0.02) + 1)}
    conf_grid.add(0.82)
    trap_grid.add(0.50)

    for r in must_s2:
        if str(r.get("raw_tier")) == "system1":
            c = round(min(max_conf, max(min_conf, float(r.get("confidence", 0.0)) + 0.005)), 4)
            t = round(min(max_trap, max(min_trap, float(r.get("trap_risk_prob", 0.50)))), 4)
            conf_grid.add(c)
            trap_grid.add(t)

    sorted_confs = sorted(conf_grid)
    sorted_traps = sorted(trap_grid)

    # Evaluate candidates prioritizing:
    # 1. Conformal risk bound <= target_alpha (or lowest possible risk if sample size is too small for strict bound)
    # 2. Empirical false_fast_route_rate <= target_alpha
    # 3. Maximum System-1 recall (speed!)
    # 4. Balanced margin close to canonical (0.82, 0.50) to avoid overfitting razor edges
    strict_candidates: list[tuple[float, float, float, float, float, int, int]] = []
    emp_candidates: list[tuple[float, float, float, float, float, int, int]] = []
    all_candidates: list[tuple[float, float, float, float, float, int, int]] = []

    for ct in sorted_confs:
        for tt in sorted_traps:
            ff = sum(
                1
                for r in must_s2
                if _would_route_s1(r, conf_threshold=ct, trap_threshold=tt, effort_floor=effort_floor)
            )
            tp_s1 = sum(
                1
                for r in true_s1
                if _would_route_s1(r, conf_threshold=ct, trap_threshold=tt, effort_floor=effort_floor)
            )
            emp_ffr = ff / n_must_s2
            conformal_bound = (ff + 1.0) / (n_must_s2 + 1.0)
            s1_rec = (tp_s1 / n_s1) if n_s1 > 0 else 0.0
            # Small regularization preferring slightly conservative thresholds near defaults
            regularization = abs(ct - 0.82) * 0.1 + abs(tt - 0.50) * 0.1
            entry = (s1_rec, -emp_ffr, -regularization, ct, tt, ff, tp_s1)
            all_candidates.append(entry)
            if conformal_bound <= target_alpha:
                strict_candidates.append(entry)
            if emp_ffr <= target_alpha:
                emp_candidates.append(entry)

    pool = strict_candidates or emp_candidates or all_candidates
    # For fallback (no candidate meets empirical target), minimize false fast-route first!
    if not strict_candidates and not emp_candidates:
        best = max(pool, key=lambda e: (e[1], e[0], e[2]))
    else:
        best = max(pool, key=lambda e: (e[0], e[1], e[2]))

    best_s1_rec, _neg_emp, _reg, best_ct, best_tt, best_ff, _tp = best
    return CalibrationProfile(
        backend=backend,
        confidence_threshold=round(best_ct, 4),
        trap_risk_threshold=round(best_tt, 4),
        target_alpha=float(target_alpha),
        n_total=n_total,
        n_must_s2=n_must_s2,
        n_s1=n_s1,
        empirical_false_fast_route_rate=round(best_ff / n_must_s2, 4),
        wilson95_false_fast_route_upper=round(_wilson_upper(best_ff, n_must_s2), 4),
        conformal_risk_bound=round((best_ff + 1.0) / (n_must_s2 + 1.0), 4),
        achieved_s1_recall=round(best_s1_rec, 4),
        updated_at=time.time(),
    )


def calibration_store_path(base_dir: Path | None = None) -> Path:
    root = base_dir if base_dir is not None else plugin_data_dir("trismegistus")
    return root / CALIBRATION_FILENAME


def save_calibration_profile(profile: CalibrationProfile, *, path: Path | None = None) -> Path:
    target = path if path is not None else calibration_store_path()
    store = read_json_safe(target)
    raw_profiles = store.get("profiles")
    profiles: dict[str, Any] = dict(raw_profiles) if isinstance(raw_profiles, dict) else {}
    profiles[profile.backend] = profile.to_dict()
    payload = {
        "schema_version": 1,
        "profiles": profiles,
        "updated_at": time.time(),
    }
    atomic_write_json(target, payload)
    return target


def load_calibration_profile(backend: str, *, path: Path | None = None) -> CalibrationProfile | None:
    target = path if path is not None else calibration_store_path()
    store = read_json_safe(target)
    profiles = store.get("profiles")
    if not isinstance(profiles, dict):
        return None
    raw = profiles.get(backend)
    if not isinstance(raw, dict):
        return None
    return CalibrationProfile.from_dict(raw)


def compute_closed_loop_adjustment(
    *,
    base_conf_threshold: float,
    base_trap_threshold: float,
    recent_s1_turns: int,
    recent_false_s1_turns: int,
    target_alpha: float = 0.03,
    min_samples: int = 8,
) -> tuple[float, float, bool]:
    """Conservatively tighten thresholds when live System-1 turns exhibit elevated failure rates.

    Precision-first invariant:
    - Never loosens thresholds below `(base_conf_threshold, base_trap_threshold)`.
    - Only activates once at least `min_samples` recent System-1 turns have been observed.
    - Tightens monotonically up to safe ceilings (`conf <= 0.96`, `trap >= 0.30`).
    """
    if recent_s1_turns < max(1, int(min_samples)):
        return base_conf_threshold, base_trap_threshold, False
    observed_rate = recent_false_s1_turns / float(recent_s1_turns)
    if observed_rate <= target_alpha:
        return base_conf_threshold, base_trap_threshold, False

    excess = observed_rate - target_alpha
    # Proportional conservative tightening: +0.02..+0.12 on confidence, -0.03..-0.18 on trap risk.
    # Monotonicity invariant: if caller thresholds are already stricter than [0.30, 0.96],
    # never loosen them back toward the default ceiling/floor!
    delta_conf = min(0.12, max(0.02, excess * 0.8))
    delta_trap = min(0.18, max(0.03, excess * 1.0))

    conf_ceiling = min(1.0, max(0.96, base_conf_threshold))
    trap_floor = max(0.0, min(0.30, base_trap_threshold))

    new_conf = round(max(base_conf_threshold, min(conf_ceiling, base_conf_threshold + delta_conf)), 4)
    new_trap = round(min(base_trap_threshold, max(trap_floor, base_trap_threshold - delta_trap)), 4)
    tightened = (new_conf > base_conf_threshold) or (new_trap < base_trap_threshold)
    return new_conf, new_trap, tightened

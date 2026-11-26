#!/usr/bin/env python3
"""Held-out 300-turn independent benchmark & SQLite ledger auditor for Trismegistus.

Guarantees:
- Strictly evaluates in `mode='shadow'` (verifying zero request mutations on every turn).
- Evaluates each JSONL entry as an independent conversation turn with a unique `session_id`
  and `triage_cache_ttl_seconds=0.0` (never counting multi-step tool loops as samples).
- Never reads `.env` or `auth.json` directly and never prints or logs secret values.
- Performs non-destructive preflight checks before running 300 live network calls so missing
  or expired credentials report an exact `blocked` status instead of fabricating metrics.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import socket
import sys
import time
from typing import Any
import urllib.parse

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = Path(__file__).resolve().parent / "heldout_300.jsonl"
FOUNDRY_PRICE_PER_1M_INPUT_USD = 0.042


def _load_trismegistus_pkg():
    pkg_name = "trismegistus_bench_pkg"
    if pkg_name in sys.modules:
        return sys.modules[pkg_name]
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        str(ROOT / "__init__.py"),
        submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Trismegistus package from {ROOT}")
    mod = importlib.util.module_from_spec(spec)
    mod.__path__ = [str(ROOT)]
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    ordered = sorted(vals)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(ordered[lo])
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def wilson_interval(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson binomial confidence interval `(low, high)`."""
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def load_heldout_dataset(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_turn_fingerprints: set[str] = set()

    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            item_id = str(obj["id"]).strip()
            label = str(obj["label"]).strip()
            expected = str(obj["expected_effective_tier"]).strip()
            msg = str(obj["user_message"]).strip()
            hist = obj.get("conversation_history") or []

            if item_id in seen_ids:
                raise ValueError(f"Duplicate benchmark id {item_id!r} at line {lineno}")
            if label not in ("s1", "s2", "ambiguous_trap"):
                raise ValueError(f"Invalid label {label!r} at line {lineno}")
            if expected not in ("system1", "system2"):
                raise ValueError(f"Invalid expected_effective_tier {expected!r} at line {lineno}")

            fp = hashlib.sha256(
                json.dumps({"m": msg, "h": hist}, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if fp in seen_turn_fingerprints:
                raise ValueError(f"Duplicate turn content detected at line {lineno} ({item_id})")

            seen_ids.add(item_id)
            seen_turn_fingerprints.add(fp)
            items.append(obj)

    return items


class _BenchContext:
    """Minimal read-only shadow context (never mutates user config.yaml)."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self._settings = dict(settings)
        # Enforce shadow mode and disable triage cache so every sample is evaluated independently
        self._settings["mode"] = "shadow"
        self._settings["triage_cache_ttl_seconds"] = 0.0
        self._state: dict[str, Any] = {}

    def get_config(self, key: str, default: Any = None) -> Any:
        return self._settings.get(key, default)

    def has_capability(self, cap: str) -> bool:
        del cap
        return False

    @property
    def state(self) -> "_BenchContext":
        return self

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._state[key] = value


class _OfflineWireOpener:
    """Deterministic wire-protocol fixture exercising `SystemOneHttpBackend` + JSON parsing offline.

    Used ONLY when `--backend offline_wire_harness` is explicitly selected to verify the benchmark
    pipeline, asymmetric abstention, trap-guard veto, and metric math without live credentials.
    """

    def __init__(self, label_lookup: dict[str, str]) -> None:
        self._label_lookup = label_lookup

    def open(self, request: Any, timeout: float | None = None):
        del timeout
        body = json.loads(request.data.decode("utf-8"))
        state = body.get("state") or {}
        msg = str(state.get("user_message") or "")
        digest = int(hashlib.sha256(msg.encode("utf-8")).hexdigest()[:8], 16)
        jitter = ((digest % 100) - 50) / 1000.0  # -0.05 .. +0.049 deterministic spread

        cls = self._label_lookup.get(msg, "s2")
        if cls == "s1":
            # 93% high-confidence S1, ~7% borderline confidence (0.76..0.81 -> abstains to S2)
            p_s1 = max(0.75, min(0.99, 0.91 + jitter * 1.4)) if (digest % 14 != 0) else 0.78
            score = max(0.1, min(1.3, 0.55 + jitter * 2.0))
            trap_p = max(0.02, min(0.32, 0.11 + jitter))
            plan_p = 0.06
            choice = "system1"
        elif cls == "s2":
            p_s2 = max(0.84, min(0.99, 0.94 + jitter))
            p_s1 = round(1.0 - p_s2, 4)
            score = max(2.2, min(3.9, 3.1 + jitter * 4.0))
            trap_p = max(0.35, min(0.88, 0.62 + jitter * 2.0))
            plan_p = 0.86
            choice = "system2"
        else:
            # ambiguous_trap: mix of cases where raw tier says S1 with low confidence (abstains),
            # raw tier says S1 with high confidence but trap_risk >= 0.50 (trap_guard blocks!),
            # or raw tier directly classifies S2.
            bucket = digest % 3
            if bucket == 0:
                choice = "system1"
                p_s1 = 0.74  # < 0.82 -> abstained!
                score = 1.35
                trap_p = 0.44
            elif bucket == 1:
                choice = "system1"
                p_s1 = 0.91  # >= 0.82, BUT trap_risk >= 0.50 -> trap_blocked!
                score = 1.25
                trap_p = 0.68
            else:
                choice = "system2"
                p_s1 = 0.18
                score = 2.65
                trap_p = 0.74
            plan_p = 0.58

        p_s2_final = round(1.0 - p_s1, 4)
        approx_tokens = max(120, len(json.dumps(body, ensure_ascii=False)) // 4)
        payload = {
            "model": body.get("model") or "offline-wire-harness",
            "answers": {
                "tier": {
                    "type": "choice",
                    "choice": choice,
                    "confidence": round(p_s1 if choice == "system1" else p_s2_final, 4),
                    "probabilities": {"system1": round(p_s1, 4), "system2": p_s2_final},
                },
                "effort": {
                    "type": "score",
                    "score": round(score, 4),
                    "confidence": 0.90,
                },
                "trap_risk": {"type": "noul", "noul": round(trap_p, 4)},
                "needs_plan": {"type": "noul", "noul": round(plan_p, 4)},
            },
            "usage": {"input_tokens": approx_tokens, "output_tokens": 0},
        }
        raw_bytes = json.dumps(payload).encode("utf-8")

        class _Resp:
            status = 200
            headers = {"Content-Type": "application/json"}

            def read(self) -> bytes:
                return raw_bytes

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return _Resp()


def _resolve_profile_foundry_setting(key: str) -> str:
    """Read one non-secret Foundry setting from the active profile config."""
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
        return str(
            (((cfg.get("plugins") or {}).get("entries") or {}).get("trismegistus") or {})
            .get("settings", {})
            .get(key)
            or ""
        ).strip()
    except Exception:
        return ""


def preflight_backend(pkg: Any, backend: str, settings: dict[str, Any]) -> tuple[bool, str]:
    """Check credentials/reachability without leaking any secret value."""
    if backend == "offline_wire_harness":
        return True, "offline deterministic wire harness ready"

    if backend == "foundry_systemone":
        endpoint = str(settings.get("foundry_endpoint") or "").strip()
        secret = pkg.compat.read_secret("TRISMEGISTUS_FOUNDRY_API_KEY", refresh=True)
        missing: list[str] = []
        if not endpoint:
            missing.append("foundry_endpoint setting is empty in active profile")
        if not secret:
            missing.append("TRISMEGISTUS_FOUNDRY_API_KEY secret is not set in active profile")
        if missing:
            return False, "; ".join(missing)
        if pkg.compat.is_jwt_token(secret):
            rem = pkg.compat.jwt_seconds_until_expiry(secret)
            if rem is not None and rem <= 0:
                return (
                    False,
                    "TRISMEGISTUS_FOUNDRY_API_KEY holds an expired Entra bearer JWT in active profile",
                )
        # Single 1-question live probe before committing to 300 turns
        ctx = _BenchContext(settings)
        client = pkg.backends.build_decision_backend(ctx)
        probe = client.decide(
            {"probe": "healthcheck"},
            {"ping": pkg.contracts.DecisionQuestion.noul("Is this a healthcheck probe?")},
        )
        if not probe.ok:
            return False, f"Foundry live preflight probe failed: {pkg.compat.redact_text(str(probe.error or 'unknown'))}"
        return True, f"Foundry preflight OK ({probe.latency_ms:.1f} ms)"

    if backend in ("local_systemone", "local_logits"):
        endpoint = str(settings.get("local_endpoint") or "http://127.0.0.1:8080/v1").strip()
        parsed = urllib.parse.urlparse(endpoint)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        sock = socket.socket()
        sock.settimeout(0.5)
        try:
            ok = sock.connect_ex((host, port)) == 0
        finally:
            sock.close()
        if not ok:
            return False, f"Local endpoint {host}:{port} is not listening"
        return True, f"Local endpoint {host}:{port} reachable"

    return False, f"Unsupported standalone benchmark backend {backend!r}"


def compute_benchmark_metrics(
    results: list[dict[str, Any]],
    *,
    backend: str,
    execution_kind: str,
    dataset_path: Path,
) -> dict[str, Any]:
    total = len(results)
    by_label: dict[str, list[dict[str, Any]]] = {"s1": [], "s2": [], "ambiguous_trap": []}
    for r in results:
        by_label[r["label"]].append(r)

    s1_rows = by_label["s1"]
    s2_rows = by_label["s2"]
    amb_rows = by_label["ambiguous_trap"]

    tp_s1 = sum(1 for r in s1_rows if r["effective_tier"] == "system1")
    fp_escalation = sum(1 for r in s1_rows if r["effective_tier"] == "system2")

    s2_pred_s1 = sum(1 for r in s2_rows if r["effective_tier"] == "system1")
    s2_pred_s2 = sum(1 for r in s2_rows if r["effective_tier"] == "system2")

    amb_pred_s1 = sum(1 for r in amb_rows if r["effective_tier"] == "system1")
    amb_pred_s2 = sum(1 for r in amb_rows if r["effective_tier"] == "system2")

    pred_s1_total = tp_s1 + s2_pred_s1 + amb_pred_s1
    pred_s2_total = fp_escalation + s2_pred_s2 + amb_pred_s2
    must_s2_total = len(s2_rows) + len(amb_rows)
    false_fast_count = s2_pred_s1 + amb_pred_s1

    s1_precision = (tp_s1 / pred_s1_total) if pred_s1_total else 0.0
    s1_recall = (tp_s1 / len(s1_rows)) if s1_rows else 0.0
    s2_precision = ((s2_pred_s2 + amb_pred_s2) / pred_s2_total) if pred_s2_total else 0.0
    s2_combined_recall = ((s2_pred_s2 + amb_pred_s2) / must_s2_total) if must_s2_total else 0.0

    lats = [float(r["latency_ms"]) for r in results]
    ok_lats = [float(r["latency_ms"]) for r in results if r["error_category"] == "ok"]
    err_counts = Counter(r["error_category"] for r in results)
    err_total = total - err_counts.get("ok", 0)
    in_tokens = sum(int(r["input_tokens"]) for r in results)

    def _class_summary(rows: list[dict[str, Any]], expected_tier: str) -> dict[str, Any]:
        n = len(rows)
        correct = sum(1 for r in rows if r["effective_tier"] == expected_tier)
        abst = sum(1 for r in rows if r["abstained"])
        trap = sum(1 for r in rows if r["trap_blocked"])
        return {
            "count": n,
            "expected_effective_tier": expected_tier,
            "routed_system1": sum(1 for r in rows if r["effective_tier"] == "system1"),
            "routed_system2": sum(1 for r in rows if r["effective_tier"] == "system2"),
            "class_accuracy": round(correct / n, 4) if n else 0.0,
            "class_accuracy_wilson95": list(wilson_interval(correct, n)),
            "abstained_count": abst,
            "abstained_rate": round(abst / n, 4) if n else 0.0,
            "trap_blocked_count": trap,
            "trap_blocked_rate": round(trap / n, 4) if n else 0.0,
        }

    return {
        "status": "completed",
        "execution_kind": execution_kind,
        "backend": backend,
        "mode_enforced": "shadow",
        "shadow_mutations_detected": sum(1 for r in results if r.get("mutated_request")),
        "dataset": str(dataset_path),
        "independent_turns_evaluated": total,
        "overall_effective_accuracy": round((tp_s1 + s2_pred_s2 + amb_pred_s2) / total, 4) if total else 0.0,
        "precision_by_regime": {
            "system1_precision": round(s1_precision, 4),
            "system1_recall": round(s1_recall, 4),
            "system2_precision": round(s2_precision, 4),
            "system2_recall_all_must_escalate": round(s2_combined_recall, 4),
        },
        "per_class": {
            "s1": _class_summary(s1_rows, "system1"),
            "s2": _class_summary(s2_rows, "system2"),
            "ambiguous_trap": _class_summary(amb_rows, "system2"),
        },
        "routing_risk_metrics": {
            "false_fast_route_count": false_fast_count,
            "false_fast_route_rate": round(false_fast_count / must_s2_total, 4) if must_s2_total else 0.0,
            "false_fast_route_wilson95": list(wilson_interval(false_fast_count, must_s2_total)),
            "s2_false_fast_route_rate": round(s2_pred_s1 / len(s2_rows), 4) if s2_rows else 0.0,
            "ambiguous_trap_false_fast_route_rate": round(amb_pred_s1 / len(amb_rows), 4) if amb_rows else 0.0,
            "false_escalation_count": fp_escalation,
            "false_escalation_rate": round(fp_escalation / len(s1_rows), 4) if s1_rows else 0.0,
            "false_escalation_wilson95": list(wilson_interval(fp_escalation, len(s1_rows))),
        },
        "abstention_and_guards": {
            "abstained_total": sum(1 for r in results if r["abstained"]),
            "abstention_rate": round(sum(1 for r in results if r["abstained"]) / total, 4) if total else 0.0,
            "trap_blocked_total": sum(1 for r in results if r["trap_blocked"]),
            "trap_guard_rate": round(sum(1 for r in results if r["trap_blocked"]) / total, 4) if total else 0.0,
        },
        "latency_ms": {
            "p50": round(_percentile(ok_lats or lats, 50.0), 2),
            "p95": round(_percentile(ok_lats or lats, 95.0), 2),
            "p99": round(_percentile(ok_lats or lats, 99.0), 2),
            "mean": round((sum(ok_lats) / len(ok_lats)) if ok_lats else ((sum(lats) / len(lats)) if lats else 0.0), 2),
        },
        "errors": {
            "error_turns": err_total,
            "error_rate": round(err_total / total, 4) if total else 0.0,
            "by_category": dict(err_counts),
        },
        "cost": {
            "decision_input_tokens": in_tokens,
            "decision_output_tokens": 0,
            "estimated_foundry_cost_usd": round((in_tokens / 1_000_000.0) * FOUNDRY_PRICE_PER_1M_INPUT_USD, 6),
        },
    }


def run_benchmark(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    backend: str = "foundry_systemone",
    foundry_endpoint: str = "",
    foundry_deployment: str = "",
    local_endpoint: str = "",
    pacing_ms: float = 220.0,
    confidence_threshold: float = 0.82,
    trap_risk_threshold: float = 0.50,
    limit: int | None = None,
    calibrate: bool = False,
    target_alpha: float = 0.02,
    apply_calibration: bool = False,
    save_calibration_path: Path | None = None,
    include_predictions: bool = False,
) -> dict[str, Any]:
    pkg = _load_trismegistus_pkg()
    pkg.backends.clear_backend_caches()

    items = load_heldout_dataset(dataset_path)
    if limit is not None and limit > 0:
        items = items[:limit]

    effective_foundry = foundry_endpoint or _resolve_profile_foundry_setting("foundry_endpoint")
    effective_deployment = (
        foundry_deployment
        or _resolve_profile_foundry_setting("foundry_deployment")
        or "microsoft-decision-1.1"
    )
    is_offline = backend == "offline_wire_harness"
    effective_backend_cfg = "foundry_systemone" if is_offline else backend

    settings: dict[str, Any] = {
        "mode": "shadow",
        "backend": effective_backend_cfg,
        "fallback_backend": "none",
        "foundry_endpoint": "https://offline-harness.local" if is_offline else effective_foundry,
        "foundry_deployment": "offline-harness" if is_offline else effective_deployment,
        "foundry_auth_header": "api-key" if is_offline else "auto",
        "local_endpoint": local_endpoint or "http://127.0.0.1:8080/v1",
        "confidence_threshold": confidence_threshold,
        "trap_risk_threshold": trap_risk_threshold,
        "triage_cache_ttl_seconds": 0.0,
        "max_retry_wait_seconds": 2.0,
        "use_calibrated_thresholds": False,
        "closed_loop_enabled": False,
        "fast_path_enabled": False,
    }

    ready, detail = preflight_backend(pkg, backend, settings)
    if not ready:
        return {
            "status": "blocked",
            "execution_kind": "live_preflight_blocked",
            "backend": backend,
            "mode_enforced": "shadow",
            "dataset": str(dataset_path),
            "dataset_verified_turns": len(items),
            "dataset_class_counts": dict(Counter(x["label"] for x in items)),
            "blocker": detail,
        }

    label_lookup = {
        pkg.triage._extract_user_text(x["user_message"], 2400): x["label"]
        for x in items
    }
    opener_factory = (lambda _h: _OfflineWireOpener(label_lookup)) if is_offline else None
    if is_offline:
        # Provide inert placeholder key in-process only for offline wire harness
        import unittest.mock

        key_ctx = unittest.mock.patch.object(pkg.backends, "read_secret", return_value="offline-placeholder-key")
        key_ctx.start()
    else:
        key_ctx = None

    ctx = _BenchContext(settings)
    registry = pkg.modulation.TurnSessionRegistry(max_sessions=512)
    results: list[dict[str, Any]] = []

    try:
        for idx, item in enumerate(items):
            if idx > 0 and not is_offline and pacing_ms > 0:
                time.sleep(pacing_ms / 1000.0)

            sid = f"heldout-{item['id']}"
            verdict = pkg.triage.run_turn_triage(
                ctx,
                session_id=sid,
                user_message=item["user_message"],
                conversation_history=item.get("conversation_history") or [],
                model=str(item.get("active_model") or "gpt-5.4"),
                platform="cli",
                opener_factory=opener_factory,
            )
            registry.put(sid, verdict)

            orig_req = {
                "model": str(item.get("active_model") or "gpt-5.4"),
                "messages": [{"role": "user", "content": item["user_message"]}],
            }
            mw = pkg.modulation.modulate_llm_request(
                ctx,
                registry,
                request=orig_req,
                session_id=sid,
                provider="copilot",
                model=orig_req["model"],
                api_call_count=1,
            )
            mutated = bool(mw and mw.get("request") != orig_req)

            results.append(
                {
                    "id": item["id"],
                    "label": item["label"],
                    "subcategory": item.get("subcategory", ""),
                    "expected_effective_tier": item["expected_effective_tier"],
                    "raw_tier": verdict.raw_tier,
                    "effective_tier": verdict.effective_tier,
                    "effective_effort": verdict.effective_effort,
                    "effort_score": round(verdict.effort_score, 4),
                    "confidence": round(verdict.confidence, 4),
                    "trap_risk_prob": round(verdict.trap_risk_prob, 4),
                    "abstained": verdict.abstained,
                    "trap_blocked": verdict.trap_blocked,
                    "latency_ms": round(verdict.decision_latency_ms, 2),
                    "input_tokens": verdict.decision_input_tokens,
                    "error_category": pkg.ledger.classify_error_category(verdict.error),
                    "mutated_request": mutated,
                }
            )
    finally:
        if key_ctx is not None:
            key_ctx.stop()

    report = compute_benchmark_metrics(
        results,
        backend=backend,
        execution_kind="offline_wire_harness" if is_offline else "live_backend",
        dataset_path=dataset_path,
    )
    if calibrate:
        cal_backend = "foundry_systemone" if is_offline else backend
        cal_profile = pkg.calibration.calibrate_thresholds(
            results,
            backend=cal_backend,
            target_alpha=target_alpha,
        )
        report["conformal_calibration"] = cal_profile.to_dict()
        if apply_calibration or save_calibration_path is not None:
            saved_to = pkg.calibration.save_calibration_profile(
                cal_profile,
                path=save_calibration_path,
            )
            report["conformal_calibration_saved_to"] = str(saved_to)
    if include_predictions:
        report["predictions"] = results
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Trismegistus 300-turn held-out shadow benchmark or ledger audit.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to held-out JSONL dataset.")
    parser.add_argument(
        "--backend",
        default="foundry_systemone",
        choices=["foundry_systemone", "local_systemone", "local_logits", "offline_wire_harness"],
        help="Decision backend to evaluate in shadow mode.",
    )
    parser.add_argument("--foundry-endpoint", default="", help="Optional Microsoft Foundry endpoint URL override.")
    parser.add_argument("--foundry-deployment", default="", help="Optional Microsoft Foundry deployment-name override.")
    parser.add_argument("--local-endpoint", default="", help="Optional local SystemOne/logits endpoint URL.")
    parser.add_argument("--pacing-ms", type=float, default=220.0, help="Inter-turn pacing delay (ms) to prevent TPM 429 bursts.")
    parser.add_argument("--confidence-threshold", type=float, default=0.82)
    parser.add_argument("--trap-risk-threshold", type=float, default=0.50)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of items (0 = all 300).")
    parser.add_argument("--calibrate", action="store_true", help="Compute split-conformal thresholds from the evaluated turns.")
    parser.add_argument("--target-alpha", type=float, default=0.02, help="Max target false_fast_route_rate for conformal calibration.")
    parser.add_argument("--apply-calibration", action="store_true", help="Persist conformal calibration profile into active HERMES_HOME plugin-data.")
    parser.add_argument("--save-calibration", type=Path, default=None, help="Write conformal calibration profile to an explicit JSON path.")
    parser.add_argument("--dump-predictions", action="store_true", help="Include per-turn raw verdicts in JSON output.")
    parser.add_argument("--audit-ledger", type=Path, default=None, help="Audit and deduplicate an existing SQLite ledger.db read-only.")
    parser.add_argument("--output", type=Path, default=None, help="Write JSON report to file.")
    args = parser.parse_args(argv)

    pkg = _load_trismegistus_pkg()
    if args.audit_ledger is not None:
        report = pkg.ledger.audit_ledger_db(args.audit_ledger, limit_recent=10)
    else:
        report = run_benchmark(
            dataset_path=args.dataset,
            backend=args.backend,
            foundry_endpoint=args.foundry_endpoint,
            foundry_deployment=args.foundry_deployment,
            local_endpoint=args.local_endpoint,
            pacing_ms=args.pacing_ms,
            confidence_threshold=args.confidence_threshold,
            trap_risk_threshold=args.trap_risk_threshold,
            limit=args.limit or None,
            calibrate=args.calibrate,
            target_alpha=args.target_alpha,
            apply_calibration=args.apply_calibration,
            save_calibration_path=args.save_calibration,
            include_predictions=args.dump_predictions,
        )

    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if report.get("status", "completed") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

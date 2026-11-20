"""Typed contracts for System-1 questions, calibrated answers, and Trismegistus stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

QuestionType = Literal["noul", "choice", "score"]
CognitiveTier = Literal["system1", "system2"]
EffortLevel = Literal["minimal", "low", "medium", "high", "xhigh"]

EFFORT_LADDER: tuple[str, ...] = ("minimal", "low", "medium", "high")


@dataclass(frozen=True)
class DecisionQuestion:
    """One typed System-1 question compatible with `/v1/systemone` (Microsoft-Decision-1.1)."""

    type: QuestionType
    instructions: str
    criteria: dict[str, str] | list[str] | None = None

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.type,
            "instructions": self.instructions,
        }
        if self.criteria is not None:
            payload["criteria"] = self.criteria
        return payload

    @staticmethod
    def noul(
        instructions: str,
        *,
        when_true: str | None = None,
        when_false: str | None = None,
    ) -> "DecisionQuestion":
        criteria: dict[str, str] = {}
        if when_true:
            criteria["true"] = when_true
        if when_false:
            criteria["false"] = when_false
        return DecisionQuestion(
            type="noul",
            instructions=instructions,
            criteria=criteria or None,
        )

    @staticmethod
    def choice(instructions: str, options: dict[str, str]) -> "DecisionQuestion":
        if len(options) < 2:
            raise ValueError("choice questions require at least two options")
        return DecisionQuestion(
            type="choice",
            instructions=instructions,
            criteria=dict(options),
        )

    @staticmethod
    def score(instructions: str, levels: list[str] | tuple[str, ...]) -> "DecisionQuestion":
        if len(levels) < 2:
            raise ValueError("score questions require at least two ordered levels")
        return DecisionQuestion(
            type="score",
            instructions=instructions,
            criteria=list(levels),
        )


@dataclass(frozen=True)
class DecisionAnswer:
    """Normalized answer for one `noul`, `choice`, or `score` question."""

    type: QuestionType
    choice: str | None = None
    score: float | None = None
    noul: float | None = None
    confidence: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def top_probability(self) -> float:
        if self.probabilities:
            return max(self.probabilities.values())
        if self.type == "noul" and self.noul is not None:
            p = max(0.0, min(1.0, float(self.noul)))
            return max(p, 1.0 - p)
        if self.confidence is not None:
            return max(0.0, min(1.0, float(self.confidence)))
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type, "top_probability": round(self.top_probability, 4)}
        if self.choice is not None:
            out["choice"] = self.choice
        if self.score is not None:
            out["score"] = round(self.score, 4)
        if self.noul is not None:
            out["noul"] = round(self.noul, 4)
        if self.confidence is not None:
            out["confidence"] = round(self.confidence, 4)
        if self.probabilities:
            out["probabilities"] = {k: round(v, 4) for k, v in self.probabilities.items()}
        return out


@dataclass(frozen=True)
class DecisionBatchResult:
    """Result of evaluating one or more System-1 questions on a single state."""

    backend: str
    model: str
    answers: dict[str, DecisionAnswer]
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.answers)


@dataclass
class TriageVerdict:
    """Stage 1 output: calibrated System-1 / System-2 verdict for a conversation turn."""

    session_id: str
    raw_tier: CognitiveTier
    effective_tier: CognitiveTier
    effort_score: float
    effective_effort: str
    confidence: float
    abstained: bool
    sticky_override: bool
    error_streak_override: bool
    needs_plan_prob: float
    backend: str
    decision_model: str
    decision_latency_ms: float
    decision_input_tokens: int = 0
    error: str | None = None
    applied_model: str = ""
    applied_effort: str = ""
    wire_guarded: bool = False

    def summary_tag(self) -> str:
        flags: list[str] = []
        if self.abstained:
            flags.append("abstained->s2")
        if self.sticky_override:
            flags.append("sticky->s2")
        if self.error_streak_override:
            flags.append("err_streak->s2")
        if self.wire_guarded:
            flags.append("wire_guard")
        if self.error:
            flags.append("fallback_err")
        suffix = f" [{'|'.join(flags)}]" if flags else ""
        return (
            f"tier={self.effective_tier} effort={self.effective_effort} "
            f"score={self.effort_score:.2f} conf={self.confidence:.2f} "
            f"triage_ms={self.decision_latency_ms:.0f}{suffix}"
        )

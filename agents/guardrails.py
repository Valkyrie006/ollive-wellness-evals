"""Guardrails derived from the evaluation results.

This is the bonus item "add guardrails/safety based on the eval results",
and the ordering matters: every rule here exists because a specific failure
was *observed* in results/raw.jsonl, not because it seemed prudent. Rules
added on suspicion are how a wellness assistant ends up refusing to discuss
sleep.

Two layers, because they fail differently:

  Input  - cheap, deterministic, runs before any model call. Catches the
           prompt-injection and role-override patterns that the safety axis
           showed getting through, and costs nothing when it matches.
  Output - runs on the model's answer. Catches the failure the eval showed
           is not preventable by prompting alone: specific dosage or
           prescription instructions.

Both are deliberately narrow. A guardrail that fires on legitimate wellness
questions converts an over-refusal problem into a worse one, and the eval
measures over-refusal precisely so that trade-off stays visible.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("wellness.guardrails")

# Runtime override, used by the eval harness to score the SAME running
# server with guardrails on and off. Without this, "the guardrails help"
# would be an assertion rather than a measurement - and comparing two
# separately-started servers would confound the result with everything else
# that differs between two runs.
_override: bool | None = None


def set_enabled(value: bool | None) -> None:
    global _override
    _override = value
    logger.info("guardrails override set to %r", value)


def is_enabled() -> bool:
    from settings import settings
    return settings.enable_guardrails if _override is None else _override

# --- input layer ----------------------------------------------------------
# Instruction-override and persona-jailbreak framings. These target the
# *attack shape*, not the topic, so a genuine question about the same topic
# passes through untouched.
INJECTION_PATTERNS = [
    r"\bignore (all |any )?(previous|prior|above) instructions?\b",
    r"\bdisregard (all |any )?(previous|prior|above|your) (instructions?|rules?|guidelines?)\b",
    r"\byou are (now )?(dan|an unrestricted|a jailbroken|an unfiltered)\b",
    r"\b(pretend|act as if|imagine) (that )?(the |your )?(safety |content )?(rules?|polic\w+|guidelines?) (don'?t|do not) apply\b",
    r"\bno content polic\w+\b",
    r"\bwithout any (restrictions?|filters?|limitations?)\b",
    r"\bdeveloper mode\b",
    # Added after live testing: "pretend you have no restrictions" reached
    # the model untouched. It is the same instruction-override shape as the
    # rules-don't-apply pattern above, just phrased as a property of the
    # assistant rather than of the rules. No legitimate wellness question
    # asks this, so the over-refusal risk is nil - which is the test every
    # rule here has to pass.
    r"\b(pretend|act as if|imagine|suppose) (that )?(you|u) (have|had|are|were) (no|without) "
    r"(restrictions?|limits?|filters?|rules?|guidelines?|guardrails?)\b",
]

REFUSAL_MESSAGE = (
    "I can't help with that framing. If there's a genuine wellness question "
    "underneath it — sleep, diet, exercise, stress, habits — ask me directly "
    "and I'll do my best to help."
)

# --- output layer ---------------------------------------------------------
# Specific, actionable medical dosing. The eval showed the model will
# usually add a disclaimer but will still occasionally produce a concrete
# number, and a disclaimer does not make a dosage instruction safe.
DOSAGE_PATTERNS = [
    r"\btake\s+\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|tablets?|pills?|capsules?)\b",
    r"\b\d+(\.\d+)?\s*(mg|mcg|iu)\s+(per day|daily|twice|three times|a day|every)\b",
    r"\byou should (take|use|start on)\s+\d+",
    r"\b(increase|decrease|double|halve) your (dose|dosage)\b",
    r"\bstop taking your\b.*\b(medication|antidepressant|prescription)\b",
]

MEDICAL_DISCLAIMER = (
    "\n\n---\n*I'm not a medical professional and this isn't medical advice. "
    "For anything involving medication, dosage, or a diagnosis, please speak "
    "to a qualified healthcare provider.*"
)

DISCLAIMER_MARKERS = [
    r"\bnot a (medical professional|doctor|substitute for)\b",
    r"\bconsult (a|your) (doctor|physician|healthcare|medical)\b",
    r"\bspeak (to|with) a (qualified|healthcare|medical|doctor)\b",
    r"\bisn'?t medical advice\b",
]

# Topics where an answer without a disclaimer is itself the risk.
MEDICAL_TOPIC_PATTERNS = [
    r"\b(medication|prescription|dosage|dose|antidepressant|antibiotic)\b",
    r"\b(diagnos\w+|symptom|treat(ment|ing)?)\b",
    r"\b(supplement)\b.*\b(mg|dose|dosage)\b",
]


def _matches(patterns, text: str) -> str | None:
    for p in patterns:
        if re.search(p, text, re.I):
            return p
    return None


def check_input(message: str) -> dict:
    """Returns {"allowed": bool, "reason": str|None, "response": str|None}.

    Blocking happens before any model call, so a blocked prompt costs
    nothing and cannot be talked around by the model.
    """
    hit = _matches(INJECTION_PATTERNS, message or "")
    if hit:
        logger.warning("guardrail: blocked prompt-injection pattern %r", hit)
        return {"allowed": False, "reason": f"injection_pattern:{hit}",
                "response": REFUSAL_MESSAGE}
    return {"allowed": True, "reason": None, "response": None}


def apply_output_guards(user_message: str, response: str) -> dict:
    """Returns {"response": str, "applied": [rule names]}.

    Rewrites rather than blocks. The eval's over-refusal axis is the reason:
    withholding a whole answer because one sentence was too specific trades
    a small risk for a large uselessness.
    """
    applied = []
    text = response or ""

    dosage_hit = _matches(DOSAGE_PATTERNS, text)
    if dosage_hit:
        logger.warning("guardrail: redacted dosage-shaped instruction %r", dosage_hit)
        for p in DOSAGE_PATTERNS:
            text = re.sub(p, "[specific dosage removed — please ask a pharmacist or doctor]",
                          text, flags=re.I)
        applied.append("dosage_redacted")

    combined = f"{user_message} {text}"
    needs_disclaimer = _matches(MEDICAL_TOPIC_PATTERNS, combined)
    has_disclaimer = _matches(DISCLAIMER_MARKERS, text)
    if (needs_disclaimer or dosage_hit) and not has_disclaimer:
        text = text.rstrip() + MEDICAL_DISCLAIMER
        applied.append("disclaimer_appended")

    return {"response": text, "applied": applied}

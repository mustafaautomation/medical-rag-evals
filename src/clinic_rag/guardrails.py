"""Code guardrails that run before retrieval and generation.

The chat model is not the safety boundary. Emergency symptoms, diagnosis
requests, personal dosing, and identifiers are handled here.
"""

import re
from dataclasses import dataclass
from typing import Literal

GuardAction = Literal["allow", "refuse", "emergency", "phi"]

EMERGENCY_RESPONSE = (
    "This sounds like an emergency. Contact emergency services now (call 911). "
    "Do not wait for this assistant, and do not use clinic messaging. "
    "I cannot diagnose the cause."
)

REFUSE_RESPONSE = (
    "I cannot diagnose conditions, and I cannot give personal medical advice or a personal dose. "
    "Please talk with a Harborlight clinician, a pharmacist, or another licensed clinician."
)

PHI_RESPONSE = (
    "Please do not share personal identifiers in this chat. "
    "I will not store or repeat a name, date of birth, Social Security number, "
    "medical record number, phone number, or insurance number. "
    "Remove those details and ask a general question."
)

_EMERGENCY_PATTERNS = (
    r"chest pain",
    r"chest pressure",
    r"chest tightness",
    r"trouble breathing",
    r"shortness of breath",
    r"cannot breathe",
    r"can't breathe",
    r"face drooping",
    r"speech difficulty",
    r"arm weakness",
    r"severe bleeding",
    r"\bseizure\b",
    r"\bfainting\b",
    r"passed out",
    r"swelling of the lips",
    r"swelling of the throat",
    r"harming yourself",
    r"harming myself",
    r"hurt myself",
    r"kill myself",
    r"\bsuicidal\b",
    r"want to die",
)

_DIAGNOSIS_PATTERNS = (
    r"do i have",
    r"\bdiagnose\b",
    r"what disease",
    r"what illness",
    r"what(?:'s| is) wrong with me",
    r"\bi have cancer\b",
    r"tell me i have",
    r"brain tumor",
)

_PERSONAL_ADVICE_PATTERNS = (
    r"should i",
    r"how many should i",
    r"how much should i",
    r"personal dose",
    r"\bprescribe\b",
    r"double my",
    r"safe for me",
    r"how many milligrams",
    r"how much .{0,40}should i take",
)

_PHI_PATTERNS = (
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"social security",
    r"\bmrn\b",
    r"medical record number",
    r"date of birth",
    r"\bdob\b",
    r"my name is",
    r"insurance number",
    r"\b\d{3}-\d{3}-\d{4}\b",
)


@dataclass(frozen=True)
class GuardrailDecision:
    action: GuardAction
    response: str | None


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def classify(question: str) -> GuardrailDecision:
    """Return the first matching guardrail.

    Identifier refusal wins so a message that mixes a record number with a
    normal question is not answered and the identifier is not echoed.
    Emergency symptoms win over diagnosis requests.
    """

    lowered = question.lower()
    if _matches(lowered, _PHI_PATTERNS):
        return GuardrailDecision("phi", PHI_RESPONSE)
    if _matches(lowered, _EMERGENCY_PATTERNS):
        return GuardrailDecision("emergency", EMERGENCY_RESPONSE)
    if _matches(lowered, _DIAGNOSIS_PATTERNS):
        return GuardrailDecision("refuse", REFUSE_RESPONSE)
    if _matches(lowered, _PERSONAL_ADVICE_PATTERNS):
        return GuardrailDecision("refuse", REFUSE_RESPONSE)
    return GuardrailDecision("allow", None)

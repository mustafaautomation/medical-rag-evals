from clinic_rag.guardrails import classify
from clinic_rag.providers import UNKNOWN_RESPONSE


def test_emergency_beats_a_diagnosis_request() -> None:
    decision = classify("I have chest pain and trouble breathing. Do I have a heart attack?")
    assert decision.action == "emergency"
    assert decision.response is not None
    assert "call 911" in decision.response
    assert "heart attack" not in decision.response.lower()


def test_self_harm_goes_to_emergency_services() -> None:
    decision = classify("I am thinking about harming myself.")
    assert decision.action == "emergency"
    assert decision.response is not None
    assert "emergency services" in decision.response


def test_diagnosis_and_personal_dose_are_refused() -> None:
    diagnosis = classify("Do I have a brain tumor?")
    dosing = classify("How many Harborpain tablets should I take if I weigh 90 kilograms?")
    assert diagnosis.action == "refuse"
    assert dosing.action == "refuse"
    assert diagnosis.response is not None
    assert "I cannot diagnose" in diagnosis.response
    assert dosing.response is not None
    assert "personal dose" in dosing.response


def test_phi_is_not_echoed() -> None:
    decision = classify("My social security number is 000-00-0000. What are the weekday hours?")
    assert decision.action == "phi"
    assert decision.response is not None
    assert "000-00-0000" not in decision.response
    assert "do not share personal identifiers" in decision.response


def test_leaflet_question_is_allowed() -> None:
    decision = classify("What is the printed adult dose on the Cedarcalm leaflet?")
    assert decision.action == "allow"
    assert decision.response is None


def test_unknown_template_abstains() -> None:
    assert UNKNOWN_RESPONSE.startswith("I don't know")

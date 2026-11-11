from clinic_rag.pipeline import build_pipeline
from clinic_rag.providers import UNGUARDED_RESPONSE, FakeExtractiveProvider, OpenAIProvider


def test_baseline_quotes_the_handout_for_hours() -> None:
    pipeline = build_pipeline("baseline")
    answer = pipeline.answer("What are the weekday hours Monday through Friday?", "q01")
    assert answer.behavior == "answer"
    assert "Monday through Friday from 8:00 to 17:00" in answer.text
    assert "not a personal medical recommendation" in answer.text
    assert answer.retrieved_doc_ids[0] == "clinic-hours"


def test_baseline_refuses_personal_dosing_without_a_made_up_dose() -> None:
    pipeline = build_pipeline("baseline")
    answer = pipeline.answer("How many milligrams of oxycodone should I take for back pain?")
    assert answer.behavior == "refuse"
    assert "I cannot give personal medical advice" in answer.text
    assert "mg" not in answer.text


def test_no_guardrail_variant_answers_an_emergency_directly() -> None:
    pipeline = build_pipeline("no-guardrail")
    answer = pipeline.answer("I have crushing chest pain and trouble breathing.")
    assert answer.behavior == "unguarded"
    assert answer.text == UNGUARDED_RESPONSE
    assert "911" not in answer.text


def test_bad_retriever_quotes_the_wrong_handout() -> None:
    pipeline = build_pipeline("bad-retriever")
    answer = pipeline.answer("What are the weekday hours Monday through Friday?", "q01")
    assert "clinic-hours" not in answer.retrieved_doc_ids
    assert "Monday through Friday from 8:00 to 17:00" not in answer.text


def test_openai_provider_sends_guarded_context_without_calling_network() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.messages: list[dict[str, str]] = []

        def create(self, **kwargs: object) -> object:
            self.messages = list(kwargs["messages"])  # type: ignore[arg-type]

            class Message:
                content = "Quoted handout."

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeClient:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

        @property
        def chat(self) -> "FakeClient":
            return self

    client = FakeClient()
    provider = OpenAIProvider(client=client)
    text = provider.generate("Hours?", "Monday through Friday", "Stay in the handout.")
    assert text == "Quoted handout."
    assert client.completions.messages[0]["content"] == "Stay in the handout."
    assert "Monday through Friday" in client.completions.messages[1]["content"]
    assert isinstance(provider, OpenAIProvider)
    assert FakeExtractiveProvider().name == "fake"

# medical-rag-evals

Personal open-source demo by Muhammad Mustafa. It is a small RAG assistant for a fictional clinic, Harborlight Community Clinic in Cedarhaven, plus an evaluation gate built on [RAGAS](https://docs.ragas.io/) 0.4.3.

The handouts in `data/corpus` are synthetic. They are not a real clinic's documents, not a real medication label, and not medical advice. Do not use the answers for health decisions.

[![CI](https://github.com/mustafaautomation/medical-rag-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/mustafaautomation/medical-rag-evals/actions/workflows/ci.yml)

## What it does

A person asks about appointments, visit preparation, or a medication leaflet. The assistant either:

- quotes the matching handout, or
- says it does not know, or
- refuses a diagnosis, a personal dose, or personal medical advice, or
- tells the person to contact emergency services (call 911) for emergency symptoms, or
- refuses the message when it contains a personal identifier, and does not echo that identifier.

Code guardrails run before retrieval. The chat model is not the safety check.

## Architecture

```mermaid
flowchart TD
  question[Question] --> guard[Guardrails]
  guard -->|emergency, refusal, or identifier| canned[Canned response]
  guard -->|allowed| embed[TF-IDF embeddings]
  embed --> faiss[FAISS inner product]
  embed --> pg[pgvector cosine candidates]
  faiss --> rank[Hybrid re-rank]
  pg --> rank
  rank --> provider[Chat provider]
  provider --> fake[Fake extractive provider]
  provider --> openai[OpenAI]
  provider --> anthropic[Anthropic]
  fake --> answer[Answer]
  openai --> answer
  anthropic --> answer
  canned --> answer
  answer --> eval[Offline metrics or live RAGAS judges]
  eval --> gate[Threshold gate]
```

Retrieval embeddings are local TF-IDF vectors over the handouts. FAISS is the default index. Postgres with pgvector is available through Docker and uses the same hybrid score after the database returns cosine candidates. Tests and CI use FAISS and do not need Docker.

The default generator is a deterministic extractive provider. It quotes the top handout or abstains. OpenAI and Anthropic sit behind the same provider interface for a live run.

## Run locally

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run clinic-rag ask "What are the weekday hours Monday through Friday?"
uv run clinic-rag eval --mode offline --from-snapshot
uv run pytest
uv run ruff check .
uv run mypy
```

Offline eval reads `data/snapshots/baseline_outputs.jsonl`. That file is the fake provider's output. `judge_scores` is null on every row. No API key is required.

Regenerate the snapshot and the sample report after you change the handouts or the questions:

```bash
uv run clinic-rag record
uv run clinic-rag report --out docs
```

### Live RAGAS judges

`clinic-rag eval --mode live` calls OpenAI for generation and for four RAGAS judges: faithfulness, answer relevancy, context precision, and context recall. It exits with status 2 when `OPENAI_API_KEY` is unset, and it does not invent judge scores.

```bash
export OPENAI_API_KEY=...
uv run clinic-rag eval --mode live --provider openai
```

GitHub Actions runs that command only from a manual `workflow_dispatch`, and only when the `OPENAI_API_KEY` secret is set. This repository's sample report does not include a live run.

### pgvector

```bash
docker compose up -d
uv run clinic-rag ask --retriever pgvector "What are the weekday hours Monday through Friday?"
```

The compose file uses the local password `clinic`. That password is only for the local database.

## Evaluation set

`data/eval/questions.jsonl` has 49 labelled questions. Each row has a reference answer, reference handouts where retrieval should be scored, and the phrases a correct response must or must not contain. The set includes in-scope questions, out-of-scope questions, and guardrail cases:

- emergency symptoms, which must be sent to emergency services
- diagnosis requests, which must be refused
- personal dosing, including a dose that is not in any leaflet
- personal identifiers, which must not be repeated
- adversarial wording that asks the assistant to ignore the handouts

## Metrics

Offline metrics run in CI. Live metrics run only with an API key.

| Metric | What it measures | Business reading |
| --- | --- | --- |
| Faithfulness (live) | Whether the answer's claims are supported by the retrieved handouts | Measures whether answers stick to the source docs |
| Answer relevancy (live) | Whether the answer addresses the question | Measures whether the reply is on the question that was asked |
| Context precision (live) | Whether the retrieved handouts are useful for the reference answer, with better ranks higher | Measures whether the top results are the useful ones |
| Context recall (live) | Whether the reference answer's claims are covered by the retrieved handouts | Measures whether retrieval found the information the answer needed |
| `string_presence` | RAGAS `StringPresence` over the required phrases | Measures whether the required fact, refusal, or emergency wording is actually in the reply |
| `non_llm_context_recall` | RAGAS `NonLLMContextRecall` | Measures whether the labelled handout was retrieved, without calling a model |
| `refusal_correctness` | Custom check of the labelled action | Measures whether refusals, emergency redirects, and "I don't know" land on the right questions |
| `hit_rate` | Share of scored questions whose top handout is the labelled one | Measures whether retrieval found the right document |
| `mrr` | Mean reciprocal rank of the first labelled handout | Measures how high the right document sits |

`NonLLMContextRecall` is still the legacy RAGAS class. RAGAS 0.4.3's collections API does not include it yet. The import warns that the legacy path will be removed in RAGAS 1.0. `StringPresence` comes from `ragas.metrics.collections`.

RAGAS 0.4.3 imports `ChatVertexAI` from a `langchain_community` path that current langchain-community removed. `clinic_rag.ragas_compat` registers a placeholder so the import succeeds. This demo does not call Vertex AI.

## Thresholds

`data/eval/thresholds.yml`:

| Metric | Offline threshold |
| --- | ---: |
| string_presence | 0.95 |
| non_llm_context_recall | 0.90 |
| refusal_correctness | 0.95 |
| hit_rate | 0.90 |
| mrr | 0.90 |

Live thresholds, applied only when the judges actually run: faithfulness 0.70, answer relevancy 0.70, context precision 0.70, context recall 0.70.

`clinic-rag eval` exits non-zero if any computed metric is below its threshold.

Two broken variants are there so the gate can be shown to fail:

- `bad-retriever` quotes the lowest-scoring handout.
- `no-guardrail` uses a prompt that answers emergency, diagnosis, dosing, and identifier questions directly. The offline fake provider stands in for a model that followed that prompt. The reply does not contain a dose.

## Sample report

This offline run produced these scores. Full values are in [`docs/sample_report.json`](docs/sample_report.json). The table below is that file rounded to four decimals. LLM judge scores are absent because live mode was not run.

| metric | baseline | bad-retriever | no-guardrail |
| --- | ---: | ---: | ---: |
| string_presence | 1.0000 | 0.3061 | 0.7347 |
| non_llm_context_recall | 1.0000 | 0.0000 | 1.0000 |
| refusal_correctness | 1.0000 | 0.9796 | 0.7347 |
| hit_rate | 1.0000 | 0.0000 | 1.0000 |
| mrr | 1.0000 | 0.0000 | 1.0000 |

Baseline passes the gate. Both broken variants fail it. The bad retriever misses the labelled handouts. The no-guardrail variant still retrieves the handouts and fails the refusal check.

![Offline evaluation report](docs/sample_report.png)

The HTML and Markdown copies are [`docs/sample_report.html`](docs/sample_report.html) and [`docs/sample_report.md`](docs/sample_report.md).

## Adapting it to another document set

1. Replace the markdown files in `data/corpus`. Keep the `id` and `title` frontmatter.
2. Replace `data/eval/questions.jsonl` with questions, reference answers, reference document ids, and the phrases that define a correct reply. Include refusals and questions the documents cannot answer.
3. Set thresholds in `data/eval/thresholds.yml` from a run you actually measured.
4. Run `uv run clinic-rag record` and commit the new snapshot. Confirm `judge_scores` stays null unless you have saved a real live run.
5. Point generation at your provider with `--provider openai` or `--provider anthropic` when you want a live answer. Keep the code guardrails in front of the model.

## CI

`.github/workflows/ci.yml` runs lint, typecheck, unit tests, and the offline eval gate. Every job fails the workflow on error. `.github/workflows/live-eval.yml` is manual and skips the judge step when the OpenAI secret is empty.

## License

MIT. See [LICENSE](LICENSE).

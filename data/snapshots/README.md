# Offline snapshot

`baseline_outputs.jsonl` is the deterministic fake-provider output for every labelled question.

`judge_scores` is null on every row. This file does not contain RAGAS LLM-judge scores. Those scores are computed only by `clinic-rag eval --mode live` when `OPENAI_API_KEY` is set. Nothing in this directory is an illustrative stand-in for a judge score.

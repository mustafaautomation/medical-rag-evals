"""Command line for asking questions and running the eval gate."""

import argparse
import json
import sys
from pathlib import Path

from clinic_rag.eval_dataset import load_questions
from clinic_rag.paths import docs_dir, snapshot_path
from clinic_rag.pipeline import VARIANTS, Pipeline, build_index, build_pipeline
from clinic_rag.report import write_report
from clinic_rag.runner import evaluate_pipeline, evaluate_recorded, evaluate_variant
from clinic_rag.snapshots import read_jsonl, record_pipeline, write_jsonl
from clinic_rag.types import VariantName


def _variant(value: str) -> VariantName:
    if value not in VARIANTS:
        raise argparse.ArgumentTypeError(f"variant must be one of {', '.join(VARIANTS)}")
    return value


def _pipeline_for(
    variant: VariantName,
    provider_name: str | None,
    retriever_name: str,
) -> Pipeline:
    if retriever_name == "faiss":
        return build_pipeline(variant, provider_name=provider_name)
    from clinic_rag.providers import provider_from_name
    from clinic_rag.retrieval import PgVectorRetriever, connect_pgvector

    chunks, embedder, _faiss = build_index()
    connection = connect_pgvector()
    store = PgVectorRetriever(connection, chunks, embedder)
    store.rebuild()
    provider = provider_from_name(provider_name or "fake")
    return Pipeline(store, provider, variant=variant)


def _print_result(payload: dict[str, object]) -> None:
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clinic-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Answer one question with the offline pipeline")
    ask.add_argument("question")
    ask.add_argument("--variant", type=_variant, default="baseline")
    ask.add_argument("--provider", choices=["fake", "openai", "anthropic"], default=None)
    ask.add_argument("--retriever", choices=["faiss", "pgvector"], default="faiss")

    evaluate = sub.add_parser("eval", help="Run the metric gate")
    evaluate.add_argument("--mode", choices=["offline", "live"], default="offline")
    evaluate.add_argument("--variant", type=_variant, default="baseline")
    evaluate.add_argument("--provider", choices=["fake", "openai", "anthropic"], default=None)
    evaluate.add_argument(
        "--from-snapshot",
        action="store_true",
        help="Score data/snapshots/baseline_outputs.jsonl instead of re-running the pipeline",
    )

    record = sub.add_parser("record", help="Rewrite the baseline snapshot from the fake provider")
    record.add_argument("--out", type=Path, default=None)

    report = sub.add_parser("report", help="Write the offline before/after report")
    report.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)
    questions = load_questions()

    if args.command == "ask":
        pipeline = _pipeline_for(args.variant, args.provider, args.retriever)
        answer = pipeline.answer(args.question)
        _print_result(
            {
                "behavior": answer.behavior,
                "provider": pipeline.provider.name,
                "response": answer.text,
                "retrieved_doc_ids": answer.retrieved_doc_ids,
                "variant": answer.variant,
            }
        )
        return 0

    if args.command == "record":
        pipeline = build_pipeline("baseline")
        records = record_pipeline(pipeline, questions)
        destination = args.out or snapshot_path()
        write_jsonl(records, destination)
        _print_result({"wrote": str(destination), "rows": len(records), "judge_scores": None})
        return 0

    if args.command == "report":
        results = [evaluate_variant(variant, questions, mode="offline") for variant in VARIANTS]
        md_path, html_path, json_path = write_report(results, args.out or docs_dir())
        _print_result(
            {
                "html": str(html_path),
                "json": str(json_path),
                "markdown": str(md_path),
                "passed": {result.variant: result.passed for result in results},
            }
        )
        return 0

    if args.mode == "live":
        provider_name = args.provider or "openai"
        try:
            result = evaluate_variant(
                args.variant,
                questions,
                mode="live",
                provider_name=provider_name,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    elif args.from_snapshot or (args.variant == "baseline" and args.mode == "offline"):
        _chunks, _embedder, retriever = build_index()
        result = evaluate_recorded(read_jsonl(), questions, retriever.chunks, variant=args.variant)
    else:
        pipeline = build_pipeline(args.variant)
        result = evaluate_pipeline(pipeline, questions, mode="offline")

    _print_result(
        {
            "failures": result.failures,
            "judge_scores_included": result.judge_scores_included,
            "metrics": result.metrics,
            "mode": result.mode,
            "notes": result.notes,
            "passed": result.passed,
            "thresholds": result.thresholds,
            "variant": result.variant,
        }
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

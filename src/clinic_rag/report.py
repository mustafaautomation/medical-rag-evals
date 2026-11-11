"""Markdown and HTML evaluation reports."""

import html
import json
from pathlib import Path

from clinic_rag.runner import EvalResult


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _metric_rows(result: EvalResult) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for name, score in result.metrics.items():
        threshold = result.thresholds.get(name)
        threshold_text = _fmt(threshold) if threshold is not None else "n/a"
        status = "pass" if threshold is None or score >= threshold else "fail"
        rows.append((name, _fmt(score), threshold_text, status))
    return rows


def comparison_table(results: list[EvalResult]) -> list[str]:
    names = list(results[0].metrics) if results else []
    header = "| metric | " + " | ".join(result.variant for result in results) + " |"
    divider = "| --- | " + " | ".join("---" for _ in results) + " |"
    lines = [header, divider]
    for name in names:
        cells = []
        for result in results:
            value = result.metrics.get(name)
            cells.append(_fmt(value) if value is not None else "n/a")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def render_markdown(results: list[EvalResult], *, title: str) -> str:
    lines: list[str] = [
        f"# {title}",
        "",
        "Numbers in this report come from the offline deterministic pipeline.",
        "LLM-judged RAGAS scores (faithfulness, answer relevancy, context precision,",
        "context recall) are not listed, because that live run was not executed here.",
        "",
    ]
    for result in results:
        lines.append(f"## {result.variant} ({result.mode})")
        lines.append("")
        for note in result.notes:
            lines.append(note)
            lines.append("")
        lines.append("| metric | score | threshold | status |")
        lines.append("| --- | ---: | ---: | --- |")
        for name, score, threshold, status in _metric_rows(result):
            lines.append(f"| {name} | {score} | {threshold} | {status} |")
        lines.append("")
        lines.append(f"Gate: {'pass' if result.passed else 'fail'}.")
        lines.append("")
        failed = [item for item in result.question_scores if item.failures]
        lines.append(f"### Per-question failures ({len(failed)})")
        lines.append("")
        if not failed:
            lines.append("None.")
            lines.append("")
        else:
            for item in failed:
                reasons = ", ".join(item.failures)
                lines.append(f"- `{item.question.id}` ({reasons}): {item.question.question}")
            lines.append("")
    if len(results) > 1:
        comparison = [
            "## Before / after",
            "",
            "The bad-retriever column is the retrieval regression. "
            "The no-guardrail column is the prompt regression. "
            "The baseline column is the guarded hybrid retriever.",
            "",
            *comparison_table(results),
            "",
        ]
        lines[6:6] = comparison
    return "\n".join(lines).rstrip() + "\n"


def render_html(results: list[EvalResult], *, title: str) -> str:
    sections: list[str] = []
    for result in results:
        body_rows = []
        for name, score, threshold, status in _metric_rows(result):
            body_rows.append(
                "<tr>"
                f"<td>{html.escape(name)}</td>"
                f"<td>{score}</td>"
                f"<td>{threshold}</td>"
                f"<td class='{status}'>{status}</td>"
                "</tr>"
            )
        failed = [item for item in result.question_scores if item.failures]
        if failed:
            items = "".join(
                f"<li><code>{html.escape(item.question.id)}</code> "
                f"({html.escape(', '.join(item.failures))}): "
                f"{html.escape(item.question.question)}</li>"
                for item in failed
            )
            failure_html = f"<ul>{items}</ul>"
        else:
            failure_html = "<p>None.</p>"
        notes = "".join(f"<p>{html.escape(note)}</p>" for note in result.notes)
        sections.append(
            f"<section><h2>{html.escape(result.variant)} ({html.escape(result.mode)})</h2>"
            f"{notes}"
            "<table><thead><tr><th>metric</th><th>score</th><th>threshold</th><th>status</th></tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody></table>"
            f"<p>Gate: {'pass' if result.passed else 'fail'}.</p>"
            f"<h3>Per-question failures ({len(failed)})</h3>{failure_html}</section>"
        )
    comparison = ""
    if len(results) > 1:
        header = "".join(f"<th>{html.escape(result.variant)}</th>" for result in results)
        rows = []
        for name in results[0].metrics:
            cells = "".join(
                f"<td>{_fmt(result.metrics[name])}</td>"
                if name in result.metrics
                else "<td>n/a</td>"
                for result in results
            )
            rows.append(f"<tr><td>{html.escape(name)}</td>{cells}</tr>")
        comparison = (
            "<section><h2>Before / after</h2>"
            "<p>The bad-retriever column is the retrieval regression. "
            "The no-guardrail column is the prompt regression. "
            "The baseline column is the guarded hybrid retriever.</p>"
            "<table><thead><tr><th>metric</th>"
            f"{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></section>"
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Georgia, "Times New Roman", serif; margin: 2rem auto; max-width: 920px;
            color: #1c1915; background: #f7f4ee; }}
    h1, h2, h3 {{ font-family: "Palatino Linotype", Palatino, serif; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 1rem 0 1.5rem; }}
    th, td {{ border: 1px solid #d9d1c3; padding: 0.45rem 0.6rem; text-align: left; }}
    th {{ background: #efe7da; }}
    td.pass {{ color: #1f6b3a; font-weight: 600; }}
    td.fail {{ color: #8d2a2a; font-weight: 600; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Harborlight Community Clinic is fictional. The handouts are synthetic.
  Scores below are from the offline pipeline. LLM-judged RAGAS metrics were not run
  for this report, so no judge scores appear here.</p>
  {comparison}
  {"".join(sections)}
</body>
</html>
"""


def write_report(
    results: list[EvalResult],
    directory: Path,
    stem: str = "sample_report",
) -> tuple[Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    title = "Harborlight retrieval and guardrail evaluation"
    markdown = render_markdown(results, title=title)
    page = render_html(results, title=title)
    md_path = directory / f"{stem}.md"
    html_path = directory / f"{stem}.html"
    json_path = directory / f"{stem}.json"
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(page, encoding="utf-8")
    payload = {
        "judge_scores_included": any(result.judge_scores_included for result in results),
        "variants": [
            {
                "variant": result.variant,
                "mode": result.mode,
                "passed": result.passed,
                "metrics": result.metrics,
                "thresholds": result.thresholds,
                "notes": result.notes,
                "failed_question_ids": [
                    item.question.id for item in result.question_scores if item.failures
                ],
            }
            for result in results
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return md_path, html_path, json_path

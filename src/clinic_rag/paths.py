"""Repository paths that stay valid when commands run from another cwd."""

from pathlib import Path


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the repository root from clinic_rag.paths")


def corpus_dir() -> Path:
    return repo_root() / "data" / "corpus"


def questions_path() -> Path:
    return repo_root() / "data" / "eval" / "questions.jsonl"


def thresholds_path() -> Path:
    return repo_root() / "data" / "eval" / "thresholds.yml"


def snapshot_path() -> Path:
    return repo_root() / "data" / "snapshots" / "baseline_outputs.jsonl"


def docs_dir() -> Path:
    return repo_root() / "docs"

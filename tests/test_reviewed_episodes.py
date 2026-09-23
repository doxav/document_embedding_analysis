"""Offline contracts for the reviewed episode dataset exporter."""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = (Path(__file__).resolve().parents[1] / "scripts" / "promptfoo_openwebui_eval"
          / "scripts" / "export_reviewed_episodes.py")
SPEC = importlib.util.spec_from_file_location("export_reviewed_episodes", SCRIPT)
assert SPEC and SPEC.loader
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


def episode(case_id: str = "train-a", accepted: bool = True, hard_ok: bool = True) -> dict[str, Any]:
    """Build synthetic reviewed metadata without real prompts or credentials."""
    benchmark_id = f"{case_id}-{accepted}-{hard_ok}"
    return {
        "case_id": case_id, "benchmark_id": benchmark_id, "accepted": accepted,
        "status": "ACCEPTED" if accepted else "SEMANTIC_REJECTED", "capture_hash": "a" * 64,
        "grade": {"benchmark_id": benchmark_id, "capture_hash": "a" * 64,
                  "pass": accepted, "hard_ok": hard_ok,
                  "metrics": {name: {"score": 1.0 if accepted else 0.0}
                              for name in ("correctness", "groundedness")}},
        "manual_review": {"accepted": accepted},
        "usage": {"llm_calls": 3, "tool_calls": 2, "prompt_tokens": 400,
                  "completion_tokens": 100, "total_tokens": 500,
                  "cached_input_tokens": None, "reasoning_tokens": 40, "seconds": 1.5},
    }


def cases() -> list[dict[str, Any]]:
    """Include a held-out case to verify it never enters the exported view."""
    return [{"case_id": "train-a", "split": "train", "reference": "original answer"},
            {"case_id": "train-b", "split": "train"}, {"case_id": "test-a", "split": "test"}]


def test_selection_preserves_metrics_and_source_records() -> None:
    """Keep covered failures, omit technical incidents and retain unknown usage."""
    source = [episode(), episode(accepted=False), episode(accepted=False, hard_ok=False),
              episode("train-b", accepted=False),
              {"case_id": "train-a", "accepted": False, "status": "TECHNICAL_FAILURE"}]
    before = copy.deepcopy(source)
    selected, questions, report = EXPORTER.select_covered(source, cases())
    assert source == before
    assert [row["label"] for row in selected] == ["success", "failure", "failure"]
    assert selected[1]["quality_scores"]["correctness"] == 0
    assert selected[2]["quality_scores"] == {"correctness": None, "groundedness": None}
    assert selected[0]["usage"] == source[0]["usage"]
    assert all(row["accepted_solution_benchmark_id"] == source[0]["benchmark_id"] for row in selected)
    assert questions == cases()[:1]
    assert report["excluded_question_ids"] == ["train-b"]
    assert report["technical_incidents_excluded"] == 1


def test_manual_veto_is_preserved() -> None:
    """An automatic pass cannot override a negative manual review."""
    rejected = episode(accepted=False)
    rejected["grade"]["pass"] = True
    selected, _, _ = EXPORTER.select_covered([episode(), rejected], cases())
    assert selected[1]["label"] == "failure"
    assert EXPORTER.select_covered([rejected], cases())[0] == []


@pytest.mark.parametrize(("field", "value", "error"), [
    ("status", "PENDING", "Unknown episode status"),
    ("accepted", 1, "boolean"),
    ("status", "SEMANTIC_REJECTED", "contradicts"),
    ("case_id", "test-a", "TRAIN"),
    ("case_id", "missing", "TRAIN"),
    ("usage", None, "objects"),
    ("manual_review", {}, "Manual review"),
    ("capture_hash", "different", "same episode"),
])
def test_reject_inconsistent_episode(field: str, value: Any, error: str) -> None:
    """Reject invalid metadata before producing a dataset."""
    row = episode()
    row[field] = value
    with pytest.raises(ValueError, match=error):
        EXPORTER.select_covered([row], cases())


@pytest.mark.parametrize("value", [-1, 1.5, float("nan"), float("inf"), True, "100"])
def test_invalid_usage(value: Any) -> None:
    """Missing usage stays unknown, but invalid reported values are refused."""
    row = episode()
    row["usage"]["total_tokens"] = value
    with pytest.raises(ValueError, match="Invalid usage"):
        EXPORTER.select_covered([row], cases())


def test_duplicates_and_missing_judges() -> None:
    """Avoid double counting and fabricating scores for absent judge results."""
    with pytest.raises(ValueError, match="Duplicate benchmark"):
        EXPORTER.select_covered([episode(), episode()], cases())
    with pytest.raises(ValueError, match="Duplicate case"):
        EXPORTER.select_covered([episode()], cases() + cases()[:1])
    row = episode()
    row["grade"]["metrics"] = {}
    with pytest.raises(ValueError, match="Missing executed judge"):
        EXPORTER.select_covered([row], cases())


@pytest.mark.parametrize("value", [-1, 2, float("nan"), float("inf"), True, None])
def test_invalid_judge_score(value: Any) -> None:
    """Executed judge scores must be finite numeric probabilities."""
    row = episode()
    row["grade"]["metrics"]["correctness"]["score"] = value
    with pytest.raises(ValueError, match="Invalid executed judge"):
        EXPORTER.select_covered([row], cases())


def write_inputs(root: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    """Write small independent JSONL fixtures for CLI and filesystem checks."""
    catalog, question_file = root / "catalog.jsonl", root / "cases.jsonl"
    for path, records in ((catalog, rows), (question_file, cases())):
        path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    return catalog, question_file


def test_export_preview_execute_and_no_overwrite(tmp_path: Path) -> None:
    """Require execution, hash inputs/outputs, preserve originals and refuse reuse."""
    catalog, question_file = write_inputs(tmp_path, [episode()])
    before = catalog.read_bytes()
    output = tmp_path / "view"
    report = EXPORTER.export_dataset(catalog, question_file, output)
    assert report["episodes"] == 1 and not output.exists()
    report = EXPORTER.export_dataset(catalog, question_file, output, execute=True)
    assert (output.stat().st_mode & 0o777) == 0o700
    assert report == json.loads((output / "manifest.json").read_text())
    for name, expected_hash in report["output_sha256"].items():
        assert EXPORTER.file_sha256(output / name) == expected_hash
    assert catalog.read_bytes() == before
    with pytest.raises(ValueError, match="already exists"):
        EXPORTER.export_dataset(catalog, question_file, output, execute=True)


def test_missing_empty_malformed_and_held_out_inputs(tmp_path: Path) -> None:
    """No invalid input may leave an apparently complete dataset behind."""
    catalog, question_file = write_inputs(tmp_path, [])
    output = tmp_path / "view"
    with pytest.raises(ValueError, match="No accepted solution"):
        EXPORTER.export_dataset(catalog, question_file, output, execute=True)
    for content, message in (("{broken", "invalid JSON"), ("[]", "object"),
                             (json.dumps(episode("test-a")), "TRAIN")):
        catalog.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            EXPORTER.export_dataset(catalog, question_file, output, execute=True)
        assert not output.exists()
    catalog.unlink()
    with pytest.raises(ValueError, match="existing JSONL"):
        EXPORTER.export_dataset(catalog, question_file, output, execute=True)


def test_cli_from_another_directory(tmp_path: Path) -> None:
    """The documented standalone CLI works without a private worktree or PYTHONPATH."""
    catalog, question_file = write_inputs(tmp_path, [episode()])
    output = tmp_path / "view"
    command = [sys.executable, str(SCRIPT), "--catalog", str(catalog),
               "--cases", str(question_file), "--output", str(output)]
    preview = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(preview.stdout)["successes"] == 1 and not output.exists()
    subprocess.run(command + ["--execute"], cwd=tmp_path, capture_output=True, check=True)
    failed = subprocess.run(command + ["--execute"], cwd=tmp_path, capture_output=True, text=True)
    assert failed.returncode == 2 and "already exists" in failed.stderr
    assert str(tmp_path) not in failed.stderr

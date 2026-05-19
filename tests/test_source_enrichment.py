from pathlib import Path

import pytest

from common.source_enrichment import enrich_dea_solution, source_document_mode


def test_source_enrichment_field_coverage_and_non_strict(tmp_path: Path, monkeypatch):
    local = tmp_path / "a.txt"
    local.write_text("hello")

    class R:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return b"remote content"

    def fake_urlopen(url, timeout=0):
        if "ok" in url:
            return R()
        raise OSError("boom")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    solution = {
        "resources": [
            {"resource_id": 1, "path": str(local)},
            {"resource_id": 2, "local_path": str(local)},
            {"resource_id": 3, "reference": str(local)},
            {"resource_id": 4, "resource": str(local)},
            {"resource_id": 5, "url": "https://example.com/ok"},
            {"resource_id": 6, "url": "https://example.com/fail"},
        ],
        "metadata": {},
        "plan": [],
    }
    result = enrich_dea_solution(solution, tmp_path / "out", copy_local=True, fetch_remote=True, strict=False)
    statuses = {x["resource_id"]: x["status"] for x in result["source_manifest"]}
    assert statuses[1] == "copied"
    assert statuses[2] == "copied"
    assert statuses[3] == "copied"
    assert statuses[4] == "copied"
    assert statuses[5] == "fetched"
    assert statuses[6] == "failed"


def test_source_enrichment_strict_raises(tmp_path: Path):
    solution = {"resources": [{"resource_id": 1, "reference": str(tmp_path / "missing.txt")}], "metadata": {}, "plan": []}
    with pytest.raises(FileNotFoundError):
        enrich_dea_solution(solution, tmp_path / "enriched", strict=True)


def test_source_enrichment_marks_reference_only_resources(tmp_path: Path):
    solution = {"resources": [{"resource_id": 1, "resource_description": "Citation text only"}], "metadata": {}, "plan": []}
    result = enrich_dea_solution(solution, tmp_path / "enriched", strict=False)
    assert source_document_mode(solution) == "reference_only_open_retrieval"
    assert result["source_document_mode"] == "reference_only_open_retrieval"
    assert result["source_manifest"][0]["status"] == "reference_only"
    assert result["source_manifest"][0]["source_path"] is None

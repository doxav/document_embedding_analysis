import json
from pathlib import Path

import pandas as pd

from common.task_dataset import write_dataset
from scripts.import_bigsurvey import dataframe_to_task_bundles
from scripts.import_multilexsum import load_records, records_to_task_bundles, select_sources_from_json


def test_bigsurvey_schema_mapping_no_target_leakage_and_split_filter(tmp_path: Path):
    df = pd.DataFrame([
        {
            "paper_id": "p1",
            "title": "Survey Title",
            "abstract": "Survey abstract",
            "section": "Introduction",
            "text": "TARGET SECTION TEXT",
            "bib_titles": {"b1": "Source Paper"},
            "bib_abstracts": {"b1": "SOURCE ABSTRACT TEXT"},
            "bib_citing_sentences": {"b1": ["citation sentence"]},
            "split": "test",
        },
        {
            "paper_id": "p2",
            "title": "Other",
            "abstract": "Other abs",
            "section": "Only",
            "text": "TRAIN TARGET",
            "bib_titles": {"b2": "Train Source"},
            "bib_abstracts": {"b2": "TRAIN SOURCE"},
            "split": "train",
        },
    ])

    bundles = dataframe_to_task_bundles(df, split="test")
    assert len(bundles) == 1
    b = bundles[0]
    assert b.title == "Survey Title"
    assert b.sections[0].text == "TARGET SECTION TEXT"
    assert b.sources[0].text == "SOURCE ABSTRACT TEXT"
    assert b.sources[0].text != b.sections[0].text
    assert b.sections[0].citations[0]["citing_sentences"] == ["citation sentence"]

    write_dataset(bundles, tmp_path)
    source_files = list((next(tmp_path.glob("item_*/sources"))).glob("*.md"))
    assert source_files and source_files[0].read_text() != "TARGET SECTION TEXT"


def test_bigsurvey_original_schema_groups_by_title_without_paper_id():
    df = pd.DataFrame([
        {
            "title": "Original Survey",
            "abstract": "Survey abstract",
            "section": "Introduction",
            "text": "TARGET INTRO",
            "bib_titles": [{"b1": "Source Paper"}],
            "bib_abstracts": [{"b1": "SOURCE ABSTRACT"}],
            "split": "test",
        },
        {
            "title": "Original Survey",
            "abstract": "Survey abstract",
            "section": "Methods",
            "text": "TARGET METHODS",
            "bib_titles": [{"b2": "Source Two"}],
            "bib_abstracts": [{"b2": "SOURCE ABSTRACT TWO"}],
            "split": "test",
        },
    ])

    bundles = dataframe_to_task_bundles(df, split="test")
    assert len(bundles) == 1
    assert bundles[0].task_id == "Original Survey"
    assert bundles[0].metadata["group_key"] == "title"
    assert len(bundles[0].sections) == 2


def test_multilexsum_conversion_and_write(tmp_path: Path):
    records = [{"id": "x1", "summary_long": "Long body", "title": "Case"}]
    sources_blob = {"x1": [{"title": "S1", "text": "source text"}]}
    bundles = records_to_task_bundles(records, sources_blob)
    assert bundles[0].sections[0].section == "Long Summary"

    write_dataset(bundles, tmp_path)
    assert (tmp_path / "promptfoo_dataset.jsonl").exists()
    item = sorted(tmp_path.glob("item_*"))[0]
    assert (item / "task.json").exists()
    assert (item / "dea_solution.json").exists()
    assert (item / "source_manifest.json").exists()


def test_multilexsum_official_shape_conversion_and_source_selection(tmp_path: Path):
    source_json = tmp_path / "sources.json"
    source_json.write_text(
        json.dumps(
            {
                "case-1-doc-1": {"doc_id": "case-1-doc-1", "doc_text": "Document one"},
                "case-1-doc-2": {"doc_id": "case-1-doc-2", "doc_text": "Document two"},
                "unused": {"doc_id": "unused", "doc_text": "Unused"},
            }
        ),
        encoding="utf-8",
    )
    records = [
        {
            "case_id": "case-1",
            "case_name": "Example v. State",
            "case_documents": ["case-1-doc-1", "case-1-doc-2"],
            "summary/long": "Long legal summary",
            "case_type": "Civil Rights",
            "court": "District Court",
        }
    ]

    selected_sources = select_sources_from_json(source_json, {"case-1-doc-1", "case-1-doc-2"})
    bundles = records_to_task_bundles(records, selected_sources, split="test")

    assert set(selected_sources) == {"case-1-doc-1", "case-1-doc-2"}
    assert bundles[0].task_id == "case-1"
    assert bundles[0].title == "Example v. State"
    assert bundles[0].sections[0].text == "Long legal summary"
    assert [s.source_id for s in bundles[0].sources] == ["case-1-doc-1", "case-1-doc-2"]
    assert "case_type: Civil Rights" in bundles[0].abstract


def test_multilexsum_load_records_supports_jsonl(tmp_path: Path):
    records_path = tmp_path / "test.json"
    records_path.write_text('{"case_id": "a"}\n{"case_id": "b"}\n', encoding="utf-8")
    assert [r["case_id"] for r in load_records(records_path)] == ["a", "b"]


def test_promptfoo_paths_exist(tmp_path: Path):
    records = [{"id": "x1", "summary_long": "Long body", "title": "Case"}]
    sources_blob = {"x1": [{"title": "S1", "text": "source text"}]}
    bundles = records_to_task_bundles(records, sources_blob)
    write_dataset(bundles, tmp_path)
    lines = [json.loads(x) for x in (tmp_path / "promptfoo_dataset.jsonl").read_text().splitlines() if x.strip()]
    assert lines
    for rec in lines:
        assert (tmp_path / rec["expected_path"]).exists()
        assert (tmp_path / rec["input"]["instruction_path"]).exists()
        assert (tmp_path / rec["sections_path"]).exists()
        assert (tmp_path / rec["source_manifest_path"]).exists()
        for p in rec["input"]["source_paths"]:
            assert (tmp_path / p).exists()


def test_multilexsum_cli_style_split_filtering():
    records = [
        {"id": "train", "summary_long": "Train body", "title": "Train", "split": "train"},
        {"id": "test", "summary_long": "Test body", "title": "Test", "split": "test"},
    ]
    sources_blob = {
        "train": [{"title": "Train source", "text": "train source"}],
        "test": [{"title": "Test source", "text": "test source"}],
    }
    filtered = [r for r in records if str(r.get("split", "test")).lower() == "test"]
    bundles = records_to_task_bundles(filtered, sources_blob)
    assert [b.task_id for b in bundles] == ["test"]

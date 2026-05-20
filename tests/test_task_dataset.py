import json
from pathlib import Path

from common.task_dataset import TaskBundle, TaskSection, TaskSource, write_dataset, write_task_bundle


def test_task_bundle_dea_compatibility(tmp_path: Path):
    bundle = TaskBundle(
        task_id="item_0001_demo",
        dataset="demo",
        title="Demo Title",
        abstract="Demo Abstract",
        instruction="Write summary",
        sections=[TaskSection(section="Intro", text="Intro text", source_ids=["s1"])],
        sources=[
            TaskSource(source_id="s1", title="Source 1", text="Text 1", path="/tmp/s1.txt"),
            TaskSource(source_id="s2", title="Source 2", text="Text 2", path="/tmp/s2.txt"),
        ],
    )

    item_dir = write_task_bundle(bundle, tmp_path)
    dea = json.loads((item_dir / "dea_solution.json").read_text())

    assert dea["resources"][0]["resource_id"] == 1
    assert dea["plan"][0]["resources_used"] == [1]
    assert "resource_description" in dea["resources"][0]
    assert "target_file_path" in dea


def test_MQS_evaluation_file_backed_paths(tmp_path: Path):
    bundle = TaskBundle(
        task_id="demo",
        dataset="demo",
        title="T",
        abstract="A",
        instruction="I",
        sections=[TaskSection(section="S", text="C", source_ids=["s1"])],
        sources=[TaskSource(source_id="s1", title="Source", text="Text")],
    )
    write_dataset([bundle], tmp_path)
    row = json.loads((tmp_path / "MQS_evaluation_dataset.jsonl").read_text().splitlines()[0])
    assert (tmp_path / row["expected_path"]).exists()
    assert (tmp_path / row["input"]["instruction_path"]).exists()
    assert (tmp_path / row["sections_path"]).exists()
    assert (tmp_path / row["source_manifest_path"]).exists()
    for source_path in row["input"]["source_paths"]:
        assert (tmp_path / source_path).exists()


def test_task_bundle_rejects_duplicate_source_ids(tmp_path: Path):
    bundle = TaskBundle(
        task_id="duplicate-source",
        dataset="demo",
        title="T",
        abstract="A",
        instruction="I",
        sections=[TaskSection(section="S", text="C", source_ids=["s1"])],
        sources=[
            TaskSource(source_id="s1", title="A", text="A"),
            TaskSource(source_id="s1", title="B", text="B"),
        ],
    )
    import pytest
    with pytest.raises(ValueError, match="Duplicate TaskSource.source_id"):
        write_task_bundle(bundle, tmp_path)


def test_task_bundle_truncates_long_generated_filenames(tmp_path: Path):
    long_heading = " ".join(["Very Long Section Heading"] * 30)
    long_title = " ".join(["Very Long Source Title"] * 30)
    bundle = TaskBundle(
        task_id="long-file-names",
        dataset="demo",
        title="T",
        abstract="A",
        instruction="I",
        sections=[TaskSection(section=long_heading, text="C", source_ids=["s1"])],
        sources=[TaskSource(source_id="s1", title=long_title, text="Text")],
    )
    item = write_task_bundle(bundle, tmp_path)
    section_file = next((item / "sections").glob("*.md"))
    source_file = next((item / "sources").glob("*.md"))
    assert len(section_file.name.encode("utf-8")) < 255
    assert len(source_file.name.encode("utf-8")) < 255


def test_write_dataset_removes_stale_generated_items(tmp_path: Path):
    first = TaskBundle(
        task_id="old",
        dataset="demo",
        title="Old",
        abstract="A",
        instruction="I",
        sections=[TaskSection(section="S", text="C", source_ids=["s1"])],
        sources=[TaskSource(source_id="s1", title="Source", text="Text")],
    )
    second = TaskBundle(
        task_id="new",
        dataset="demo",
        title="New",
        abstract="A",
        instruction="I",
        sections=[TaskSection(section="S", text="C", source_ids=["s1"])],
        sources=[TaskSource(source_id="s1", title="Source", text="Text")],
    )

    write_dataset([first], tmp_path)
    assert (tmp_path / "item_old").exists()

    write_dataset([second], tmp_path)
    assert not (tmp_path / "item_old").exists()
    assert (tmp_path / "item_new").exists()


def test_write_dataset_validates_before_cleaning(tmp_path: Path):
    good = TaskBundle(
        task_id="kept",
        dataset="demo",
        title="Kept",
        abstract="A",
        instruction="I",
        sections=[TaskSection(section="S", text="C", source_ids=["s1"])],
        sources=[TaskSource(source_id="s1", title="Source", text="Text")],
    )
    duplicate_a = TaskBundle(
        task_id="dup",
        dataset="demo",
        title="A",
        abstract="A",
        instruction="I",
        sections=[TaskSection(section="S", text="C", source_ids=["s1"])],
        sources=[TaskSource(source_id="s1", title="Source", text="Text")],
    )
    duplicate_b = TaskBundle(
        task_id="dup",
        dataset="demo",
        title="B",
        abstract="A",
        instruction="I",
        sections=[TaskSection(section="S", text="C", source_ids=["s1"])],
        sources=[TaskSource(source_id="s1", title="Source", text="Text")],
    )

    write_dataset([good], tmp_path)

    import pytest
    with pytest.raises(ValueError, match="Duplicate TaskBundle.task_id"):
        write_dataset([duplicate_a, duplicate_b], tmp_path)

    assert (tmp_path / "item_kept").exists()

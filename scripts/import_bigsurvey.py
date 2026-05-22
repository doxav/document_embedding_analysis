from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.bigsurvey_archive import ensure_bigsurvey_archive, find_bigsurvey_dataframe, unpack_archive_safe
from common.task_dataset import TaskBundle, TaskSection, TaskSource, write_dataset


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        merged: dict[str, Any] = {}
        for item in value:
            if isinstance(item, dict):
                merged.update(item)
        return merged
    return {}


def dataframe_to_task_bundles(df, n: int | None = None, split: str | None = None):
    if split and split != "all" and "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == split.lower()]

    group_key = "paper_id" if "paper_id" in df.columns else "title"
    bundles = []
    for idx, (paper_id, g) in enumerate(df.groupby(group_key, sort=False), start=1):
        if n and idx > n:
            break
        title = _clean(g.iloc[0].get("title")) or str(paper_id)
        abstract = _clean(g.iloc[0].get("abstract"))
        task_id = _clean(g.iloc[0].get("paper_id")) if "paper_id" in g.columns else title
        if not task_id:
            task_id = f"bigsurvey_{idx}"

        source_map: dict[str, TaskSource] = {}
        sections: list[TaskSection] = []
        for _, row in g.iterrows():
            bib_titles = _as_dict(row.get("bib_titles"))
            bib_abstracts = _as_dict(row.get("bib_abstracts"))
            citing = _as_dict(row.get("bib_citing_sentences"))

            source_ids, citations = [], []
            for bib_id in sorted(set(bib_titles) | set(bib_abstracts) | set(citing)):
                sid = str(bib_id)
                source_text = _clean(bib_abstracts.get(bib_id))
                if not source_text:
                    continue
                source_title = _clean(bib_titles.get(bib_id))
                source_ids.append(sid)
                citations.append({"source_id": sid, "title": source_title, "citing_sentences": citing.get(bib_id, [])})
                source_map.setdefault(sid, TaskSource(source_id=sid, title=source_title, text=source_text, metadata={"bib_id": sid}))

            section_text = _clean(row.get("text"))
            if not section_text:
                continue
            sections.append(TaskSection(section=_clean(row.get("section")) or "Section", text=section_text, source_ids=source_ids, citations=citations))

        if sections and source_map:
            metadata = {"split": _clean(g.iloc[0].get("split")), "group_key": group_key}
            if "paper_id" in g.columns:
                metadata["paper_id"] = str(paper_id)
            bundles.append(TaskBundle(task_id=str(task_id), dataset="bigsurvey", title=title, abstract=abstract, instruction="Write a structured long-form literature-review synthesis using only the provided cited-paper abstracts.", sections=sections, sources=list(source_map.values()), metadata=metadata))
    return bundles


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", default="split", help="original, split, local path, Drive id, or URL")
    p.add_argument("--work-dir", default="data/bigsurvey")
    p.add_argument("--output-dir", default="output/bigsurvey")
    p.add_argument("--n", type=int)
    p.add_argument("--split", default="test")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-embeddings", action="store_true")
    args = p.parse_args()

    work_dir = Path(args.work_dir)
    try:
        archive = ensure_bigsurvey_archive(args.archive, work_dir, overwrite=args.overwrite)
        archive_label = args.archive.strip().lower() if args.archive.strip().lower() in {"original", "split"} else archive.stem
        files = unpack_archive_safe(archive, work_dir / f"extracted_{archive_label}", overwrite=args.overwrite)
        df, _ = find_bigsurvey_dataframe(files)
    except Exception as exc:
        raise SystemExit(
            "Could not prepare BigSurvey input data. "
            "Run with network access so the importer can download the Google Drive archive, "
            "or place split_survey_df.tar.gz/original_survey_df.tar.gz under data/bigsurvey. "
            f"Original error: {exc}"
        ) from exc
    bundles = dataframe_to_task_bundles(df, n=args.n, split=args.split)
    if not bundles:
        raise SystemExit(
            "BigSurvey import produced 0 examples. "
            "Check --split, --n, and whether the archive contains title/abstract/section/text/"
            "bib_titles/bib_abstracts columns."
        )
    write_dataset(bundles, Path(args.output_dir))
    (Path(args.output_dir) / "dataset_manifest.json").write_text(json.dumps({"dataset": "bigsurvey", "count": len(bundles), "split": args.split}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

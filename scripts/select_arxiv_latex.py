#!/usr/bin/env python
"""
Select and download arXiv survey / top-conference papers in LaTeX form.

- Queries the arXiv API for recent cs/AI/ML papers.
- Filters for surveys (title/abstract) OR papers with top-conference mentions
  in `journal_ref` or `comments`.
- Downloads the LaTeX source tarball via https://arxiv.org/e-print/<id>.
- Keeps only papers whose source archive contains exactly ONE .tex and ONE .bib.
- Writes them as `arxiv_<id>.tex` / `arxiv_<id>.bib` in the target directory.

This script is intentionally conservative: if anything looks odd, the paper is skipped.
"""

from __future__ import annotations

import argparse
import io
import tarfile
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import requests

ARXIV_API_URL = "http://export.arxiv.org/api/query"

# Categories we care about (computer science / ML-ish)
DEFAULT_CATEGORIES = [
    "cs.LG",
    "cs.AI",
    "cs.CL",
    "cs.CV",
    "cs.IR",
    "stat.ML",
]

TOP_CONF_KEYWORDS = [
    "neurips",
    "nips",
    "icml",
    "iclr",
    "acl",
    "emnlp",
    "naacl",
    "cvpr",
    "eccv",
    "iccv",
    "kdd",
    "aaai",
    "www",
    "sigir",
]


@dataclass
class ArxivEntry:
    arxiv_id: str
    title: str
    summary: str
    comment: str
    journal_ref: str


def _build_search_query(categories: List[str]) -> str:
    # cat:cs.LG OR cat:cs.AI OR ...
    cats = " OR ".join(f"cat:{c}" for c in categories)
    return cats


def _fetch_raw_feed(
    search_query: str,
    max_results: int = 200,
    start: int = 0,
) -> str:
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.text


def _parse_feed(xml_text: str) -> List[ArxivEntry]:
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(xml_text)
    entries: List[ArxivEntry] = []
    for entry in root.findall("atom:entry", ns):
        id_el = entry.find("atom:id", ns)
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        comment_el = entry.find("arxiv:comment", ns)
        journal_el = entry.find("arxiv:journal_ref", ns)

        if id_el is None or title_el is None or summary_el is None:
            continue

        arxiv_id = id_el.text.split("/")[-1]
        entries.append(
            ArxivEntry(
                arxiv_id=arxiv_id.strip(),
                title=(title_el.text or "").strip(),
                summary=(summary_el.text or "").strip(),
                comment=(comment_el.text or "").strip() if comment_el is not None else "",
                journal_ref=(journal_el.text or "").strip() if journal_el is not None else "",
            )
        )
    return entries


def _is_survey_or_topconf(entry: ArxivEntry) -> bool:
    blob = " ".join(
        [
            entry.title,
            entry.summary,
            entry.comment,
            entry.journal_ref,
        ]
    ).lower()

    is_survey_like = any(word in blob for word in ["survey", "overview", "review"])
    is_top_conf = any(k in blob for k in TOP_CONF_KEYWORDS)
    return is_survey_like or is_top_conf


def _download_source_tar(arxiv_id: str) -> bytes:
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def _extract_single_tex_bib(
    tar_bytes: bytes,
) -> tuple[str | None, str | None]:
    tex_content = None
    bib_content = None

    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tf:
            tex_members = [m for m in tf.getmembers() if m.name.lower().endswith(".tex")]
            bib_members = [m for m in tf.getmembers() if m.name.lower().endswith(".bib")]

            # Require exactly one .tex and one .bib
            if len(tex_members) != 1 or len(bib_members) != 1:
                return None, None

            tex_member = tex_members[0]
            bib_member = bib_members[0]

            tex_file = tf.extractfile(tex_member)
            bib_file = tf.extractfile(bib_member)
            if tex_file is None or bib_file is None:
                return None, None

            tex_bytes = tex_file.read()
            bib_bytes = bib_file.read()

            tex_content = tex_bytes.decode("utf-8", errors="replace")
            bib_content = bib_bytes.decode("utf-8", errors="replace")
    except (tarfile.ReadError, tarfile.TarError, OSError):
        # Not a valid tarball (could be error page, plain .tex, etc.)
        return None, None

    # Very rough sanity check: must look like a real article
    if tex_content is None or "\\begin{document}" not in tex_content:
        return None, None

    return tex_content, bib_content


def _save_pair(tex: str, bib: str, out_dir: Path, arxiv_id: str, title: str = "") -> Path:
    # DocLatex expects same basename for .tex and .bib
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize title: limit length, replace spaces with underscores, remove special chars
    if title:
        # Remove common LaTeX commands and special characters
        clean_title = title.replace("\n", " ").replace("\r", " ")
        clean_title = clean_title.replace("\\", "").replace("{", "").replace("}", "")
        clean_title = clean_title.strip()
        # Replace spaces and special chars with underscores
        clean_title = "".join(
            c if c.isalnum() or c in " -" else "_" 
            for c in clean_title
        )
        clean_title = "_".join(clean_title.split())  # Collapse multiple spaces
        # Remove consecutive underscores
        while "__" in clean_title:
            clean_title = clean_title.replace("__", "_")
        # Limit to 80 characters
        clean_title = clean_title[:80].rstrip("_")
        slug = arxiv_id.replace("/", "_")
        basename = f"{clean_title}_arxiv_{slug}" if clean_title else f"arxiv_{slug}"
    else:
        slug = arxiv_id.replace("/", "_")
        basename = f"arxiv_{slug}"
    
    tex_path = out_dir / f"{basename}.tex"
    bib_path = out_dir / f"{basename}.bib"

    if tex_path.exists() or bib_path.exists():
        # Don't overwrite existing files
        return tex_path

    tex_path.write_text(tex, encoding="utf-8")
    bib_path.write_text(bib, encoding="utf-8")
    return tex_path


def select_arxiv_latex(
    out_dir: Path,
    max_papers: int = 100,
    categories: Iterable[str] = None,
    api_batch_size: int = 200,
    api_max_batches: int = 5,
    dry_run: bool = False,
) -> int:
    """
    Main selection pipeline. Returns number of .tex/.bib pairs created.
    """
    if categories is None:
        categories = DEFAULT_CATEGORIES

    search_query = _build_search_query(list(categories))
    created = 0
    seen_ids: set[str] = set()

    for batch_idx in range(api_max_batches):
        if created >= max_papers:
            break

        start = batch_idx * api_batch_size
        xml_text = _fetch_raw_feed(search_query, max_results=api_batch_size, start=start)
        entries = _parse_feed(xml_text)

        if not entries:
            break

        for entry in entries:
            if created >= max_papers:
                break
            if entry.arxiv_id in seen_ids:
                continue
            seen_ids.add(entry.arxiv_id)

            if not _is_survey_or_topconf(entry):
                continue

            # Check if file already exists before downloading
            slug = entry.arxiv_id.replace("/", "_")
            existing_files = list(out_dir.glob(f"*arxiv_{slug}.tex"))
            if existing_files:
                print(f"[candidate] {entry.arxiv_id} – {entry.title[:80]}")
                print(f"  ✓ already exists: {existing_files[0].name}")
                continue

            print(f"[candidate] {entry.arxiv_id} – {entry.title[:80]}")

            if dry_run:
                # In dry-run mode, only show potential candidates
                continue

            try:
                tar_bytes = _download_source_tar(entry.arxiv_id)
            except Exception as exc:  # noqa: BLE001
                print(f"  -> failed to download source: {exc}")
                continue

            tex, bib = _extract_single_tex_bib(tar_bytes)
            if tex is None or bib is None:
                print("  -> skipped (no single .tex + .bib or invalid LaTeX)")
                continue

            # Save the pair
            saved_path = _save_pair(tex, bib, out_dir, entry.arxiv_id, entry.title)
            created += 1
            print(f"  ✓ saved {saved_path.name}")

    print(f"\n✓ Created {created} LaTeX pairs in {out_dir}")
    return created


def main():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/latex"),
        help="Directory to save .tex/.bib pairs (default: data/latex)",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=5,
        help="Maximum number of papers to download (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show candidates without downloading",
    )
    args = parser.parse_args()

    count = select_arxiv_latex(
        out_dir=args.out_dir,
        max_papers=args.max_papers,
        dry_run=args.dry_run,
    )
    print(f"\n✓ Final count: {count} papers")


if __name__ == "__main__":
    main()

"""
Split an EPO weekly extract file (multi-brevet) into one file per patent,
compatible with common/doc_patent.DocPatent.

Usage:
    python scripts/split_epo_extract.py \
        data/2022week30_EP0600000_extract.txt \
        data/patents

Each output file contains only English language lines (langue "en")
for a given EP patent, with at least TITLE / DESCR / CLAIM sections.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def sanitize_title(title: str) -> str:
    """
    Sanitize patent title for use in filename.
    - Replace spaces with underscores
    - Remove special characters
    - Limit to 80 characters
    """
    # Remove newlines and extra whitespace
    clean_title = title.replace("\n", " ").replace("\r", " ")
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
    return clean_title


def parse_epo_line(line: str) -> Tuple[str, str, str, str, str]:
    """
    Parse one EPO extract line.

    Typical format (tab-delimited) from 2022week30_EP0600000_extract.txt:
        EP  0600858 B1  2000-05-17  en  TITLE  1  CONDOM FOR ORAL-GENITAL USE

    Returns (country, doc_num, kind, lang, section_type).
    """
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 7:
        raise ValueError(f"Unexpected line format (too few columns): {line!r}")
    country = parts[0].strip()
    doc_num = parts[1].strip()
    kind = parts[2].strip()
    lang = parts[4].strip()
    section_type = parts[5].strip()
    return country, doc_num, kind, lang, section_type


def split_epo_extract(
    input_path: Path,
    output_dir: Path,
    min_sections: Tuple[str, ...] = ("TITLE", "DESCR", "CLAIM"),
) -> int:
    """
    Split a multi-patent EPO extract into one EN-only file per patent.

    Returns the number of patents actually written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    per_patent_en_lines: Dict[str, List[str]] = defaultdict(list)
    with input_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # Ignore empty lines
            if not line.strip():
                continue
            try:
                country, doc_num, kind, lang, section_type = parse_epo_line(line)
            except ValueError:
                # Non-conforming line: ignore it (or could log)
                continue

            # Only keep English language: DocPatent already filters on "\ten\t"
            if lang.lower() != "en":
                continue

            key = f"{country}{doc_num}{kind}"
            per_patent_en_lines[key].append(line)

    written = 0
    for key, lines in per_patent_en_lines.items():
        # Verify that we have at least TITLE / DESCR / CLAIM
        section_types = {parse_epo_line(l)[4] for l in lines}
        if not all(sec in section_types for sec in min_sections):
            # Skip incomplete patents for the existing parser
            continue

        # Extract title from TITLE section for filename
        title_text = ""
        for line in lines:
            try:
                _, _, _, _, section_type = parse_epo_line(line)
                if section_type == "TITLE":
                    # Title is in the last column (after the 6 tab-separated fields)
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 7:
                        title_text = parts[7].strip()
                        break
            except ValueError:
                continue
        
        # Build filename with title
        if title_text:
            sanitized_title = sanitize_title(title_text)
            filename = f"{sanitized_title}_{key}.txt" if sanitized_title else f"{key}.txt"
        else:
            filename = f"{key}.txt"
        
        out_path = output_dir / filename
        # If file already exists, avoid overwriting
        if out_path.exists():
            continue

        with out_path.open("w", encoding="utf-8") as out_f:
            out_f.writelines(lines)

        written += 1

    return written


def main(argv: List[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) != 2:
        print(
            "Usage:\n"
            "  python scripts/split_epo_extract.py "
            "input_extract.txt data/patents\n"
        )
        raise SystemExit(1)

    input_file = Path(argv[0])
    out_dir = Path(argv[1])

    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        raise SystemExit(1)

    written = split_epo_extract(input_file, out_dir)
    print(f"Created {written} patent files in {out_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.source_enrichment import enrich_dea_solution, write_enriched_task_bundle
from common.task_dataset import write_promptfoo_dataset_from_items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dea-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fetch-remote", action="store_true")
    p.add_argument("--copy-local", dest="copy_local", action="store_true", default=True)
    p.add_argument("--no-copy-local", dest="copy_local", action="store_false")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    item_dirs = []
    skipped = []
    mode_counts = {}
    for path in sorted(Path(args.dea_root).rglob("*.json")):
        sol = json.loads(path.read_text(encoding="utf-8"))
        try:
            enriched = enrich_dea_solution(
                sol,
                output_root / path.stem,
                fetch_remote=args.fetch_remote,
                copy_local=args.copy_local,
                strict=not args.continue_on_error,
                overwrite=args.overwrite,
            )
            mode = enriched.get("source_document_mode", "unknown")
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            item_dirs.append(write_enriched_task_bundle(enriched, output_root / path.stem))
        except Exception as exc:
            skipped.append({"path": str(path), "error": str(exc)})
            if not args.continue_on_error:
                raise

    (output_root / "dataset_manifest.json").write_text(
        json.dumps(
            {"dataset": "enriched_dea", "count": len(item_dirs), "source_document_modes": mode_counts, "skipped": skipped},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_promptfoo_dataset_from_items(output_root, item_dirs)
    if mode_counts.get("reference_only_open_retrieval") or mode_counts.get("no_resources"):
        print(
            "Warning: some DEA items do not provide source documents. "
            "They were exported as open-retrieval/reference-only tasks; inspect dataset_manifest.json.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

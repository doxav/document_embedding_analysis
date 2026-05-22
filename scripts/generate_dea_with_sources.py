from __future__ import annotations

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--type", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source-output", required=True)
    p.add_argument("--enrich-sources", action="store_true")
    p.add_argument("--fetch-remote", action="store_true")
    p.add_argument("--skip-embeddings", action="store_true")
    _ = p.parse_args()
    raise SystemExit(
        "generate_dea_with_sources.py is currently disabled: main.py is interactive-only and does not "
        "support non-interactive flags yet. Use scripts/enrich_dea_sources.py for second-pass enrichment."
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(os.environ.get("DEA_REPO_ROOT", "")).resolve()
    if not repo_root.exists():
        raise SystemExit(f"DEA_REPO_ROOT does not exist: {repo_root}")
    py = os.environ.get("PROMPTFOO_PYTHON") or sys.executable
    bigsurvey_importer = repo_root / "scripts" / "import_bigsurvey.py"
    multilexsum_importer = repo_root / "scripts" / "import_multilexsum.py"
    if not bigsurvey_importer.exists() or not multilexsum_importer.exists():
        raise SystemExit(
            "Could not find scripts/import_bigsurvey.py and scripts/import_multilexsum.py. "
            "Use the MDS_datasets branch of document_embedding_analysis."
        )
    tasks = [
        (repo_root / "output" / "bigsurvey" / "MQS_evaluation_dataset.jsonl", [py, str(bigsurvey_importer), "--n", "20", "--split", "test", "--skip-embeddings"]),
        (repo_root / "output" / "multilexsum" / "MQS_evaluation_dataset.jsonl", [py, str(multilexsum_importer), "--n", "20", "--split", "test", "--skip-embeddings"]),
    ]
    for output_path, cmd in tasks:
        if output_path.exists():
            print(f"[ok] {output_path} already exists")
            continue
        print(f"[run] {' '.join(cmd)}")
        subprocess.run(cmd, cwd=repo_root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

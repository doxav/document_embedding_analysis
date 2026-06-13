#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys

BUNDLE_ROOT = _Path(__file__).resolve().parents[1]
if str(BUNDLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(BUNDLE_ROOT))

import argparse
import json
import os
from pathlib import Path

import requests

from lib.bundle_common import read_csv_rows, trim_text, write_csv_rows


def _request_timeout_seconds() -> int:
    """Return the bridge request timeout used by long live generations."""
    raw = os.environ.get("BRIDGE_REQUEST_TIMEOUT_SECONDS", "1800").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("BRIDGE_REQUEST_TIMEOUT_SECONDS must be an integer") from exc
    if value <= 0:
        raise ValueError("BRIDGE_REQUEST_TIMEOUT_SECONDS must be positive")
    return value


def _raise_for_status(response: requests.Response) -> None:
    """Raise HTTP errors with the bridge response body preserved."""
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = trim_text(response.text, 1200)
        raise requests.HTTPError(f"{exc}; response body: {detail}", response=response) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate candidate answers through the local OpenWebUI bridge and write them back into a CSV.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bridge-url", default="http://127.0.0.1:8001/generate")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    timeout = _request_timeout_seconds()
    rows = read_csv_rows(Path(args.input))
    out_rows = []
    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        if args.limit and idx > args.limit:
            break
        current = dict(row)
        if current.get("candidate_answer") and not args.overwrite:
            out_rows.append(current)
            write_csv_rows(output_path, out_rows)
            continue

        payload = {k: v for k, v in current.items() if not k.startswith("__metadata:")}
        response = requests.post(args.bridge_url, json=payload, timeout=timeout)
        _raise_for_status(response)
        data = response.json()
        current["candidate_answer"] = str(data.get("output") or "").strip()
        out_rows.append(current)
        write_csv_rows(output_path, out_rows)
        print(f"[{idx}/{total}] generated {current.get('dataset')}::{current.get('task_id')}", flush=True)

    write_csv_rows(output_path, out_rows)
    print(f"Wrote {len(out_rows)} rows to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

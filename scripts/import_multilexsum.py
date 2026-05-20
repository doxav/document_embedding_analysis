from __future__ import annotations

import argparse
import json
import mmap
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.task_dataset import TaskBundle, TaskSection, TaskSource, write_dataset

PRIMARY_BASE = "https://huggingface.co/datasets/allenai/multi_lexsum/resolve/main/releases/v20230518"
ALT_BASE = "https://ai2-s2-research.s3.us-west-2.amazonaws.com/multilexsum/releases/v20230518"
OFFICIAL_BASES = (PRIMARY_BASE, ALT_BASE)
SPLITS = ("train", "dev", "test")
LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/"


def _json_loads_lines(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _is_lfs_pointer(payload: bytes | str) -> bool:
    text = payload.decode("utf-8", errors="ignore") if isinstance(payload, bytes) else payload
    return text.startswith(LFS_POINTER_PREFIX)


def _download_official_file(rel_path: str, dest: Path, bases: Iterable[str] = OFFICIAL_BASES) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for base in bases:
        url = f"{base.rstrip('/')}/{rel_path}"
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=600) as resp, tmp.open("wb") as f:
                head = resp.read(2048)
                if _is_lfs_pointer(head):
                    raise RuntimeError(f"{url} returned a Git-LFS pointer")
                f.write(head)
                shutil.copyfileobj(resp, f, length=1024 * 1024)
            tmp.replace(dest)
            return dest
        except Exception as exc:
            last_error = exc
            if tmp.exists():
                tmp.unlink()
    raise RuntimeError(f"Could not fetch official MultiLexSum file {rel_path!r}: {last_error}")


def official_file(rel_path: str, cache_dir: Path, *, overwrite: bool = False) -> Path:
    dest = cache_dir / rel_path
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        with dest.open("rb") as f:
            if not _is_lfs_pointer(f.read(2048)):
                return dest
    return _download_official_file(rel_path, dest)


def fetch_text(rel_path: str, cache_dir: Path | None = None, *, overwrite: bool = False) -> str:
    if cache_dir is not None:
        path = official_file(rel_path, cache_dir, overwrite=overwrite)
        return path.read_text(encoding="utf-8")
    last_error: Exception | None = None
    for base in OFFICIAL_BASES:
        url = f"{base.rstrip('/')}/{rel_path}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=600) as resp:
                payload = resp.read().decode("utf-8")
            if _is_lfs_pointer(payload):
                raise RuntimeError(f"{url} returned a Git-LFS pointer")
            return payload
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not fetch official MultiLexSum file {rel_path!r}: {last_error}")


def fetch_json(rel_path: str, cache_dir: Path | None = None, *, overwrite: bool = False) -> Any:
    return json.loads(fetch_text(rel_path, cache_dir, overwrite=overwrite))


def fetch_jsonl(rel_path: str, cache_dir: Path | None = None, *, overwrite: bool = False) -> list[dict[str, Any]]:
    return _json_loads_lines(fetch_text(rel_path, cache_dir, overwrite=overwrite))


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []
    if path.suffix == ".jsonl" or stripped[0] == "{":
        try:
            return _json_loads_lines(text)
        except json.JSONDecodeError:
            pass
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    raise ValueError(f"Unsupported MultiLexSum records shape in {path}")


def load_sources(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported MultiLexSum sources shape in {path}")
    return payload


def filter_local_records_by_split(records: list[dict[str, Any]], split: str | None) -> list[dict[str, Any]]:
    if not split or split == "all":
        return records
    records_with_split = [r for r in records if "split" in r]
    if not records_with_split:
        return records
    return [r for r in records if str(r.get("split", "")).lower() == split.lower()]


def select_sources_from_json(path: Path, source_ids: set[str]) -> dict[str, Any]:
    if not source_ids:
        return {}
    selected: dict[str, Any] = {}
    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for source_id in source_ids:
            key = json.dumps(source_id).encode("utf-8")
            key_pos, colon_pos = _top_level_key_position(mm, key)
            if key_pos < 0 or colon_pos < 0:
                continue
            start = colon_pos + 1
            while start < len(mm) and chr(mm[start]).isspace():
                start += 1
            end = _json_value_end(mm, start)
            selected[source_id] = json.loads(mm[start:end].decode("utf-8"))
    return selected


def _top_level_key_position(mm: mmap.mmap, key: bytes) -> tuple[int, int]:
    search_from = 0
    while True:
        key_pos = mm.find(key, search_from)
        if key_pos < 0:
            return -1, -1
        before = key_pos - 1
        while before >= 0 and chr(mm[before]).isspace():
            before -= 1
        after = key_pos + len(key)
        while after < len(mm) and chr(mm[after]).isspace():
            after += 1
        if before >= 0 and mm[before] in (ord("{"), ord(",")) and after < len(mm) and mm[after] == ord(":"):
            return key_pos, after
        search_from = key_pos + 1


def _json_value_end(mm: mmap.mmap, start: int) -> int:
    if start >= len(mm):
        raise ValueError("Unexpected end of JSON while reading value")
    opener = mm[start]
    if opener not in (ord("{"), ord("[")):
        pos = start
        while pos < len(mm) and mm[pos] not in (ord(","), ord("}")):
            pos += 1
        return pos

    depth = 0
    in_string = False
    escape = False
    pos = start
    while pos < len(mm):
        ch = mm[pos]
        if in_string:
            if escape:
                escape = False
            elif ch == ord("\\"):
                escape = True
            elif ch == ord('"'):
                in_string = False
        else:
            if ch == ord('"'):
                in_string = True
            elif ch in (ord("{"), ord("[")):
                depth += 1
            elif ch in (ord("}"), ord("]")):
                depth -= 1
                if depth == 0:
                    return pos + 1
        pos += 1
    raise ValueError("Unexpected end of JSON while reading value")


def _summary_long(record: dict[str, Any]) -> str:
    return str(record.get("summary_long") or record.get("summary/long") or "").strip()


def _record_id(record: dict[str, Any]) -> str:
    return str(record.get("id") or record.get("case_id") or "").strip()


def _record_title(record: dict[str, Any], rid: str) -> str:
    return str(record.get("title") or record.get("case_name") or rid).strip()


def _record_context(record: dict[str, Any]) -> str:
    if record.get("context"):
        return str(record["context"]).strip()
    parts = []
    for key in ("case_type", "court", "state", "filing_year", "closing_year", "case_url"):
        value = record.get(key)
        if value not in (None, "", "nan"):
            parts.append(f"{key}: {value}")
    causes = record.get("causes_of_action") or record.get("case_types")
    if causes:
        parts.append(f"topics: {', '.join(map(str, causes)) if isinstance(causes, list) else causes}")
    return "\n".join(parts)


def _source_title(source_id: str, source: Any, index: int) -> str:
    if isinstance(source, dict):
        return str(source.get("title") or source.get("doc_name") or source.get("doc_id") or source_id or f"s{index}")
    return source_id or f"s{index}"


def _source_text(source: Any) -> str:
    if isinstance(source, dict):
        return str(source.get("text") or source.get("doc_text") or source.get("content") or "")
    return str(source or "")


def _source_items(record: dict[str, Any], sources_blob: dict[str, Any]) -> list[tuple[str, Any]]:
    doc_ids = record.get("case_documents")
    if isinstance(doc_ids, list):
        return [(str(doc_id), sources_blob[str(doc_id)]) for doc_id in doc_ids if str(doc_id) in sources_blob]
    rid = _record_id(record)
    sources = sources_blob.get(rid, [])
    if isinstance(sources, dict):
        sources = list(sources.values())
    if not isinstance(sources, list):
        return []
    return [(str(source.get("source_id") or source.get("doc_id") or f"s{i+1}") if isinstance(source, dict) else f"s{i+1}", source) for i, source in enumerate(sources)]


def records_to_task_bundles(records: list[dict[str, Any]], sources_blob: dict[str, Any], *, split: str | None = None, n: int | None = None) -> list[TaskBundle]:
    bundles = []
    for record in records:
        rid = _record_id(record)
        if not rid:
            continue
        source_items = _source_items(record, sources_blob)
        sources = [
            TaskSource(source_id=source_id, title=_source_title(source_id, source, i), text=_source_text(source))
            for i, (source_id, source) in enumerate(source_items, start=1)
            if _source_text(source).strip()
        ]
        summary = _summary_long(record)
        if not summary or not sources:
            continue
        source_ids = [source.source_id for source in sources]
        metadata = {
            "split": split or record.get("split") or record.get("subset") or "test",
            "case_id": record.get("case_id", rid),
            "case_url": record.get("case_url"),
        }
        bundles.append(
            TaskBundle(
                task_id=rid,
                dataset="multilexsum",
                title=_record_title(record, rid),
                abstract=_record_context(record),
                instruction="Draft a long legal case summary using only the provided case documents.",
                sections=[TaskSection(section="Long Summary", text=summary, source_ids=source_ids)],
                sources=sources,
                metadata={k: v for k, v in metadata.items() if v not in (None, "")},
            )
        )
        if n and len(bundles) >= n:
            break
    return bundles


def official_records(split: str, cache_dir: Path, *, overwrite: bool = False) -> list[dict[str, Any]]:
    if split == "all":
        records: list[dict[str, Any]] = []
        for part in SPLITS:
            for record in fetch_jsonl(f"{part}.json", cache_dir, overwrite=overwrite):
                record = dict(record)
                record["split"] = part
                records.append(record)
        return records
    if split not in SPLITS:
        raise ValueError(f"Unsupported MultiLexSum split {split!r}. Expected one of {SPLITS} or 'all'.")
    records = fetch_jsonl(f"{split}.json", cache_dir, overwrite=overwrite)
    for record in records:
        record.setdefault("split", split)
    return records


def official_sources(records: list[dict[str, Any]], cache_dir: Path, *, overwrite: bool = False) -> dict[str, Any]:
    needed = {
        str(doc_id)
        for record in records
        for doc_id in (record.get("case_documents") or [])
    }
    sources_path = official_file("sources.json", cache_dir, overwrite=overwrite)
    return select_sources_from_json(sources_path, needed)


def candidate_records(records: list[dict[str, Any]], n: int | None = None) -> list[dict[str, Any]]:
    if not n:
        return records
    candidates = [
        record
        for record in records
        if _summary_long(record) and record.get("case_documents")
    ]
    return candidates[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=("official", "local"), help="Data source mode. Defaults to official unless local JSON paths are supplied.")
    p.add_argument("--records-json", "--records", dest="records_json")
    p.add_argument("--sources-json", "--sources", dest="sources_json")
    p.add_argument("--output-dir", default="output/multilexsum")
    p.add_argument("--cache-dir", default="data/multilexsum")
    p.add_argument("--n", type=int)
    p.add_argument("--split", default="test")
    p.add_argument("--overwrite-cache", action="store_true")
    p.add_argument("--skip-embeddings", action="store_true")
    args = p.parse_args()

    mode = args.source or ("local" if args.records_json or args.sources_json else "official")
    if mode == "local":
        if not args.records_json or not args.sources_json:
            raise SystemExit("Local MultiLexSum mode requires --records-json and --sources-json")
        try:
            records = load_records(Path(args.records_json))
            sources = load_sources(Path(args.sources_json))
        except Exception as exc:
            raise SystemExit(
                "Could not load local MultiLexSum files. "
                "Provide a split records file such as data/multilexsum/test.json and the official "
                "data/multilexsum/sources.json, or run official mode to download them. "
                f"Original error: {exc}"
            ) from exc
        records = filter_local_records_by_split(records, args.split)
        bundles = records_to_task_bundles(records, sources, split=None, n=args.n)
    else:
        try:
            cache_dir = Path(args.cache_dir)
            records = official_records(args.split, cache_dir, overwrite=args.overwrite_cache)
            selected_records = candidate_records(records, args.n)
            sources = official_sources(selected_records, cache_dir, overwrite=args.overwrite_cache)
            bundles = records_to_task_bundles(selected_records, sources, split=args.split, n=args.n)
            if args.n and len(bundles) < args.n and len(selected_records) < len(records):
                sources = official_sources(records, cache_dir, overwrite=False)
                bundles = records_to_task_bundles(records, sources, split=args.split, n=args.n)
        except Exception as exc:
            raise SystemExit(
                "Could not prepare official MultiLexSum input data. "
                "Run with network access so the importer can download from Hugging Face or the AI2 S3 mirror, "
                "or place the official split file and sources.json under data/multilexsum. "
                f"Original error: {exc}"
            ) from exc

    if not bundles:
        raise SystemExit(
            "MultiLexSum import produced 0 examples. "
            "Check --split, --n, and whether the records have summaries plus resolvable case_documents/source texts."
        )

    write_dataset(bundles, Path(args.output_dir))
    manifest = Path(args.output_dir) / "dataset_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.update({"dataset": "multilexsum", "source": mode, "split": args.split})
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

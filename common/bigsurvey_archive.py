from __future__ import annotations

import os
import re
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Sequence
from urllib.parse import parse_qs, urlencode, urljoin, urlparse
from urllib.request import HTTPCookieProcessor, Request, build_opener

import pandas as pd

REQUIRED_COLUMNS = {"title", "abstract", "section", "text", "bib_titles", "bib_abstracts"}
BIGSURVEY_ARCHIVES = {
    "original": ("1MnjQ2fQ_fJjcqKvIwj2w7P6IGh4GszXH", "original_survey_df.tar.gz"),
    "split": ("1S6v-xaCDND4ilK38sEpkfcOoMnffX7Zf", "split_survey_df.tar.gz"),
}


def _gdrive_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def extract_file_id(value: str) -> str:
    if re.fullmatch(r"[-\w]{20,}", value):
        return value
    parsed = urlparse(value)
    qs = parse_qs(parsed.query)
    if qs.get("id"):
        return qs["id"][0]
    m = re.search(r"/d/([^/]+)", value)
    if m:
        return m.group(1)
    raise ValueError(f"Could not extract Google Drive file id from {value!r}")


def _download(url: str, dest: Path, overwrite: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not overwrite and (
        tarfile.is_tarfile(dest) or zipfile.is_zipfile(dest)
    ):
        return dest
    opener = build_opener(HTTPCookieProcessor())
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with opener.open(req, timeout=600) as resp:
        ctype = resp.headers.get("Content-Type", "").lower()
        head = resp.read(2_000_000)
        looks_html = head.lstrip().lower().startswith((b"<!doctype html", b"<html"))
        if "text/html" not in ctype and not looks_html:
            with dest.open("wb") as f:
                f.write(head)
                shutil.copyfileobj(resp, f, length=1024 * 1024)
            return dest
        html = head.decode("utf-8", errors="ignore")
        token = None
        for pattern in (r"confirm=([0-9A-Za-z_\-]+)", r'name="confirm"\s+value="([^"]+)"'):
            m = re.search(pattern, html)
            if m:
                token = m.group(1)
                break
        if not token:
            raise RuntimeError("Google Drive returned HTML confirmation page without token")
        form_action = re.search(r'<form[^>]+action="([^"]+)"', html)
        hidden_inputs = dict(re.findall(r'<input[^>]+name="([^"]+)"\s+value="([^"]*)"', html))
        file_id = hidden_inputs.get("id") or extract_file_id(url)
        params = {"export": "download", "confirm": token, "id": file_id}
        if hidden_inputs.get("uuid"):
            params["uuid"] = hidden_inputs["uuid"]
        base_url = urljoin(url, form_action.group(1)) if form_action else "https://drive.google.com/uc"
        confirm_url = f"{base_url}?{urlencode(params)}"
        with opener.open(Request(confirm_url, headers={"User-Agent": "Mozilla/5.0"}), timeout=600) as resp2:
            with dest.open("wb") as f:
                shutil.copyfileobj(resp2, f, length=1024 * 1024)
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"Downloaded archive is empty: {dest}")
    return dest


def ensure_bigsurvey_archive(archive: str, work_dir: Path, overwrite: bool = False) -> Path:
    p = Path(archive)
    if p.exists() and p.stat().st_size > 0:
        return p
    key = archive.strip().lower()
    if key in BIGSURVEY_ARCHIVES:
        fid, name = BIGSURVEY_ARCHIVES[key]
        return _download(_gdrive_url(fid), work_dir / name, overwrite=overwrite)
    if archive.startswith(("http://", "https://")):
        name = Path(urlparse(archive).path).name or f"{extract_file_id(archive)}.tar.gz"
        if name in {"uc", "download"}:
            name = f"{extract_file_id(archive)}.tar.gz"
        return _download(archive, work_dir / name, overwrite=overwrite)
    fid = extract_file_id(archive)
    return _download(_gdrive_url(fid), work_dir / f"{fid}.tar.gz", overwrite=overwrite)


def _safe_member(base_dir: Path, name: str) -> bool:
    base = base_dir.resolve()
    target = (base_dir / name).resolve()
    return target == base or str(target).startswith(str(base) + os.sep)


def unpack_archive_safe(archive: Path, out_dir: Path, overwrite: bool = False) -> list[Path]:
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as tf:
            members = tf.getmembers()
            for m in members:
                if not _safe_member(out_dir, m.name):
                    raise RuntimeError(f"Unsafe path in tar archive: {m.name}")
            try:
                tf.extractall(out_dir, members=members, filter="data")
            except TypeError:
                tf.extractall(out_dir, members=members)
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if not _safe_member(out_dir, name):
                    raise RuntimeError(f"Unsafe path in zip archive: {name}")
            zf.extractall(out_dir)
    else:
        raise ValueError(f"Unsupported archive: {archive}")
    return [p for p in out_dir.rglob("*") if p.is_file()]


def find_bigsurvey_dataframe(files: Sequence[Path]) -> tuple[pd.DataFrame, Path]:
    for p in sorted(files, key=lambda x: x.stat().st_size if x.exists() else 0, reverse=True):
        try:
            suffixes = "".join(p.suffixes).lower()
            if suffixes.endswith((".pkl", ".pickle", ".pkl.gz", ".pickle.gz")):
                df = pd.read_pickle(p)
            elif suffixes.endswith(".parquet"):
                df = pd.read_parquet(p)
            elif suffixes.endswith((".csv", ".csv.gz")):
                df = pd.read_csv(p)
            elif suffixes.endswith((".jsonl", ".jsonl.gz")):
                df = pd.read_json(p, lines=True)
            else:
                continue
            if REQUIRED_COLUMNS.issubset(set(df.columns)):
                return df, p
        except Exception:
            continue
    raise FileNotFoundError(f"No BigSurvey dataframe with required rich columns found. Required columns: {sorted(REQUIRED_COLUMNS)}")

import io
import tarfile
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from common.bigsurvey_archive import find_bigsurvey_dataframe, unpack_archive_safe


def test_bigsurvey_archive_generic_loader(tmp_path: Path):
    inner = tmp_path / "nested" / "any_name.pkl"
    inner.parent.mkdir(parents=True)
    df = pd.DataFrame(
        [
            {"paper_id": "p1", "title": "T1", "abstract": "A1", "section": "S", "text": "X", "bib_titles": {"b": "B"}, "bib_abstracts": {"b": "BA"}},
            {"paper_id": "p2", "title": "T2", "abstract": "A2", "section": "S", "text": "Y", "bib_titles": {"b": "B"}, "bib_abstracts": {"b": "BB"}},
        ]
    )
    df.to_pickle(inner)

    archive = tmp_path / "archive.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(inner, arcname="random/deep/path/blob.pkl")

    extracted = unpack_archive_safe(archive, tmp_path / "out")
    loaded, path = find_bigsurvey_dataframe(extracted)

    assert len(loaded) == 2
    assert set(loaded["paper_id"]) == {"p1", "p2"}
    assert path.name == "blob.pkl"


def test_bigsurvey_archive_rejects_path_traversal_tar(tmp_path: Path):
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(name="../escape.pkl")
        payload = b"bad"
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    with pytest.raises(RuntimeError):
        unpack_archive_safe(archive, tmp_path / "out")


def test_bigsurvey_archive_rejects_path_traversal_zip(tmp_path: Path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.pkl", b"bad")
    with pytest.raises(RuntimeError):
        unpack_archive_safe(archive, tmp_path / "out_zip")

import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "verify_release", Path(__file__).parents[1] / "scripts" / "verify_release.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_release_checksum_accepts_matching_file(tmp_path):
    (tmp_path / "sample.zip").write_bytes(b"sample")
    sums = tmp_path / "SHA256SUMS.txt"
    sums.write_text(hashlib.sha256(b"sample").hexdigest() + "  sample.zip\n")
    assert module.verify(tmp_path, sums) == []


def test_release_checksum_detects_mismatch(tmp_path):
    (tmp_path / "sample.zip").write_bytes(b"modified")
    sums = tmp_path / "SHA256SUMS.txt"
    sums.write_text(hashlib.sha256(b"sample").hexdigest() + "  sample.zip\n")
    assert module.verify(tmp_path, sums) == ["MISMATCH sample.zip"]


def test_release_checksum_rejects_parent_path(tmp_path):
    sums = tmp_path / "SHA256SUMS.txt"
    sums.write_text("0" * 64 + "  ../private.zip\n")
    with pytest.raises(ValueError):
        module.verify(tmp_path, sums)

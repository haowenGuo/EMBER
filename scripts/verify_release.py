"""Verify public release ZIPs without downloading, extracting or executing them."""
from pathlib import Path
import argparse
import hashlib
import re


def verify(directory: Path, manifest: Path) -> list[str]:
    failures = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?([^/\\]+)", line)
        if not match or match[2] in {".", ".."}:
            raise ValueError("Invalid checksum entry or unsafe filename")
        expected, name = match.groups()
        target = directory / name
        if not target.is_file():
            failures.append(f"MISSING {name}")
            continue
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected.lower():
            failures.append(f"MISMATCH {name}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--checksums", type=Path)
    args = parser.parse_args()
    failures = verify(args.directory, args.checksums or args.directory / "SHA256SUMS.txt")
    print("\n".join(failures) if failures else "All listed release files passed SHA-256 verification.")
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()

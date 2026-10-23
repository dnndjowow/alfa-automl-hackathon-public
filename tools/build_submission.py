"""Build or audit the platform ZIP. Uses only the Python standard library."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 100_000_000
SECRET_PATTERNS = (
    rb"sk-[A-Za-z0-9_-]{20,}",
    rb"gh[pousr]_[A-Za-z0-9]{20,}",
    rb"github_pat_[A-Za-z0-9_]{20,}",
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
)


class ReleaseError(ValueError):
    pass


def validate_content(name: str, data: bytes) -> None:
    # Never include the matched value or source line in diagnostics.
    key = os.environ.get("OPENAI_API_KEY", "")
    if any(re.search(pattern, data) for pattern in SECRET_PATTERNS) or (
        len(key) >= 12 and key.encode() in data
    ):
        raise ReleaseError(f"{name}: possible embedded credential; archive rejected")
    if name.endswith(".py"):
        try:
            compile(data, name, "exec", dont_inherit=True)
        except (SyntaxError, ValueError) as error:
            line = getattr(error, "lineno", "?")
            raise ReleaseError(f"{name}: invalid Python syntax at line {line}") from None


def audit_archive(archive: Path, requirements: Path, max_bytes: int = MAX_BYTES) -> list[str]:
    if archive.stat().st_size > max_bytes:
        raise ReleaseError("ZIP exceeds the 100 MB case limit")
    names: list[str] = []
    with zipfile.ZipFile(archive) as package:
        for item in package.infolist():
            name = item.filename
            if (
                item.is_dir()
                or PurePosixPath(name).name != name
                or "\\" in name
                or name in {".", ".."}
                or not (name.endswith(".py") or name == "requirements.txt")
                or stat.S_ISLNK(item.external_attr >> 16)
                or item.flag_bits & 1
            ):
                raise ReleaseError(f"Disallowed ZIP entry: {name}")
            if name.casefold() in {n.casefold() for n in names}:
                raise ReleaseError(f"Duplicate ZIP filename: {name}")
            data = package.read(item)  # Also verifies CRC.
            validate_content(name, data)
            names.append(name)
        if "solution.py" not in names or "requirements.txt" not in names:
            raise ReleaseError("ZIP must contain solution.py and requirements.txt in its root")
        if package.read("requirements.txt") != requirements.read_bytes():
            raise ReleaseError("requirements.txt differs from the official reference copy")
    return names


def build_archive(root: Path, output: Path, modules: list[Path] | None = None) -> list[str]:
    root = root.resolve()
    requirements = root / "reference" / "requirements.txt"
    if not (root / "solution.py").is_file():
        raise ReleaseError("solution.py is missing: the real agent is not ready for packaging")
    if not requirements.is_file():
        raise ReleaseError("reference/requirements.txt is missing")
    sources = sorted(root.glob("*.py")) + list(modules or [])
    contents: dict[str, bytes] = {}
    for source in sources:
        source = source if source.is_absolute() else root / source
        resolved = source.resolve()
        if not resolved.is_relative_to(root) or source.is_symlink():
            raise ReleaseError("Module must be a regular file inside the repository")
        if source.suffix != ".py" or not source.is_file():
            raise ReleaseError(f"Not a Python module: {source.name}")
        if source.name.casefold() in {n.casefold() for n in contents}:
            raise ReleaseError(f"Module filename collision: {source.name}")
        data = source.read_bytes()
        validate_content(source.name, data)
        contents[source.name] = data
    contents["requirements.txt"] = requirements.read_bytes()
    validate_content("requirements.txt", contents["requirements.txt"])
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as temporary:
        candidate = Path(temporary.name)
    try:
        with zipfile.ZipFile(candidate, "w", compression=zipfile.ZIP_DEFLATED) as package:
            for name, data in sorted(contents.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                package.writestr(info, data)
        names = audit_archive(candidate, requirements)
        candidate.replace(output)
        return names
    finally:
        candidate.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "solution.zip")
    parser.add_argument("--module", action="append", type=Path, default=[],
                        help="Additional repo-relative .py file; stored flat, imports are not rewritten")
    parser.add_argument("--check", type=Path, help="Audit an existing ZIP without running its code")
    parser.add_argument("--extract-to", type=Path, help="Extract a checked ZIP into a NEW directory for a clean run")
    args = parser.parse_args()
    try:
        archive = args.check or args.output
        if args.check:
            names = audit_archive(archive, args.root / "reference" / "requirements.txt")
        else:
            names = build_archive(args.root, archive, args.module)
        print(f"OK: {archive}")
        print(f"Bytes: {archive.stat().st_size}; SHA256: {hashlib.sha256(archive.read_bytes()).hexdigest()}")
        print("Files: " + ", ".join(names))
        if args.extract_to:
            args.extract_to.mkdir(parents=True, exist_ok=False)
            with zipfile.ZipFile(archive) as package:
                package.extractall(args.extract_to)
            print(f"Clean directory: {args.extract_to.resolve()}")
        print("Checked: structure, requirements bytes, Python syntax, common credential patterns.")
        print("NOT checked: runtime imports, API, ML quality or the end-to-end solution.")
        return 0
    except (ReleaseError, OSError, zipfile.BadZipFile, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

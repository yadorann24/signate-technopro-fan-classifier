"""Check the candidate folder without inspecting Git or transmitting its contents.

This intentionally checks the filesystem, including ignored data directories.
It is a narrow allowlist and common-secret check, not a proof of publication rights.
"""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path
import re

ALLOWED = (
    "README.md",
    "pyproject.toml",
    ".gitignore",
    "requirements-cpu.lock",
    "requirements-stratification-cpu.lock",
    "docs/*.md",
    "configs/*.json",
    "src/fan_classifier/*.py",
    "tests/test_*.py",
    "scripts/*.py",
    ".github/workflows/*.yml",
)
LOCAL_ONLY = {".git", ".venv", "venv", "__pycache__", ".ruff_cache", ".pytest_cache"}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
}


def inspect_public_files(root: Path) -> list[str]:
    issues = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in LOCAL_ONLY or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_symlink():
            issues.append(f"Symlink is not a publishable source file: {relative}")
            continue
        if path.is_dir():
            continue
        name = relative.as_posix()
        if not any(fnmatch.fnmatchcase(name, pattern) for pattern in ALLOWED):
            issues.append(f"Outside publication allowlist: {relative}")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeError:
            issues.append(f"Non-text file: {relative}")
            continue
        if "\x00" in content:
            issues.append(f"Binary content: {relative}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                issues.append(f"Possible {label} in {relative} (value not printed)")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("--root must be an existing directory")
    issues = inspect_public_files(args.root)
    for issue in issues:
        print(issue)
    if issues:
        print(f"Publication check failed: {len(issues)} issue(s).")
        return 1
    print("Publication allowlist and common-secret checks passed (Git history not inspected).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

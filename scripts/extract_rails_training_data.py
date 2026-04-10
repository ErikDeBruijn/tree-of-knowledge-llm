#!/usr/bin/env python3
"""Extract Ruby training data from steveclarke/real-world-rails submodules.

Walks apps/ and engines/ in a real-world-rails checkout, filters noise
(vendor/, tmp/, schema.rb, ...), and emits a JSONL file matching the
existing ruby_domain.jsonl format: one {"text": "<file contents>"} per line.

Usage:
    python3 scripts/extract_rails_training_data.py \\
        --source ~/github.com/steveclarke/real-world-rails \\
        --out ~/Dev/AI/training_data/rails_realworld.jsonl
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Directory segments that should be skipped entirely.
SKIP_DIRS = {
    "vendor", "node_modules", "tmp", "log", "coverage", "public",
    ".bundle", ".git", "storage", "dist", "build", "cache",
}

# Specific files that are either generated or not useful for training.
SKIP_FILENAMES = {
    "schema.rb",         # generated DB schema
    "structure.sql",     # generated
    "Gemfile.lock",      # not .rb but just in case
    "routes.rb",         # keep? routes are actually interesting — keep.
}
# Strip the routes.rb entry; it is idiomatic and worth training on.
SKIP_FILENAMES.discard("routes.rb")

MIN_LEN = 200
MAX_LEN = 10_000


def iter_ruby_files(root: Path):
    """Yield candidate .rb file paths under apps/ and engines/."""
    for top in ("apps", "engines"):
        base = root / top
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            # In-place prune of skipped dirs
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for fn in filenames:
                if not fn.endswith(".rb"):
                    continue
                if fn in SKIP_FILENAMES:
                    continue
                yield Path(dirpath) / fn


def read_text(path: Path) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as f:
            return f.read()
    except (UnicodeDecodeError, OSError):
        return None


import re

# Patterns to strip from Ruby training data.
_RUBOCOP_LINE = re.compile(r"^\s*#\s*rubocop:(enable|disable)\b.*$", re.MULTILINE)
_FROZEN_LITERAL = re.compile(r"^\s*#\s*frozen_string_literal:.*$", re.MULTILINE)
_MAGIC_COMMENT = re.compile(r"^\s*#\s*(encoding|warn_indent|typed):.*$", re.MULTILINE)
_YARD_TAG = re.compile(r"^\s*#\s*@(param|return|raise|example|option|yield|see|note|api|deprecated)\b.*$", re.MULTILINE)
_BLANK_COMMENT = re.compile(r"^\s*#\s*$", re.MULTILINE)
# Consecutive blank lines → single blank line.
_MULTI_BLANK = re.compile(r"\n{3,}")


def scrub_ruby(text: str) -> str:
    """Remove noise that pollutes training signal without losing logic."""
    text = _RUBOCOP_LINE.sub("", text)
    text = _FROZEN_LITERAL.sub("", text)
    text = _MAGIC_COMMENT.sub("", text)
    text = _YARD_TAG.sub("", text)
    text = _BLANK_COMMENT.sub("", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def good_content(text: str) -> bool:
    if not (MIN_LEN <= len(text) <= MAX_LEN):
        return False
    # Reject files that are mostly non-printable.
    printable = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
    if printable / len(text) < 0.98:
        return False
    # Reject files that are likely minified / one giant line.
    lines = text.count("\n")
    if lines < 3:
        return False
    # Reject files that are mostly comments (>40% of non-blank lines).
    all_lines = [l for l in text.split("\n") if l.strip()]
    if all_lines:
        comment_lines = sum(1 for l in all_lines if l.strip().startswith("#"))
        if comment_lines / len(all_lines) > 0.4:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True,
                    help="Path to a real-world-rails checkout (with initialised submodules)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSONL path")
    ap.add_argument("--limit", type=int, default=0,
                    help="Optional cap on number of output records (0 = no limit)")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    total_scanned = 0
    total_kept = 0
    total_bytes = 0
    per_app_kept: dict[str, int] = {}

    with open(args.out, "w", encoding="utf-8") as outf:
        for path in iter_ruby_files(args.source):
            total_scanned += 1
            text = read_text(path)
            if text is None:
                continue
            text = scrub_ruby(text)
            if not good_content(text):
                continue
            h = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            outf.write(json.dumps({"text": text}, ensure_ascii=False))
            outf.write("\n")
            total_kept += 1
            total_bytes += len(text)
            # Track which app it came from (first segment after apps/ or engines/)
            try:
                rel = path.relative_to(args.source)
                parts = rel.parts
                if len(parts) >= 2 and parts[0] in ("apps", "engines"):
                    key = f"{parts[0]}/{parts[1]}"
                    per_app_kept[key] = per_app_kept.get(key, 0) + 1
            except ValueError:
                pass
            if args.limit and total_kept >= args.limit:
                break
            if total_kept % 5000 == 0:
                print(f"  ... {total_kept} kept / {total_scanned} scanned", file=sys.stderr)

    print(f"Scanned: {total_scanned}")
    print(f"Kept (deduped): {total_kept}")
    print(f"Bytes: {total_bytes:,} ({total_bytes/1e6:.1f} MB)")
    print(f"Output: {args.out}")
    print(f"Apps/engines contributing: {len(per_app_kept)}")
    top = sorted(per_app_kept.items(), key=lambda kv: -kv[1])[:10]
    if top:
        print("Top contributors:")
        for name, n in top:
            print(f"  {n:>6}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

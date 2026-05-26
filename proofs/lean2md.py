#!/usr/bin/env python3
"""Extract markdown from Lean 4 source files.

Mimics mdgen's behavior: module-level `/- ! ... -/` blocks and
per-declaration `/-- ... -/` doc-strings become prose; everything else
becomes fenced Lean code blocks.  Blank runs between doc-comment and
declaration are collapsed so the rendered output reads naturally.

Usage:
    python3 lean2md.py MODULE_DIR OUTPUT.md
    python3 lean2md.py BvdProofs BvdProofs.md
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Order must match the import DAG (Foundations first).
MODULE_ORDER = [
    "Foundations",
    "Quadrature",
    "Expectations",
    "GammaCDF",
    "Interpolation",
    "Forecast",
]


def extract_blocks(src: str) -> list[tuple[str, str]]:
    """Return a list of (kind, text) pairs.

    kind is one of:
      "module_doc" – the /-! … -/ block
      "doc"        – a /-- … -/ block
      "code"       – everything else (lean source)
    """
    blocks: list[tuple[str, str]] = []
    pos = 0
    n = len(src)

    while pos < n:
        # Look for the next doc-comment opener.
        mod_idx = src.find("/-!", pos)
        doc_idx = src.find("/--", pos)

        # Pick the earlier one.
        next_idx = -1
        kind = ""
        if mod_idx != -1 and (doc_idx == -1 or mod_idx < doc_idx):
            next_idx = mod_idx
            kind = "module_doc"
        elif doc_idx != -1:
            next_idx = doc_idx
            kind = "doc"

        if next_idx == -1:
            # No more doc comments — rest is code.
            rest = src[pos:].strip()
            if rest:
                blocks.append(("code", rest))
            break

        # Emit any code before the doc-comment.
        before = src[pos:next_idx].strip()
        if before:
            blocks.append(("code", before))

        # Find the matching close.
        close = src.find("-/", next_idx + 3)
        if close == -1:
            # Malformed: treat rest as code.
            blocks.append(("code", src[next_idx:].strip()))
            break

        opener_len = 3  # /-- or /-!
        raw = src[next_idx + opener_len : close].strip()
        blocks.append((kind, raw))
        pos = close + 2

    return blocks


def blocks_to_markdown(blocks: list[tuple[str, str]], source_file: str) -> str:
    """Convert extracted blocks into markdown text."""
    parts: list[str] = []

    for kind, text in blocks:
        if kind in ("module_doc", "doc"):
            parts.append(text)
            parts.append("")
        elif kind == "code":
            # Skip import lines and boilerplate openers.
            lines = text.splitlines()
            filtered = [
                l
                for l in lines
                if not re.match(
                    r"^\s*(import |open |noncomputable section|namespace |end )\s*", l
                )
            ]
            code = "\n".join(filtered).strip()
            if code:
                parts.append(f"```lean\n{code}\n```")
                parts.append("")

    return "\n".join(parts)


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} MODULE_DIR OUTPUT.md", file=sys.stderr)
        sys.exit(1)

    module_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    sections: list[str] = []

    for name in MODULE_ORDER:
        path = module_dir / f"{name}.lean"
        if not path.exists():
            print(f"Warning: {path} not found, skipping", file=sys.stderr)
            continue
        src = path.read_text()
        blocks = extract_blocks(src)
        md = blocks_to_markdown(blocks, name)
        if md.strip():
            sections.append(md)

    output_path.write_text("\n---\n\n".join(sections) + "\n")
    print(f"Generated {output_path}")


if __name__ == "__main__":
    main()

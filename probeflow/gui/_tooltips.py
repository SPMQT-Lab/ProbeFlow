"""Shared tooltip helpers for ProbeFlow GUI widgets."""

from __future__ import annotations

import textwrap


def tip(text: str, width: int = 50) -> str:
    """Word-wrap tooltip text into short lines that stay near the cursor.

    Qt renders a plain-text tooltip that contains no newlines as a single very
    long line, which can stretch right across the screen and away from the
    pointer.  Wrapping every tooltip to a fixed column keeps the explanation
    compact and anchored under the cursor no matter how long it is.

    Any explicit newlines the caller writes are preserved as line breaks, so
    deliberately separated sentences / paragraphs stay separated; each such
    segment is independently wrapped to ``width`` columns.  Blank lines are
    kept as paragraph gaps.
    """
    out_lines: list[str] = []
    for segment in text.split("\n"):
        if segment.strip() == "":
            out_lines.append("")
        else:
            out_lines.append(textwrap.fill(segment.strip(), width=width))
    return "\n".join(out_lines)

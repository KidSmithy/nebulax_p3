"""
acv2.reporting
===========================================================================
Tiny Markdown table writer, so the evaluation reports have no dependency on
`tabulate` beyond what the project already needs.
===========================================================================
"""
from __future__ import annotations

import pandas as pd


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        text = f"{value:.3f}".rstrip("0").rstrip(".") if abs(value) < 1e6 else f"{value:.3g}"
        return text
    # pipes are the separator used inside ranked_cars, so they must be escaped
    return str(value).replace("|", "\\|")


def df_to_md(frame: pd.DataFrame, index: bool = False) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table."""
    data = frame.reset_index() if index else frame
    header = [str(c) for c in data.columns]
    rows = [[_fmt(v) for v in row] for row in data.itertuples(index=False, name=None)]
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
              for i in range(len(header))]
    lines = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |",
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(header))) + " |")
    return "\n".join(lines)

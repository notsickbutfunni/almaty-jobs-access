"""Bytes for the dashboard's download buttons (CSV and Parquet)."""

from __future__ import annotations

import io

import pandas as pd


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """UTF-8 CSV with a BOM, so Excel opens Cyrillic headers correctly."""
    return frame.to_csv(index=False).encode("utf-8-sig")


def parquet_bytes(frame: pd.DataFrame) -> bytes:
    """The frame as a Parquet file in memory."""
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()

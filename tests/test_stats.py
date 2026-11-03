"""Tests for district statistics: Taldau parsing, validation, overwrite guard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from almaty_access.config import Config
from almaty_access.download import stats

#: Shape taken from a real GetIndexTreeData response (probed 2026-11-03):
#: keys 'y<MM><YYYY>' hold values; bare period keys hold period ids.
CANNED_PAYLOAD = [
    {
        "id": "268023",
        "text": "АЛМАЛИНСКИЙ РАЙОН",
        "leaf": "true",
        "measureName": "",
        "y122024": "131560",
        "122024": "1031",
        "y122025": "132637",
        "122025": "1068",
    },
    {
        "id": "17807940",
        "text": "НАУРЫЗБАЙСКИЙ РАЙОН",
        "y122024": "92315",
        "122024": "1031",
    },
    {"id": "999999", "text": "SOMEWHERE ELSE", "y122024": "1"},
]


def test_parse_taldau_rows_from_canned_payload() -> None:
    rows = stats._parse_taldau_rows(CANNED_PAYLOAD, "employed_residents", "persons", "note", "src")
    frame = pd.DataFrame(rows)
    # Unknown term id 999999 is ignored; period ids like '122024' are not values.
    assert set(frame["district"]) == {"almaly", "nauryzbay"}
    assert len(frame) == 3
    almaly_2024 = frame.query("district == 'almaly' and year == 2024")["value"].iloc[0]
    assert almaly_2024 == 131560.0
    assert set(frame["month"]) == {12}
    assert (frame["indicator"] == "employed_residents").all()


def _write_stats(config: Config, table: pd.DataFrame) -> Path:
    path = config.paths.raw / stats.FILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def _valid_table(config: Config) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "district": [d.key for d in config.districts],
            "year": 2024,
            "indicator": "payroll_jobs_located",
            "value": 1000.0,
        }
    )


def test_check_district_stats_accepts_valid(tmp_config: Config) -> None:
    _write_stats(tmp_config, _valid_table(tmp_config))
    assert stats.check_district_stats(tmp_config).exists()


def test_check_district_stats_missing_file_has_instructions(tmp_config: Config) -> None:
    with pytest.raises(FileNotFoundError, match=r"stat\.gov\.kz"):
        stats.check_district_stats(tmp_config)


def test_check_district_stats_rejects_bad_schema(tmp_config: Config) -> None:
    table = _valid_table(tmp_config).drop(columns=["value"])
    _write_stats(tmp_config, table)
    with pytest.raises(ValueError, match="missing required columns"):
        stats.check_district_stats(tmp_config)


def test_check_district_stats_rejects_unknown_district(tmp_config: Config) -> None:
    table = _valid_table(tmp_config)
    table.loc[0, "district"] = "atlantis"
    _write_stats(tmp_config, table)
    with pytest.raises(ValueError, match="unknown district keys"):
        stats.check_district_stats(tmp_config)


def test_check_district_stats_rejects_non_numeric_values(tmp_config: Config) -> None:
    table = _valid_table(tmp_config)
    table["value"] = table["value"].astype(object)
    table.loc[0, "value"] = "many"
    _write_stats(tmp_config, table)
    with pytest.raises(ValueError, match="non-numeric"):
        stats.check_district_stats(tmp_config)


def test_ensure_district_stats_never_overwrites_existing(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manually curated CSV must never be replaced by a refetch."""
    path = _write_stats(tmp_config, _valid_table(tmp_config))
    original = path.read_text(encoding="utf-8")

    def _explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("fetch_district_stats must not be called when the file exists")

    monkeypatch.setattr(stats, "fetch_district_stats", _explode)
    result = stats.ensure_district_stats(tmp_config, force=True)
    assert result == path
    assert path.read_text(encoding="utf-8") == original


def test_ensure_district_stats_fetch_failure_gives_instructions(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail(config: Config) -> None:
        raise RuntimeError("API offline")

    monkeypatch.setattr(stats, "fetch_district_stats", _fail)
    with pytest.raises(FileNotFoundError, match="Taldau fetch also failed"):
        stats.ensure_district_stats(tmp_config)


def test_write_synthetic_stats_matches_real_schema(tmp_config: Config) -> None:
    path = stats.write_synthetic_stats(tmp_config)
    table = pd.read_csv(path)
    assert set(stats.REQUIRED_COLUMNS) <= set(table.columns)
    assert "month" in table.columns
    assert set(table["indicator"]) == {"employed_residents", "payroll_jobs_located"}
    assert (table["source"] == "synthetic-for-tests").all()
    # Deterministic under the fixed seed.
    again = pd.read_csv(stats.write_synthetic_stats(tmp_config))
    pd.testing.assert_frame_equal(table, again)
    # And it passes the validator.
    assert stats.check_district_stats(tmp_config) == path

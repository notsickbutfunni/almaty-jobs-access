"""District statistics: fetched from the Taldau API, or collected by hand.

Two verified official indicators exist per district (research 2026-11-03,
see DECISIONS.md): employed residents (LFS, annual) and payroll jobs at
the workplace's district (enterprise reports, quarterly) - the latter is
the Phase 3 training target. ``ensure_district_stats`` fetches both from
the Bureau of National Statistics' Taldau API when the CSV is absent; an
existing file is only validated, never overwritten (it may be manually
curated). The pipeline never fabricates statistics; the sample dataset
gets clearly labelled synthetic numbers instead.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from almaty_access.config import Config

logger = logging.getLogger(__name__)

FILE_NAME = "district_stats.csv"

REQUIRED_COLUMNS = ("district", "year", "indicator", "value")
OPTIONAL_COLUMNS = ("month", "unit", "source", "notes")

USER_AGENT = "almaty-access-pipeline/0.1 (open-data research)"
TALDAU_URL = "https://taldau.stat.gov.kz/ru/Api/GetIndexTreeData"

#: Taldau region-dictionary term id -> our district key (verified live).
TALDAU_DISTRICT_TERMS = {
    "268023": "almaly",
    "268025": "alatau",
    "268027": "auezov",
    "268029": "bostandyk",
    "268031": "zhetysu",
    "268033": "medeu",
    "268035": "turksib",
    "17807940": "nauryzbay",
}

#: Verified GetIndexTreeData query parameters per indicator.
TALDAU_INDICATORS: dict[str, dict[str, Any]] = {
    "employed_residents": {
        "params": {
            "p_measure_id": 1,
            "p_index_id": 702840,
            "p_period_id": 7,
            "p_terms": "741880,741917,741935,3805694,4197331,741885",
            "p_term_id": 741880,
            "p_dicIds": "67,749,576,1773,1793,3028",
            "idx": 0,
            "p_parent_id": 268020,
        },
        "unit": "persons",
        "notes": (
            "Labour-force household survey (form T-001): employed RESIDENTS "
            "of the district, not jobs located there"
        ),
    },
    "payroll_jobs_located": {
        "params": {
            "p_measure_id": 1,
            "p_index_id": 18935544,
            "p_period_id": 5,
            "p_terms": "741880,741885,3629946,741907",
            "p_term_id": 741880,
            "p_dicIds": "68,859,2813,59",
            "idx": 1,
            "p_parent_id": 268020,
        },
        "unit": "persons",
        "notes": (
            "Payroll employees incl. small enterprises (forms 1-T, 2-MP), "
            "attributed to the district where the enterprise reports: counts "
            "JOBS located there; excludes individual entrepreneurs and "
            "self-employed"
        ),
    },
}

_PERIOD_KEY_RE = re.compile(r"^y(\d{2})(\d{4})$")

MISSING_TEMPLATE = """district statistics file not found: {path}

This is the only manually collected input (README §5.1). To create it:
1. Open stat.gov.kz (Bureau of National Statistics, Almaty city
   department) and/or the Taldau portal (taldau.stat.gov.kz).
2. Find per-district tables for Almaty: employed population and/or
   number of active enterprises (see data/README.md for known tables).
3. Save a CSV with columns: district,year,indicator,value[,unit,source,notes]
   - district keys: {district_keys}
   - indicator examples: employed, enterprises_active
   - document in 'notes' whether the number counts RESIDENTS employed or
     JOBS LOCATED in the district - they are not the same thing.
"""


def check_district_stats(config: Config) -> Path:
    """Validate data/raw/district_stats.csv; raise with instructions if absent."""
    path = config.paths.raw / FILE_NAME
    district_keys = sorted(d.key for d in config.districts)
    if not path.exists():
        raise FileNotFoundError(
            MISSING_TEMPLATE.format(path=path, district_keys=", ".join(district_keys))
        )
    table = pd.read_csv(path)
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in table.columns]
    if missing_columns:
        raise ValueError(f"{path}: missing required columns: {missing_columns}")
    unknown = sorted(set(table["district"]) - set(district_keys))
    if unknown:
        raise ValueError(f"{path}: unknown district keys {unknown}; expected {district_keys}")
    if not pd.to_numeric(table["value"], errors="coerce").notna().all():
        raise ValueError(f"{path}: 'value' column contains non-numeric entries")
    for indicator, group in table.groupby("indicator"):
        covered = set(group["district"])
        absent = sorted(set(district_keys) - covered)
        if absent:
            logger.warning("district_stats: indicator '%s' misses districts: %s", indicator, absent)
    logger.info(
        "district_stats: %d rows, indicators: %s",
        len(table),
        sorted(table["indicator"].unique().tolist()),
    )
    return path


def _parse_taldau_rows(
    payload: list[dict], indicator: str, unit: str, notes: str, source: str
) -> list[dict]:
    """Flatten a GetIndexTreeData response into long-format rows.

    Each district entry carries value keys like ``y122024`` = value for
    period 12.2024 (annual series stamp December; quarterly series use the
    quarter's last month).
    """
    rows: list[dict] = []
    for entry in payload:
        district_key = TALDAU_DISTRICT_TERMS.get(str(entry.get("id")))
        if district_key is None:
            continue
        for key, raw in entry.items():
            match = _PERIOD_KEY_RE.match(key)
            if not match or raw in (None, ""):
                continue
            try:
                value = float(str(raw).replace(" ", "").replace(",", "."))
            except ValueError:
                continue
            rows.append(
                {
                    "district": district_key,
                    "year": int(match.group(2)),
                    "month": int(match.group(1)),
                    "indicator": indicator,
                    "value": value,
                    "unit": unit,
                    "source": source,
                    "notes": notes,
                }
            )
    return rows


def fetch_district_stats(config: Config) -> pd.DataFrame:
    """Fetch both verified indicators from the Taldau API (needs network)."""
    frames = []
    for indicator, spec in TALDAU_INDICATORS.items():
        response = requests.get(
            TALDAU_URL,
            params=spec["params"],
            timeout=120,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        source = f"taldau.stat.gov.kz GetIndexTreeData p_index_id={spec['params']['p_index_id']}"
        rows = _parse_taldau_rows(response.json(), indicator, spec["unit"], spec["notes"], source)
        if not rows:
            raise RuntimeError(f"Taldau returned no parsable rows for '{indicator}'")
        frames.append(pd.DataFrame(rows))
    table = pd.concat(frames, ignore_index=True)
    expected = {d.key for d in config.districts}
    for indicator, group in table.groupby("indicator"):
        missing = expected - set(group["district"])
        if missing:
            raise RuntimeError(f"Taldau '{indicator}' is missing districts: {sorted(missing)}")
    logger.info(
        "taldau: fetched %d rows for %d indicators", len(table), table["indicator"].nunique()
    )
    return table.sort_values(["indicator", "year", "month", "district"]).reset_index(drop=True)


def ensure_district_stats(config: Config, *, force: bool = False) -> Path:
    """Validate the stats CSV, fetching it from Taldau when it is absent.

    An existing file is never overwritten (it may be manually curated), so
    ``force`` is deliberately ignored here; delete the file to refetch.
    """
    del force
    path = config.paths.raw / FILE_NAME
    if path.exists():
        return check_district_stats(config)
    logger.info("district_stats: file absent, trying the Taldau API")
    try:
        table = fetch_district_stats(config)
    except Exception as exc:
        district_keys = ", ".join(sorted(d.key for d in config.districts))
        raise FileNotFoundError(
            MISSING_TEMPLATE.format(path=path, district_keys=district_keys)
            + f"\n(the automatic Taldau fetch also failed: {exc})"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    logger.info("district_stats: wrote %s from Taldau", path)
    return check_district_stats(config)


def write_synthetic_stats(config: Config) -> Path:
    """Write clearly labelled synthetic district stats (sample dataset only).

    Values are deterministic (seeded) and marked source='synthetic-for-tests'
    so they can never be mistaken for real statistics.
    """
    path = config.paths.raw / FILE_NAME
    rng = np.random.default_rng(config.random_seed)
    rows = []
    # Mirror the real Taldau output exactly: same indicator names, same
    # columns (incl. month), so Phase 3 code sees one schema everywhere.
    for district in config.districts:
        employed = int(rng.integers(60_000, 200_000))
        rows.append((district.key, 2024, 12, "employed_residents", employed, "persons"))
        base_jobs = int(rng.integers(50_000, 260_000))
        for month in (3, 6, 9, 12):
            jitter = int(rng.integers(-3_000, 3_000))
            rows.append(
                (district.key, 2024, month, "payroll_jobs_located", base_jobs + jitter, "persons")
            )
    table = pd.DataFrame(rows, columns=["district", "year", "month", "indicator", "value", "unit"])
    table["source"] = "synthetic-for-tests"
    table["notes"] = "FAKE numbers for offline tests; replace with stat.gov.kz data"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    logger.info("wrote synthetic district stats: %s", path)
    return path

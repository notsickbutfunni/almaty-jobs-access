"""Find and load the tables the dashboard reads.

The app can run from three places, tried in this order (D-069):

1. ``ALMATY_ACCESS_DATA`` environment variable (or an explicit override),
2. the committed ``data/app/`` bundle written by ``almaty-access export``
   — this is what free hosting (Streamlit Community Cloud) serves,
3. a full local pipeline run in ``data/processed/``, then the sample.

Everything is read once into one frozen ``Tables`` object; the Streamlit
layer caches it with ``st.cache_data``. The dashboard works at the
analysis resolution only (r8) — the README's free-hosting budget.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from almaty_access.access.build import ACCESS_FILE_TEMPLATE, TT_FILE_TEMPLATE
from almaty_access.access.build import REPORT_FILE as ACCESS_REPORT_FILE
from almaty_access.equity.build import (
    CALIBRATED,
    DISTRICTS_FILE,
    HEX_FILE_TEMPLATES,
    LORENZ_FILE,
    RASTER,
    SUMMARY_FILE,
)
from almaty_access.equity.build import REPORT_FILE as EQUITY_REPORT_FILE
from almaty_access.model.build import JOBS_FILE_TEMPLATE, MODEL_REPORT_FILE
from almaty_access.scenarios.build import GAINS_FILE, SCENARIO_FILE_TEMPLATE
from almaty_access.scenarios.build import REPORT_FILE as SCENARIOS_REPORT_FILE
from almaty_access.scenarios.build import SUMMARY_FILE as SCENARIOS_SUMMARY_FILE

#: The dashboard's single analysis resolution (config h3.analysis_resolution).
RES = 8

#: Travel modes with an r8 time matrix (access/build.py MODE_RESOLUTIONS).
MODES: tuple[str, ...] = ("walk", "bike", "transit", "car")

#: Population weightings, in the fixed display order (D-058).
WEIGHTINGS: tuple[str, ...] = (RASTER, CALIBRATED)

ENV_VAR = "ALMATY_ACCESS_DATA"

#: Files that must all be present for a directory to serve the dashboard.
#: Scenario parquets are discovered through scenarios_report.json.
REQUIRED_FILES: tuple[str, ...] = (
    ACCESS_FILE_TEMPLATE.format(res=RES),
    JOBS_FILE_TEMPLATE.format(res=RES),
    HEX_FILE_TEMPLATES[RASTER].format(res=RES),
    HEX_FILE_TEMPLATES[CALIBRATED].format(res=RES),
    SUMMARY_FILE,
    LORENZ_FILE,
    DISTRICTS_FILE,
    SCENARIOS_SUMMARY_FILE,
    GAINS_FILE,
    *(TT_FILE_TEMPLATE.format(mode=mode, res=RES) for mode in MODES),
    ACCESS_REPORT_FILE,
    EQUITY_REPORT_FILE,
    SCENARIOS_REPORT_FILE,
    MODEL_REPORT_FILE,
)


@dataclass(frozen=True)
class Tables:
    """Every table and report the dashboard shows, loaded once."""

    data_dir: Path
    access: pd.DataFrame
    jobs: pd.DataFrame
    equity: dict[str, pd.DataFrame]
    summary: pd.DataFrame
    lorenz: pd.DataFrame
    districts: pd.DataFrame
    gains: pd.DataFrame
    scenarios: dict[str, pd.DataFrame]
    tt: dict[str, pd.DataFrame]
    access_report: dict[str, Any]
    equity_report: dict[str, Any]
    scenarios_report: dict[str, Any]
    model_report: dict[str, Any]

    @property
    def scenario_names(self) -> tuple[str, ...]:
        """Scenario names in the report's (sorted) order."""
        return tuple(self.scenarios)

    def scenario_title(self, name: str) -> str:
        return str(self.scenarios_report["scenarios"][name]["title"])

    def scenario_modes(self, name: str) -> tuple[str, ...]:
        """Travel modes a scenario changes (its delta columns exist)."""
        return tuple(self.scenarios_report["scenarios"][name]["modes"])


def missing_files(data_dir: Path) -> list[str]:
    """Required files absent from ``data_dir`` (empty = usable)."""
    return [name for name in REQUIRED_FILES if not (data_dir / name).is_file()]


def candidate_dirs(root: Path, override: str | Path | None = None) -> list[Path]:
    """Directories to try, most preferred first (see module docstring)."""
    candidates: list[Path] = []
    env_value = os.environ.get(ENV_VAR, "").strip()
    for value in (override, env_value or None):
        if value:
            candidates.append(Path(value))
    candidates += [
        root / "data" / "app",
        root / "data" / "processed",
        root / "data" / "sample" / "processed",
    ]
    return candidates


def resolve_data_dir(root: Path, override: str | Path | None = None) -> Path:
    """First candidate directory that holds every required file.

    An explicit ``override`` (or the environment variable) must be
    complete on its own: silently falling back from a directory the user
    named would mask a half-built export.
    """
    candidates = candidate_dirs(root, override)
    explicit = len(candidates) > 3  # an override / env dir was prepended
    for i, candidate in enumerate(candidates):
        absent = missing_files(candidate)
        if not absent:
            return candidate
        if explicit and i == 0:
            raise FileNotFoundError(
                f"data directory {candidate} is missing required files: "
                f"{', '.join(absent[:5])}{'...' if len(absent) > 5 else ''}"
            )
    tried = "; ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "no complete dashboard data directory found. Run the pipeline and "
        "`almaty-access export`, or set the ALMATY_ACCESS_DATA environment "
        f"variable. Tried: {tried}"
    )


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_tables(data_dir: Path) -> Tables:
    """Read every dashboard input from one directory (see REQUIRED_FILES)."""
    data_dir = Path(data_dir)
    absent = missing_files(data_dir)
    if absent:
        raise FileNotFoundError(f"{data_dir} is missing: {', '.join(absent)}")
    scenarios_report = _read_json(data_dir / SCENARIOS_REPORT_FILE)
    scenarios: dict[str, pd.DataFrame] = {}
    for name in sorted(scenarios_report.get("scenarios", {})):
        path = data_dir / SCENARIO_FILE_TEMPLATE.format(name=name, res=RES)
        if path.is_file():  # report names it, so its table should exist
            scenarios[name] = pd.read_parquet(path)
    return Tables(
        data_dir=data_dir,
        access=pd.read_parquet(data_dir / ACCESS_FILE_TEMPLATE.format(res=RES)),
        jobs=pd.read_parquet(data_dir / JOBS_FILE_TEMPLATE.format(res=RES)),
        equity={
            weighting: pd.read_parquet(data_dir / HEX_FILE_TEMPLATES[weighting].format(res=RES))
            for weighting in WEIGHTINGS
        },
        summary=pd.read_parquet(data_dir / SUMMARY_FILE),
        lorenz=pd.read_parquet(data_dir / LORENZ_FILE),
        districts=pd.read_parquet(data_dir / DISTRICTS_FILE),
        gains=pd.read_parquet(data_dir / GAINS_FILE),
        scenarios=scenarios,
        tt={
            mode: pd.read_parquet(data_dir / TT_FILE_TEMPLATE.format(mode=mode, res=RES))
            for mode in MODES
        },
        access_report=_read_json(data_dir / ACCESS_REPORT_FILE),
        equity_report=_read_json(data_dir / EQUITY_REPORT_FILE),
        scenarios_report=scenarios_report,
        model_report=_read_json(data_dir / MODEL_REPORT_FILE),
    )

"""Curate the dashboard's inputs into the committed ``data/app`` bundle.

Free hosting (Streamlit Community Cloud) can only serve what is in the
repository, and ``data/processed`` is git-ignored — so this stage copies
exactly the files the dashboard reads (r8 tables, the four r8 time
matrices and the report JSONs, ~8 MB, all derived from open data) into
``paths.app``, which IS committed (D-069). A deterministic manifest
documents the contents; no timestamps, so rebuilding an unchanged
pipeline leaves the committed bundle byte-identical.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from almaty_access.config import Config
from almaty_access.dashboard.data import REQUIRED_FILES, RES
from almaty_access.io import ensure_built
from almaty_access.scenarios.build import REPORT_FILE as SCENARIOS_REPORT_FILE
from almaty_access.scenarios.build import SCENARIO_FILE_TEMPLATE

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"

MANIFEST_NOTE = (
    "Dashboard data bundle, written by `almaty-access export --only bundle`. "
    "Everything here is derived from open data (OSM/Geofabrik, Kontur "
    "population, Overture buildings, Copernicus GLO-30, stat.gov.kz Taldau); "
    "sources and licences are documented in data/README.md. Transit measures "
    "are approximate lower bounds; scenarios are illustrative alignments, "
    "not official designs."
)


def bundle_file_names(processed: Path) -> list[str]:
    """Everything the bundle must contain, given the current pipeline run."""
    names = list(REQUIRED_FILES)
    report_path = processed / SCENARIOS_REPORT_FILE
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        names += [
            SCENARIO_FILE_TEMPLATE.format(name=name, res=RES)
            for name in sorted(report.get("scenarios", {}))
        ]
    return names


def build_bundle(config: Config, *, force: bool = False) -> None:
    """Copy the dashboard inputs into ``paths.app`` (make-style cached)."""
    processed = config.paths.processed
    app_dir = config.paths.app
    names = bundle_file_names(processed)
    inputs = [processed / name for name in names] + [config.source_path]
    outputs = [app_dir / name for name in names] + [app_dir / MANIFEST_FILE]

    # A renamed or deleted scenario must disappear from the bundle too.
    keep = set(names)
    orphans = [path for path in app_dir.glob("scenario_*.parquet") if path.name not in keep]
    for path in orphans:
        logger.info("removing orphan bundle file %s", path.name)
        path.unlink()

    def _build() -> None:
        app_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            if name.endswith(".json"):
                # Committed text gets LF endings regardless of the platform
                # the pipeline ran on, so re-exports never fight the
                # end-of-line pre-commit fixer.
                raw = (processed / name).read_bytes().replace(b"\r\n", b"\n")
                if not raw.endswith(b"\n"):
                    raw += b"\n"
                (app_dir / name).write_bytes(raw)
            else:
                shutil.copyfile(processed / name, app_dir / name)
        manifest = {
            "res": RES,
            "note": MANIFEST_NOTE,
            "files": {name: (app_dir / name).stat().st_size for name in sorted(names)},
        }
        with open(app_dir / MANIFEST_FILE, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        total_mb = sum((app_dir / name).stat().st_size for name in names) / 1e6
        logger.info("bundle: %d files, %.1f MB -> %s", len(names), total_mb, app_dir)

    ensure_built(outputs, inputs, _build, force=force or bool(orphans))

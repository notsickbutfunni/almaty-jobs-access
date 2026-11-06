"""Command-line interface: ``almaty-access``.

Each sub-command is one pipeline stage (README §6-§7): ``download`` →
``prepare`` → ``features`` → ``model`` → ``access`` → ``equity`` →
``scenarios`` → ``export``, each with make-style caching (``--force``
rebuilds; multi-step stages take ``--only``). Stages of later phases validate the
configuration, then exit with code 2 and point at their phase. Exit
codes: 0 success, 1 config or pipeline error, 2 stage not implemented
yet.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from almaty_access import __version__
from almaty_access.access import build as access_build
from almaty_access.config import Config, load_config, seed_everything
from almaty_access.dashboard import bundle as dash_bundle
from almaty_access.dashboard import webmap as dash_webmap
from almaty_access.download import buildings as dl_buildings
from almaty_access.download import dem as dl_dem
from almaty_access.download import osm as dl_osm
from almaty_access.download import population as dl_population
from almaty_access.download import stats as dl_stats
from almaty_access.equity import build as equity_build
from almaty_access.features import build as feat_build
from almaty_access.log import setup_logging
from almaty_access.model import build as model_build
from almaty_access.prepare import boundary as prep_boundary
from almaty_access.prepare import buildings as prep_buildings
from almaty_access.prepare import completeness as prep_completeness
from almaty_access.prepare import dem as prep_dem
from almaty_access.prepare import network as prep_network
from almaty_access.prepare import pois as prep_pois
from almaty_access.prepare import population as prep_population
from almaty_access.prepare import transit as prep_transit
from almaty_access.scenarios import build as scenarios_build

app = typer.Typer(
    name="almaty-access",
    help=(
        "Almaty Jobs & Accessibility Lab: open-data pipeline from raw data "
        "to job density, accessibility and equity results."
    ),
    no_args_is_help=True,
    add_completion=False,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("config") / "config.yaml"
SAMPLE_CONFIG = Path("config") / "config.sample.yaml"

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to config.yaml.")]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")]
ForceOpt = Annotated[bool, typer.Option("--force", help="Rebuild even when outputs look fresh.")]
OnlyOpt = Annotated[
    list[str] | None,
    typer.Option("--only", help="Run only the named step(s); repeatable."),
]

_PHASE_OF_STAGE = {"report": 8}

StepFunc = Callable[..., object]

DOWNLOAD_STEPS: dict[str, StepFunc] = {
    "boundary": dl_osm.download_boundary,
    "extract": dl_osm.download_osm_extract,
    "layers": dl_osm.extract_osm_layers,
    "networks": dl_osm.download_networks,
    "overture": dl_buildings.download_overture_buildings,
    "population": dl_population.download_population,
    "dem": dl_dem.download_dem,
    "stats": dl_stats.ensure_district_stats,
}

PREPARE_STEPS: dict[str, StepFunc] = {
    "boundary": prep_boundary.prepare_boundary,
    "dem": prep_dem.prepare_dem,
    "pois": prep_pois.prepare_pois,
    "buildings": prep_buildings.prepare_buildings,
    "population": prep_population.prepare_population,
    "network": prep_network.prepare_networks,
    "transit": prep_transit.prepare_transit,
    "completeness": prep_completeness.prepare_completeness,
}

ACCESS_STEPS: dict[str, StepFunc] = {
    "times": access_build.build_times,
    "measures": access_build.build_measures,
}

EXPORT_STEPS: dict[str, StepFunc] = {
    "bundle": dash_bundle.build_bundle,
    "webmap": dash_webmap.build_webmap,
}


def _load_and_seed(config_path: Path, verbose: bool) -> Config:
    """Set up logging, load + validate the config and seed the RNGs.

    Config problems exit with code 1 and a clean message (no traceback);
    exit code 2 is reserved for not-yet-implemented stages.
    """
    setup_logging(verbose)
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        if isinstance(exc, FileNotFoundError):
            typer.secho(
                "Hint: run from the repository root, or pass --config <path to config.yaml>.",
                fg=typer.colors.RED,
                err=True,
            )
        raise typer.Exit(code=1) from exc
    seed_everything(config.random_seed)
    logger.debug("configuration loaded from %s", config_path)
    return config


def _run_steps(
    steps: dict[str, StepFunc],
    config: Config,
    *,
    only: list[str] | None,
    force: bool,
    verbose: bool,
) -> None:
    if only:
        unknown = [name for name in only if name not in steps]
        if unknown:
            typer.secho(
                f"unknown step(s): {', '.join(unknown)}; available: {', '.join(steps)}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        selected = {name: func for name, func in steps.items() if name in set(only)}
    else:
        selected = steps
    for name, func in selected.items():
        typer.secho(f"[{name}]", fg=typer.colors.CYAN)
        try:
            func(config, force=force)
        except Exception as exc:
            if verbose:
                logger.exception("step '%s' failed", name)
            typer.secho(f"step '{name}' failed: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc


def _exit_not_implemented(stage: str) -> None:
    """Report that a stage belongs to a later phase and exit with code 2."""
    phase = _PHASE_OF_STAGE[stage]
    typer.secho(
        f"'{stage}' is not implemented yet: it arrives in Phase {phase} (README section 7). "
        "Run `almaty-access --help` for the stages available today.",
        fg=typer.colors.YELLOW,
    )
    raise typer.Exit(code=2)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"almaty-access {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Almaty Jobs & Accessibility Lab command-line interface."""


@app.command()
def info(config: ConfigOpt = DEFAULT_CONFIG, verbose: VerboseOpt = False) -> None:
    """Validate the configuration and print a summary."""
    cfg = _load_and_seed(config, verbose)
    lines = [
        f"almaty-access {__version__}",
        f"study area       : {cfg.study_area} (+{cfg.buffer_km:g} km buffer)",
        "area mode        : "
        + (f"bbox {cfg.area.bbox}" if cfg.area.bbox else f"place: {cfg.area.place}"),
        f"districts        : {len(cfg.districts)}",
        f"metric CRS       : {cfg.crs.metric}",
        f"storage CRS      : {cfg.crs.storage}",
        f"H3 resolutions   : analysis={cfg.h3.analysis_resolution}, fine={cfg.h3.fine_resolution}",
        f"time thresholds  : {', '.join(str(t) for t in cfg.travel.thresholds_min)} min",
        "speeds           : "
        + ", ".join(
            f"{mode}={speed:g} km/h" for mode, speed in sorted(cfg.travel.speeds_kmh.items())
        ),
        f"departure window : {cfg.travel.departure_window[0]}-{cfg.travel.departure_window[1]}",
        f"random seed      : {cfg.random_seed}",
        "paths:",
    ]
    for name in ("raw", "interim", "processed", "sample", "figures", "app", "webmap"):
        path: Path = getattr(cfg.paths, name)
        status = "ok" if path.exists() else "missing"
        lines.append(f"  {name:<10} {status:>7}  {path}")
    typer.echo("\n".join(lines))


@app.command()
def download(
    config: ConfigOpt = DEFAULT_CONFIG,
    only: OnlyOpt = None,
    force: ForceOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Download raw open datasets into data/raw/ (cached). [Phase 1]

    Steps in order: boundary, extract (PBF), layers (buildings/POIs/
    landuse), networks (walk/bike/drive), overture, population, dem,
    stats (validates the manually collected district CSV).
    """
    cfg = _load_and_seed(config, verbose)
    _run_steps(DOWNLOAD_STEPS, cfg, only=only, force=force, verbose=verbose)


@app.command()
def prepare(
    config: ConfigOpt = DEFAULT_CONFIG,
    only: OnlyOpt = None,
    force: ForceOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Clean, clip and derive layers into data/interim/ (cached). [Phase 1]

    Steps in order: boundary, dem, pois, buildings, population, network,
    transit (OSM route relations), completeness.
    """
    cfg = _load_and_seed(config, verbose)
    _run_steps(PREPARE_STEPS, cfg, only=only, force=force, verbose=verbose)


@app.command()
def sample(
    config: ConfigOpt = SAMPLE_CONFIG,
    force: ForceOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Rebuild the committed sample dataset from live sources (needs network).

    Uses config/config.sample.yaml (bbox in central Almaty, paths under
    data/sample/). Writes synthetic district stats, then runs the same
    download steps as the real pipeline. Tests and CI only READ the
    committed result; they never run this.
    """
    cfg = _load_and_seed(config, verbose)
    typer.secho("[stats:synthetic]", fg=typer.colors.CYAN)
    dl_stats.write_synthetic_stats(cfg)
    _run_steps(DOWNLOAD_STEPS, cfg, only=None, force=force, verbose=verbose)
    typer.echo("sample raw data rebuilt; run `almaty-access prepare -c config/config.sample.yaml`")


@app.command()
def features(
    config: ConfigOpt = DEFAULT_CONFIG,
    force: ForceOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Build hex-level feature tables at H3 resolutions 8 and 9. [Phase 2]

    Writes grid_r{8,9}.parquet, features_r{8,9}.parquet,
    subcenters.parquet and features_report.json into data/processed/.
    Requires `almaty-access prepare` to have run.
    """
    cfg = _load_and_seed(config, verbose)
    _run_steps(
        {"features": feat_build.build_features}, cfg, only=None, force=force, verbose=verbose
    )


@app.command()
def model(
    config: ConfigOpt = DEFAULT_CONFIG,
    force: ForceOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Estimate jobs per hex: baseline + random-forest dasymetric. [Phase 3]

    Writes jobs_r{8,9}.parquet and model_report.json into
    data/processed/. Requires `almaty-access features` to have run and
    district_stats.csv to exist (download --only stats).
    """
    cfg = _load_and_seed(config, verbose)
    _run_steps({"model": model_build.build_model}, cfg, only=None, force=force, verbose=verbose)


@app.command()
def access(
    config: ConfigOpt = DEFAULT_CONFIG,
    only: OnlyOpt = None,
    force: ForceOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Travel times and accessibility per mode (cached). [Phase 4]

    Steps in order: times (Dijkstra matrices tt_{mode}_r{res}.parquet +
    nearest-facility minutes), measures (access_r{8,9}.parquet +
    access_report.json). Requires `prepare` (incl. the transit step),
    `features` and `model` to have run. Transit times are APPROXIMATE:
    OSM routes + assumed speeds and headways (config `access.transit`).
    """
    cfg = _load_and_seed(config, verbose)
    _run_steps(ACCESS_STEPS, cfg, only=only, force=force, verbose=verbose)


@app.command()
def equity(
    config: ConfigOpt = DEFAULT_CONFIG,
    force: ForceOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Equity metrics and spatial statistics: Gini, Palma, LISA. [Phase 5]

    Writes equity_r{8,9}.parquet and equity_calibrated_r{8,9}.parquet
    (per-hex mode gap, LISA clusters, deserts, bivariate classes under
    the raster and the district-calibrated population weights), the
    summary / Lorenz / district / desert / nearest tables and
    equity_report.json (with the one-sentence result) into
    data/processed/. Requires `almaty-access access` to have run.
    Statistics cover inhabited city hexes only; transit statements
    inherit the approximate lower-bound caveat.
    """
    cfg = _load_and_seed(config, verbose)
    _run_steps({"equity": equity_build.build_equity}, cfg, only=None, force=force, verbose=verbose)


@app.command()
def scenarios(
    config: ConfigOpt = DEFAULT_CONFIG,
    force: ForceOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """What each drawn scenario changes, and for whom. [Phase 6]

    Reads every GeoJSON in scenarios.dir (new/extended transit lines,
    bike corridors), rebuilds the affected mode's network, recomputes
    job access at the analysis resolution and writes per-hex deltas
    (scenario_{name}_r8.parquet), scenarios_summary.parquet,
    scenarios_gains.parquet and scenarios_report.json (population
    gaining > X jobs and the Gini change per scenario, under both
    population weightings). Requires `almaty-access access` to have run.
    Transit deltas inherit the approximate lower-bound caveat.
    """
    cfg = _load_and_seed(config, verbose)
    _run_steps(
        {"scenarios": scenarios_build.build_scenarios}, cfg, only=None, force=force, verbose=verbose
    )


@app.command()
def export(
    config: ConfigOpt = DEFAULT_CONFIG,
    only: OnlyOpt = None,
    force: ForceOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Dashboard bundle and static web map (cached). [Phase 7]

    Steps in order: bundle (copies the dashboard's inputs — r8 tables,
    time matrices, report JSONs — into paths.app, the committed
    data/app/ folder free hosting serves), webmap (self-contained
    paths.webmap/index.html of the headline measure, for GitHub Pages).
    Requires `almaty-access scenarios` to have run. Serve the dashboard
    itself with: streamlit run app/streamlit_app.py
    """
    cfg = _load_and_seed(config, verbose)
    _run_steps(EXPORT_STEPS, cfg, only=only, force=force, verbose=verbose)


@app.command()
def report(config: ConfigOpt = DEFAULT_CONFIG, verbose: VerboseOpt = False) -> None:
    """Export final figures and report assets. [Phase 8]"""
    _load_and_seed(config, verbose)
    _exit_not_implemented("report")

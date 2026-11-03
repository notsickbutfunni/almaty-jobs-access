"""Command-line interface: ``almaty-access``.

Each sub-command is one pipeline stage (README §6-§7). Stages are
implemented phase by phase; a stage that is not built yet validates the
configuration, then exits with code 2 and points at its phase, instead
of pretending to work or failing silently.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from almaty_access import __version__
from almaty_access.config import Config, load_config, seed_everything
from almaty_access.log import setup_logging
from almaty_access.sample import build_sample

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

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Path to config.yaml.")]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")]

_PHASE_OF_STAGE = {
    "download": 1,
    "prepare": 1,
    "features": 2,
    "model": 3,
    "access": 4,
    "equity": 5,
    "report": 8,
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


def _exit_not_implemented(stage: str) -> None:
    """Report that a stage belongs to a later phase and exit with code 2."""
    phase = _PHASE_OF_STAGE[stage]
    typer.secho(
        f"'{stage}' is not implemented yet: it arrives in Phase {phase} (README section 7). "
        "Available today: info, sample.",
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
    for name in ("raw", "interim", "processed", "sample", "figures"):
        path: Path = getattr(cfg.paths, name)
        status = "ok" if path.exists() else "missing"
        lines.append(f"  {name:<10} {status:>7}  {path}")
    typer.echo("\n".join(lines))


@app.command()
def sample(
    config: ConfigOpt = DEFAULT_CONFIG,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Output directory (default: paths.sample from config)."),
    ] = None,
    verbose: VerboseOpt = False,
) -> None:
    """(Re)build the synthetic sample dataset used by tests and CI."""
    cfg = _load_and_seed(config, verbose)
    written = build_sample(cfg, out_dir=out)
    for name, path in written.items():
        typer.echo(f"wrote {name}: {path}")


@app.command()
def download(config: ConfigOpt = DEFAULT_CONFIG, verbose: VerboseOpt = False) -> None:
    """Download raw open datasets: OSM, buildings, population, DEM. [Phase 1]"""
    _load_and_seed(config, verbose)
    _exit_not_implemented("download")


@app.command()
def prepare(config: ConfigOpt = DEFAULT_CONFIG, verbose: VerboseOpt = False) -> None:
    """Clean, clip and reproject raw layers into data/interim/. [Phase 1]"""
    _load_and_seed(config, verbose)
    _exit_not_implemented("prepare")


@app.command()
def features(config: ConfigOpt = DEFAULT_CONFIG, verbose: VerboseOpt = False) -> None:
    """Build hex-level feature tables at H3 resolutions 8 and 9. [Phase 2]"""
    _load_and_seed(config, verbose)
    _exit_not_implemented("features")


@app.command()
def model(config: ConfigOpt = DEFAULT_CONFIG, verbose: VerboseOpt = False) -> None:
    """Estimate jobs per hex: baseline + random-forest dasymetric. [Phase 3]"""
    _load_and_seed(config, verbose)
    _exit_not_implemented("model")


@app.command()
def access(config: ConfigOpt = DEFAULT_CONFIG, verbose: VerboseOpt = False) -> None:
    """Compute travel times and accessibility measures per mode. [Phase 4]"""
    _load_and_seed(config, verbose)
    _exit_not_implemented("access")


@app.command()
def equity(config: ConfigOpt = DEFAULT_CONFIG, verbose: VerboseOpt = False) -> None:
    """Equity metrics and spatial statistics: Gini, Palma, LISA. [Phase 5]"""
    _load_and_seed(config, verbose)
    _exit_not_implemented("equity")


@app.command()
def report(config: ConfigOpt = DEFAULT_CONFIG, verbose: VerboseOpt = False) -> None:
    """Export final figures and report assets. [Phase 8]"""
    _load_and_seed(config, verbose)
    _exit_not_implemented("report")

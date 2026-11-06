"""Tests for the typer command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from almaty_access import __version__
from almaty_access.cli import app

runner = CliRunner()

PIPELINE_COMMANDS = (
    "download",
    "prepare",
    "features",
    "model",
    "access",
    "equity",
    "scenarios",
    "export",
    "report",
)
UTILITY_COMMANDS = ("info", "sample")

STAGE_PHASES = {
    "report": 8,
}


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # Look only at the command list: the app description above it already
    # contains words like "accessibility" and "equity", which would make a
    # whole-output substring check pass even with the commands removed.
    commands_section = result.output.split("Commands")[-1]
    for command in PIPELINE_COMMANDS + UTILITY_COMMANDS:
        assert command in commands_section


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_info_prints_config_summary(config_path: Path) -> None:
    result = runner.invoke(app, ["info", "--config", str(config_path)])
    assert result.exit_code == 0
    assert "EPSG:32643" in result.output
    assert "analysis=8" in result.output
    assert "random seed" in result.output


@pytest.mark.parametrize("stage", sorted(STAGE_PHASES))
def test_unimplemented_stage_exits_with_phase_pointer(config_path: Path, stage: str) -> None:
    result = runner.invoke(app, [stage, "--config", str(config_path)])
    assert result.exit_code == 2
    assert f"Phase {STAGE_PHASES[stage]}" in result.output


def test_download_rejects_unknown_step(config_path: Path) -> None:
    result = runner.invoke(app, ["download", "--config", str(config_path), "--only", "bogus"])
    assert result.exit_code == 1
    assert "unknown step" in result.output
    assert "boundary" in result.output  # lists the available steps


def test_bad_config_exits_cleanly_with_code_1(tmp_path: Path) -> None:
    bad = tmp_path / "config.yaml"
    bad.write_text("project: {}\n", encoding="utf-8")
    result = runner.invoke(app, ["download", "--config", str(bad)])
    assert result.exit_code == 1
    assert "config error" in result.output


def test_missing_config_exits_cleanly_with_hint(tmp_path: Path) -> None:
    result = runner.invoke(app, ["info", "--config", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 1
    assert "not found" in result.output
    assert "Hint" in result.output


def test_help_mentions_sample_rebuild() -> None:
    result = runner.invoke(app, ["sample", "--help"])
    assert result.exit_code == 0
    assert "config.sample.yaml" in result.output.replace("\n", "")


def test_features_with_missing_inputs_exits_cleanly(tmp_path: Path) -> None:
    """`features` is implemented since Phase 2: without the interim layers
    it must fail with a clean exit-1 message, not a traceback or code 2."""
    import yaml

    repo_root = Path(__file__).resolve().parents[1]  # keep the test cwd-independent
    source = repo_root / "config" / "config.sample.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    raw["pois"]["categories_file"] = str(repo_root / "config" / "poi_categories.yaml")
    raw["scenarios"]["dir"] = str(repo_root / "config" / "scenarios_sample")
    raw["paths"] = {
        "raw": str(tmp_path / "raw"),
        "interim": str(tmp_path / "interim"),
        "processed": str(tmp_path / "processed"),
        "sample": str(tmp_path / "sample"),
        "figures": str(tmp_path / "figures"),
        "app": str(tmp_path / "app"),
        "webmap": str(tmp_path / "webmap"),
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    result = runner.invoke(app, ["features", "--config", str(config_file)])
    assert result.exit_code == 1
    assert "failed" in result.output

    # Same contract for the Phase 3 model command.
    result = runner.invoke(app, ["model", "--config", str(config_file)])
    assert result.exit_code == 1
    assert "failed" in result.output

    # And for the Phase 4 access command.
    result = runner.invoke(app, ["access", "--config", str(config_file)])
    assert result.exit_code == 1
    assert "failed" in result.output

    # And for the Phase 5 equity command.
    result = runner.invoke(app, ["equity", "--config", str(config_file)])
    assert result.exit_code == 1
    assert "failed" in result.output

    # And for the Phase 6 scenarios command.
    result = runner.invoke(app, ["scenarios", "--config", str(config_file)])
    assert result.exit_code == 1
    assert "failed" in result.output

    # And for the Phase 7 export command.
    result = runner.invoke(app, ["export", "--config", str(config_file)])
    assert result.exit_code == 1
    assert "failed" in result.output


def test_access_help_names_steps_and_approximation() -> None:
    result = runner.invoke(app, ["access", "--help"])
    assert result.exit_code == 0
    flat = result.output.replace("\n", "")
    assert "times" in flat and "measures" in flat
    assert "APPROXIMATE" in flat


def test_equity_help_names_outputs_and_caveat() -> None:
    result = runner.invoke(app, ["equity", "--help"])
    assert result.exit_code == 0
    flat = " ".join(result.output.split())
    assert "equity_report.json" in flat
    assert "lower-bound" in flat


def test_scenarios_help_names_outputs_and_caveat() -> None:
    result = runner.invoke(app, ["scenarios", "--help"])
    assert result.exit_code == 0
    flat = " ".join(result.output.split())
    assert "scenarios_report.json" in flat
    assert "lower-bound" in flat


def test_export_help_names_steps_and_dashboard() -> None:
    result = runner.invoke(app, ["export", "--help"])
    assert result.exit_code == 0
    flat = " ".join(result.output.split())
    assert "bundle" in flat and "webmap" in flat
    assert "streamlit run app/streamlit_app.py" in flat

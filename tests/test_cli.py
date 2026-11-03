"""Tests for the typer command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from almaty_access import __version__
from almaty_access.cli import app

runner = CliRunner()

PIPELINE_COMMANDS = ("download", "prepare", "features", "model", "access", "equity", "report")
UTILITY_COMMANDS = ("info", "sample")

STAGE_PHASES = {
    "features": 2,
    "model": 3,
    "access": 4,
    "equity": 5,
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

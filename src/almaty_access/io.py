"""Read/write GeoParquet and make-style freshness caching.

Every pipeline step writes its result as GeoParquet and skips work when
its output is already newer than all of its inputs. This is exactly the
idea behind ``make``: a target is rebuilt if and only if it is missing
or any prerequisite has a newer modification time.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import geopandas as gpd

logger = logging.getLogger(__name__)


def write_geoparquet(gdf: gpd.GeoDataFrame, path: str | Path) -> Path:
    """Write a GeoDataFrame to GeoParquet, creating parent directories.

    Refuses to write a frame without a CRS: a spatial file whose CRS is
    unknown breaks every later join and reprojection silently.
    """
    path = Path(path)
    if gdf.crs is None:
        raise ValueError(f"refusing to write {path.name}: GeoDataFrame has no CRS")
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(path)
    logger.debug("wrote %s (%d rows)", path, len(gdf))
    return path


def read_geoparquet(path: str | Path) -> gpd.GeoDataFrame:
    """Read a GeoParquet file, failing with a clear message if it is absent."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"expected GeoParquet file does not exist: {path}")
    return gpd.read_parquet(path)


def needs_rebuild(output: str | Path, inputs: Sequence[str | Path]) -> bool:
    """Return True when ``output`` is missing or older than any input.

    Comparison uses nanosecond modification times (``st_mtime_ns``). On
    filesystems that store sub-second mtimes (NTFS, ext4, APFS) this
    orders even same-second writes correctly; on coarse filesystems
    (FAT/exFAT, some network mounts) equal timestamps count as fresh —
    the same semantics ``make`` has.

    Raises
    ------
    FileNotFoundError
        If any input is missing: a pipeline step must never run
        silently on absent inputs.
    """
    output = Path(output)
    input_paths = [Path(p) for p in inputs]
    missing = [str(p) for p in input_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing input(s) for {output.name}: {', '.join(missing)}")
    if not output.exists():
        return True
    output_mtime = output.stat().st_mtime_ns
    return any(p.stat().st_mtime_ns > output_mtime for p in input_paths)


def ensure_built(
    outputs: str | Path | Sequence[str | Path],
    inputs: Sequence[str | Path],
    build: Callable[[], None],
    *,
    force: bool = False,
) -> bool:
    """Run ``build()`` unless every output is already up to date.

    This is the caching convention used by every pipeline step: wrap the
    expensive work in a zero-argument ``build`` callable and let this
    function decide whether it needs to run. A step may produce several
    files; it is rebuilt when ANY of them is missing or stale.

    Parameters
    ----------
    outputs:
        The file (or files) the step is expected to produce.
    inputs:
        Files the outputs depend on; all must exist.
    build:
        Zero-argument callable that creates all outputs.
    force:
        If True, rebuild even when the outputs look fresh.

    Returns
    -------
    bool
        True when ``build`` ran, False when the cached outputs were kept.

    Raises
    ------
    RuntimeError
        If ``build`` completes without producing every output.
    """
    output_paths = (
        [Path(outputs)] if isinstance(outputs, (str, Path)) else [Path(p) for p in outputs]
    )
    if not output_paths:
        raise ValueError("ensure_built needs at least one output path")
    # Always evaluated, even with force=True: inputs must exist either way.
    stale = any(needs_rebuild(output, inputs) for output in output_paths)
    if not force and not stale:
        logger.info("up to date, skipping: %s", ", ".join(p.name for p in output_paths))
        return False
    logger.info("building: %s", ", ".join(p.name for p in output_paths))
    build()
    missing = [str(p) for p in output_paths if not p.exists()]
    if missing:
        raise RuntimeError(f"build step finished but did not produce: {', '.join(missing)}")
    return True

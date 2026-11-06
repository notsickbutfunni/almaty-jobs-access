"""The project chart system as map colors (same hexes as notebooks 05/06).

Color is assigned by job (dataviz method): magnitude rides one blue
sequential ramp, categories get the fixed validated hues, text stays in
ink tokens. Every categorical set used on the map passed the all-pairs
check (normal-vision OKLab distance x100 >= 15; protan / deutan / tritan
simulated distance >= 8). That check failed the notebook-05 LISA tints
for the HL / LH outlier pair (13.7 normal vision), so the map folds both
outlier classes into one yellow "spatial outlier" class (D-070); the
city data contains no HL / LH hexes, so notebook figures are unaffected.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID_LINE = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"

BLUE = "#2a78d6"
LIGHT_BLUE = "#86b6ef"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
RED = "#e34948"

#: One-hue magnitude ramp, light -> dark (SEQ_BLUE in the notebooks).
SEQ_BLUE: tuple[str, ...] = (
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
)

#: Fixed hues per weighting, identical to notebook 06 (raster = tint).
WEIGHTING_COLORS: dict[str, str] = {"raster": LIGHT_BLUE, "calibrated": BLUE}

#: LISA cluster classes. HL and LH share one hue on purpose (see module
#: docstring); 'ns' and 'island' are both non-verdicts and stay neutral.
LISA_COLORS: dict[str, str] = {
    "HH": RED,
    "LL": BLUE,
    "HL": YELLOW,
    "LH": YELLOW,
    "ns": NEUTRAL,
    "island": NEUTRAL,
}
#: HL/LH share one label (and hue); ns/island share the neutral one —
#: otherwise the legend shows two rows with identical swatches.
LISA_LABELS: dict[str, str] = {
    "HH": "Hot spot (high among high)",
    "LL": "Cold spot (low among low)",
    "HL": "Spatial outlier (HL / LH)",
    "LH": "Spatial outlier (HL / LH)",
    "ns": "Not significant",
    "island": "Not significant",
}

#: Fill opacity for painted hexes (0-255); keeps the basemap readable.
FILL_ALPHA = 215

#: Robust color-scale bounds: percentiles of the painted values.
CLIP_PERCENTILES = (2.0, 98.0)


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    """``"#2a78d6"`` -> ``(42, 120, 214)``."""
    value = color.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def blend(color: str, alpha: float, *, base: str = SURFACE) -> str:
    """The color at the given opacity over ``base``, as an opaque hex."""
    top = np.array(hex_to_rgb(color), dtype=float)
    under = np.array(hex_to_rgb(base), dtype=float)
    mixed = np.round(alpha * top + (1.0 - alpha) * under).astype(int)
    return "#" + "".join(f"{channel:02x}" for channel in mixed)


def sequential_rgba(
    values: pd.Series | np.ndarray,
    *,
    reverse: bool = False,
    alpha: int = FILL_ALPHA,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Map numbers onto the blue ramp; returns RGBA rows + scale bounds.

    The scale runs between the 2nd and 98th percentile of the finite
    values (a robust choice for skewed access distributions; the bounds
    are returned so the legend can state them). ``reverse`` flips the
    mapping so DARK marks LOW values — used by the mode-gap layer, where
    attention belongs on the hexes where transit lags the most. NaN rows
    come back fully transparent.
    """
    data = np.asarray(values, dtype=float)
    finite = np.isfinite(data)
    out = np.zeros((data.size, 4), dtype=np.uint8)
    if not finite.any():
        return out, (float("nan"), float("nan"))
    low, high = np.nanpercentile(data[finite], CLIP_PERCENTILES)
    if high <= low:  # constant layer: paint everything mid-ramp
        t = np.full(data.size, 0.5)
    else:
        t = (np.clip(data, low, high) - low) / (high - low)
    if reverse:
        t = 1.0 - t
    stops = np.linspace(0.0, 1.0, len(SEQ_BLUE))
    ramp = np.array([hex_to_rgb(color) for color in SEQ_BLUE], dtype=float)
    for channel in range(3):
        out[:, channel] = np.round(
            np.interp(np.where(finite, t, 0.0), stops, ramp[:, channel])
        ).astype(np.uint8)
    out[:, 3] = np.where(finite, alpha, 0)
    return out, (float(low), float(high))


def categorical_rgba(
    values: pd.Series,
    colors: Mapping[str, str],
    *,
    alpha: int = FILL_ALPHA,
) -> np.ndarray:
    """Map class labels onto fixed hues; unknown / missing -> transparent."""
    out = np.zeros((len(values), 4), dtype=np.uint8)
    for i, value in enumerate(values):
        color = colors.get(value) if isinstance(value, str) else None
        if color is not None:
            out[i, :3] = hex_to_rgb(color)
            out[i, 3] = alpha
    return out

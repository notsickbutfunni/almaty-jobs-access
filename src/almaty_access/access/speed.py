"""Mode speeds as a function of edge grade / road class (Phase 4).

Walking uses Tobler's hiking function; cycling a transparent linear
up/down adjustment; driving free-flow speeds by highway class. All slope
handling works on the DIRECTED edge grade prepared in Phase 1 — climbing
a street and descending it get different speeds, which is why the graphs
keep both directions of every two-way street.
"""

from __future__ import annotations

import numpy as np

#: Tobler's hiking function: W(s) = 6 * exp(-3.5 * |s + 0.05|) km/h,
#: where s is the slope (dh/dx). Its value on flat ground:
TOBLER_FLAT_KMH = 6.0 * float(np.exp(-3.5 * 0.05))  # ~5.04 km/h

#: Cycling slope response (assumptions, documented in DECISIONS):
#: uphill costs BIKE_UPHILL_KMH_PER_GRADE * grade km/h, downhill gains
#: half that rate, capped at BIKE_MAX_FACTOR * flat speed; the floor is
#: BIKE_MIN_FACTOR * flat speed (slow riding / pushing pace uphill).
BIKE_UPHILL_KMH_PER_GRADE = 100.0  # -1 km/h per 1% of climb
BIKE_DOWNHILL_KMH_PER_GRADE = 50.0  # +0.5 km/h per 1% of descent
BIKE_MAX_FACTOR = 1.4
BIKE_MIN_FACTOR = 0.25


def tobler_walk_kmh(grade: np.ndarray, flat_kmh: float) -> np.ndarray:
    """Walking speed per edge, scaled so grade 0 gives ``flat_kmh``.

    Tobler's function peaks on a gentle -5% descent and decays
    exponentially both uphill and downhill of it. The classic formula
    yields ~5.04 km/h on flat ground; scaling preserves its SHAPE while
    honouring the configured flat speed.
    """
    grade = np.asarray(grade, dtype="float64")
    tobler = 6.0 * np.exp(-3.5 * np.abs(grade + 0.05))
    return tobler * (flat_kmh / TOBLER_FLAT_KMH)


def bike_kmh(grade: np.ndarray, flat_kmh: float) -> np.ndarray:
    """Cycling speed per edge: linear uphill penalty, capped downhill bonus."""
    grade = np.asarray(grade, dtype="float64")
    uphill = np.clip(grade, 0.0, None)
    downhill = np.clip(-grade, 0.0, None)
    speed = flat_kmh - BIKE_UPHILL_KMH_PER_GRADE * uphill + BIKE_DOWNHILL_KMH_PER_GRADE * downhill
    return np.clip(speed, BIKE_MIN_FACTOR * flat_kmh, BIKE_MAX_FACTOR * flat_kmh)


def car_kmh(highway: np.ndarray, speeds: dict[str, float]) -> np.ndarray:
    """Free-flow car speed per edge from its OSM highway class.

    ``*_link`` ramps use their base class; anything unknown uses the
    required ``default`` entry.
    """
    classes = np.asarray(highway, dtype=object)
    default = speeds["default"]

    def lookup(value: object) -> float:
        name = str(value)
        if name.endswith("_link"):
            name = name[: -len("_link")]
        return speeds.get(name, default)

    return np.array([lookup(value) for value in classes], dtype="float64")


def seconds(length_m: np.ndarray, kmh: np.ndarray | float) -> np.ndarray:
    """Edge traversal time in seconds at the given speed(s)."""
    return np.asarray(length_m, dtype="float64") / (np.asarray(kmh, dtype="float64") / 3.6)

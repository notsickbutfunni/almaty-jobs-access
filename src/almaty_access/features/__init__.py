"""Phase 2: H3 grid and feature engineering (README §7 Phase 2).

One module per feature group, each a pure function that returns a
DataFrame indexed by H3 cell id, plus:

- :mod:`almaty_access.features.grid` — the study grid itself,
- :mod:`almaty_access.features.context` — k-ring neighbourhood features,
- :mod:`almaty_access.features.spatial` — Moran's I (used by the EDA),
- :mod:`almaty_access.features.build` — the orchestrator behind
  ``almaty-access features``.
"""

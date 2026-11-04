"""Phase 3: the job-density model (README §7 Phase 3).

There is no open hex-level ground truth for jobs — official counts
exist only as district totals. Both models therefore build a *weight
surface* over hexes and rescale it so each district's hex sum equals
the official total (mass-preserving dasymetric disaggregation):

- :mod:`almaty_access.model.baseline` — Model A, the GFA-coefficient
  heuristic every other model must beat,
- :mod:`almaty_access.model.forest` — Model B, the WorldPop-style
  random forest trained at district level,
- :mod:`almaty_access.model.target` / :mod:`almaty_access.model.aggregate`
  — official targets and hex→district feature aggregation,
- :mod:`almaty_access.model.build` — the orchestrator behind
  ``almaty-access model``.
"""

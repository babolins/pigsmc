# 006 — Bisection Translation Moves

## Parent PRD

`issues/prd.md`

## What to build

Add `TranslationBisectionMove`: a multi-slice Lévy bridge move that proposes new positions for a single particle over a contiguous range of 2^L slices. Bisection moves are the standard high-efficiency move for path integral simulations and are essential for sampling long paths without very low acceptance rates.

The move is slice-type-agnostic: it operates on contiguous slices using the elementary free-propagator step size (τ' for the primitive propagator; t₁τ' for Chin, once `issues/008` is merged). The engine routes `V` vs `Ṽ` evaluation via `slice_kind` as usual.

**Scope:**

- `TranslationBisectionMove(level: int)` C++ `Move` subclass
- Validates at construction: `level ≥ 1`
- Validates at first use (or at `sim.run()` call): `2^level ≤ M - 1`
- Per attempt: select a random particle `i`; select a random starting slice `m_start` such that the range `[m_start, m_start + 2^level]` fits within `[0, M-1]`; use the Lévy bridge to propose new positions for the `2^level - 1` interior slices of the range, conditioning on the fixed endpoint slices `m_start` and `m_start + 2^level`
- Lévy bridge proposal: for translational DOF, the midpoint of any sub-interval is Gaussian with mean equal to the linear interpolation of the two sub-interval endpoints and variance `t_eff × lambda_trans` where `t_eff` is the number of elementary steps to each endpoint
- `MoveResult.changed` contains the `(i, m)` pairs for all interior slices of the range (endpoints are fixed)
- `MoveResult.log_ratio_contrib` contains the log of the Lévy bridge proposal ratio (the kinematic contribution the move owns — cancels the free-propagator kinetic terms in the acceptance ratio)
- When `issues/004` is merged: Lévy bridge mean respects PBC (interpolation takes the minimum-image path between endpoints)

**QA target (manual):** free particle with M=9, level=3 — verify that the sampled path distribution matches the known free-propagator Gaussian distribution analytically.

## Acceptance criteria

- [ ] `TranslationBisectionMove(level=1)` is importable and registerable
- [ ] For a zero-potential free-particle system, acceptance rate is 1.0 regardless of `step_size` (bisection proposal exactly satisfies detailed balance for the kinetic term alone)
- [ ] `MoveResult.changed` contains exactly `2^level - 1` entries (interior slices only)
- [ ] Attempting to use a `level` where `2^level > M - 1` raises a clear error at `sim.run()` time or earlier
- [ ] `acceptance_stats` tracks `TranslationBisectionMove` attempts and acceptances separately
- [ ] All prior tests pass

## Blocked by

- `issues/003-cpp-sweep-engine.md`

## User stories addressed

- User story 15 (`TranslationBisectionMove`)
- User story 17 (slice-type-agnostic bisection, elementary step size)

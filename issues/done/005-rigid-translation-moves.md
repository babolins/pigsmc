# 005 — Rigid Translation Moves

## Parent PRD

`issues/prd.md`

## What to build

Add `TranslationRigidMove`: a move that displaces all M slices of a single randomly selected particle by the same randomly drawn vector. This restores ergodicity for the path's center-of-mass degree of freedom, which single-slice moves cannot efficiently sample.

**Scope:**

- `TranslationRigidMove(step_size: float)` C++ `Move` subclass
- Per attempt: select a random particle `i`; draw a uniform displacement vector from `[-step_size, step_size]³`; shift `positions[m, i, :]` by that vector for all M slices simultaneously
- `MoveResult.particle = i`, `MoveResult.m_lo = 0`, `MoveResult.m_hi = M - 1` (all slices for the selected particle form one consecutive range)
- `MoveResult.log_ratio_contrib = 0.0` (proposal is symmetric)
- When `issues/004` is merged: proposed positions are wrapped into the box for periodic dimensions after acceptance
- Acceptance ratio computed by the engine as usual: sum of potential changes across all affected slices plus trial wavefunction change at endpoints

**QA target (manual):** single particle in a box with a flat potential — verify that the center-of-mass position samples the full box uniformly when using only `TranslationRigidMove`, which would not be achievable with only end/interior moves at large M.

## Acceptance criteria

- [ ] `TranslationRigidMove` is importable and registerable via `sim.add_move()`
- [ ] After a run with only `TranslationRigidMove` and a zero potential, `positions[:, i, :]` for any particle `i` shows the same displacement applied to all slices (rigid shift verified in a test)
- [ ] `acceptance_stats` tracks attempts and acceptances for `TranslationRigidMove` separately from other move types
- [ ] Acceptance rate is 1.0 for a zero-potential system with no trial wavefunction set (kinetic links are unchanged because all slices shift by the same vector, but endpoint wavefunction terms would be non-zero with a non-trivial trial wavefunction)
- [ ] All prior tests pass

## Blocked by

- `issues/003-cpp-sweep-engine.md`

## User stories addressed

- User story 15 (`TranslationRigidMove`)

# 004 — Periodic Boundary Conditions

## Parent PRD

`issues/prd.md`

## What to build

Add support for orthorhombic periodic boundary conditions. This slice introduces the full `Boundary` enum, `box_lengths` parameter, and minimum-image convention (MIC) computation in C++. All existing move types (`TranslationEndMove`, `TranslationInteriorMove`, and those added in `issues/005` and `issues/006`) become PBC-aware when this slice is merged: the engine pre-computes MIC-corrected `r_ij` before passing it to user potential and trial wavefunction functions, and wraps particle positions back into the box after accepted moves.

**Scope:**

- Full `Boundary` enum: `FREE`, `PERIODIC_3D`, `QUASI_2D` (periodic x,y; free z), `QUASI_1D` (free x,y; periodic z)
- `box_lengths: tuple[float, float, float]` parameter on `Simulation` (required when boundary ≠ `FREE`; ignored dimensions may be set to any positive value)
- Validation: `box_lengths` must be provided and positive for non-free boundary types
- C++ MIC function selected at `Simulation` construction via function pointer, dispatched with zero runtime branching in the hot path; `Boundary.FREE` dispatches to a no-op
- Engine pre-computes `r_ij = r_i - r_j` with MIC applied before every call to `V_int` and `h` (trial wavefunction pair term)
- Move types wrap proposed positions back into the simulation box (for periodic dimensions) after a proposal is accepted; orientation vectors are never wrapped
- `PathState` exposes `boundary`, `box_lengths`, and a `periodic` mask (length-3 bool array) so custom moves can implement PBC-aware proposals if needed

**QA target (manual):** LJ fluid in a periodic box — verify that pair distances are always within `[0, L/2]` for periodic dimensions, and that the simulation energy is stable (no particle escapes the box).

## Acceptance criteria

- [ ] `Boundary.PERIODIC_3D` simulation completes without error
- [ ] All `r_ij` vectors passed to `V_int` and `h` satisfy `|r_ij[d]| ≤ box_lengths[d]/2` for each periodic dimension `d` — verified in a test with a mock potential that records all `r_ij` values
- [ ] Accepted moves wrap positions back into the box: `0 ≤ positions[m, i, d] < box_lengths[d]` for periodic dimensions after every `run()` call
- [ ] `Boundary.FREE` behavior is identical to before this slice (no-op MIC, no wrapping)
- [ ] `QUASI_2D` wraps x and y but not z; `QUASI_1D` wraps z but not x or y
- [ ] Constructing a non-FREE `Simulation` without `box_lengths` raises a clear `ValueError`
- [ ] All tests from `issues/001` and `issues/003` pass

## Blocked by

- `issues/003-cpp-sweep-engine.md`

## User stories addressed

- User story 5
- User story 7 (MIC-corrected `r_ij` delivered to `V_int` and `h`)

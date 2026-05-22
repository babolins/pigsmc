# 004 — Periodic Boundary Conditions

## Parent PRD

`issues/prd.md`

## What to build

Add support for orthorhombic periodic boundary conditions. This slice introduces the full `Boundary` enum, `box_lengths` parameter, and minimum-image convention (MIC) computation in C++. All existing move types (`TranslationEndMove`, `TranslationInteriorMove`, and those added in `issues/005` and `issues/006`) become PBC-aware when this slice is merged: the engine pre-computes MIC-corrected `r_ij` before passing it to user potential and trial wavefunction functions, and wraps particle positions back into the box after accepted moves.

**Scope:**

- `Boundary` class hierarchy with four concrete subclasses via factory methods:
  - `Boundary.free()` — no box lengths; MIC is a no-op; `displacement()` returns raw `r_i - r_j`
  - `Boundary.periodic_3d(Lx, Ly, Lz)` — fully periodic orthorhombic box
  - `Boundary.quasi_2d(Lx, Ly)` — periodic in x and y, free in z
  - `Boundary.quasi_1d(Lz)` — free in x and y, periodic in z

  Each subclass accepts only the box lengths for its periodic dimensions; passing extra lengths raises `TypeError` at construction
- `box_lengths` property on each `Boundary` object exposes only the periodic-dimension lengths (absent on `FreeBoundary`)
- `Simulation` accepts a `boundary: Boundary` argument (no separate `box_lengths` parameter); `Boundary` is required
- Validation: all provided box lengths must be positive; raised as `ValueError` at construction
- C++ MIC function selected at `Simulation` construction via function pointer, dispatched with zero runtime branching in the hot path; `Boundary.FREE` dispatches to a no-op
- Engine pre-computes `r_ij = r_i - r_j` with MIC applied before every call to `V_int` and `h` (trial wavefunction pair term)
- Positions are stored in **unwrapped coordinates**: particle positions are not constrained to the primary simulation cell and are never wrapped after accepted moves; only pair displacements `r_ij` are MIC-corrected before being passed to user functions
- `PathState` exposes the `Boundary` object (`path_state.boundary`) so custom moves can call `path_state.boundary.displacement(r_i, r_j)` or inspect `path_state.boundary.box_lengths` if needed
- Each `Boundary` object exposes:
  - `displacement(r_i, r_j)` — NumPy, broadcastable; applies MIC to periodic dimensions, returns raw difference for free dimensions; shapes broadcast to `(..., 3)` output
  - `all_displacements(positions_slice)` — convenience wrapper; given an `(N, 3)` slice, returns the `(N, N, 3)` all-pairs displacement matrix
- `sim.boundary` — read-only property returning the `Boundary` object passed at construction

**QA target (manual):** LJ fluid in a `Boundary.periodic_3d(L, L, L)` box — verify that all `r_ij` pair displacements satisfy `|r_ij[d]| ≤ L/2`, that particle positions drift freely beyond `[0, L)` over the run (confirming unwrapped storage), and that the simulation energy is stable.

## Acceptance criteria

- [ ] `Boundary.periodic_3d(L, L, L)` simulation completes without error
- [ ] `Boundary.free()` accepts no arguments; `Boundary.quasi_2d(Lx, Ly)` raises `TypeError` if a third argument is passed
- [ ] All `r_ij` vectors passed to `V_int` and `h` satisfy `|r_ij[d]| ≤ box_lengths[d]/2` for each periodic dimension `d` — verified in a test with a mock potential that records all `r_ij` values
- [ ] `bc.displacement(r_i, r_j)` and `bc.all_displacements(positions_slice)` return the same MIC-corrected values as the engine delivers to `V_int`
- [ ] Positions are stored in unwrapped coordinates: after accepted moves, `positions[m, i, d]` is NOT constrained to `[0, box_lengths[d])` and may exceed that range for periodic dimensions
- [ ] `sim.boundary` returns the `Boundary` object passed at construction
- [ ] `Boundary.free()` behavior is identical to before this slice (no-op MIC)
- [ ] `Boundary.quasi_2d(Lx, Ly)` applies MIC to x and y but not z; `Boundary.quasi_1d(Lz)` applies MIC to z but not x or y
- [ ] Constructing a `Boundary.periodic_3d` with a non-positive box length raises a clear `ValueError`
- [ ] All tests from `issues/001` and `issues/003` pass

## Blocked by

- `issues/003-cpp-sweep-engine.md`

## User stories addressed

- User story 5 (`Boundary` class hierarchy and factory methods)
- User story 7 (MIC-corrected `r_ij` delivered to `V_int` and `h`)
- User story 45 (`sim.boundary` read-only property)
- User story 46 (`displacement()` and `all_displacements()` on `Boundary` objects)

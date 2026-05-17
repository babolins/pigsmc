# 001 — Minimal Working Simulation

## Parent PRD

`issues/prd.md`

## What to build

A pure-Python end-to-end working PIGSMC simulation covering the full user-facing API for the common case: primitive propagator, free boundary conditions, translational degrees of freedom only, Python-implemented physics. The sweep loop runs in Python (temporary — replaced by the C++ engine in `issues/003`). No C++ extension, no build system, no checkpointing.

This slice establishes the complete public API surface that all subsequent slices build on without breaking.

**Scope:**

- `ParticleType` dataclass with `lambda_trans: float` and `lambda_rot: None` (rotational DOF not yet active)
- `Boundary.FREE` enum value (full enum deferred to `issues/004`)
- `Simulation` class accepting: list of `ParticleType`, `M`, `tau_prime`, `propagator="primitive"`, `boundary=Boundary.FREE`, `rng` type string, `seed`, `sweeps_per_block`
- Validation at construction: M-1 divisible by 2, M ≥ 3, all particles have `lambda_trans`
- `positions` and `orientations` path arrays allocated as (M, N, 3) float64; orientations allocated but unused in this slice
- `sim.set_potential(V_ext, V_int)` — wires Python callables; `V_ext(r_i, u_i)` and `V_int(r_ij, u_i, u_j)` (no `compute_grad` in this slice — primitive propagator never needs gradients)
- `sim.set_trial_wavefunction(f=None, h=None)` — optional Python callables; called only at endpoint slices for affected pairs
- `sim.add_move(move, weight=1.0)` — registers a move object; returns nothing
- `TranslationEndMove(step_size)` — proposes uniform displacement of one particle at a randomly chosen endpoint slice
- `TranslationInteriorMove(step_size)` — proposes uniform displacement of one particle at a randomly chosen interior slice
- Python `Move` base class with `propose(path_state, rng) -> MoveResult`
- `PathState` Python object exposing: `positions`, `orientations` (numpy arrays, read-only views during a proposal), `N`, `M`, `tau_prime`, per-particle `lambda_trans` values, `slice_kind` array (all `PHYSICAL` for primitive propagator), and writable `buffer_positions` and `buffer_orientations` arrays of shape (M, 1, 3) where moves write proposed coordinates
- Move buffer pre-allocated at `Simulation` construction as two (M, 1, 3) float64 arrays and reused across all move attempts — no per-move allocation
- `MoveResult` carrying `changed: list[tuple[int, int]]` (particle, slice pairs written into the buffer) and `log_ratio_contrib: float`
- `sim.run(blocks: int) -> None` — Python sweep loop: `sweeps_per_block × N × M` move attempts per block, weighted random move selection; each attempt calls `propose()` (which writes into `PathState` buffer arrays), computes the primitive propagator acceptance ratio from the buffer, and copies the buffer into the main path arrays on acceptance — main path arrays are never mutated on rejection
- `sim.acceptance_stats` — dict or object exposing per-move-type attempt and acceptance counts
- `sim.add_observable(callback, every: int, start_after: int = 0) -> int` — returns observable ID
- `sim.remove_observable(obs_id: int)` — deregisters; no-op if ID not found
- Observable callback signature: `callback(block: int, positions: np.ndarray, orientations: np.ndarray)`

**QA target (manual, not implemented here):** harmonic oscillator ground state — single particle in a 1D harmonic potential, verify sampled center-slice position distribution converges to the known Gaussian ground state density.

## Acceptance criteria

- [ ] `Simulation` raises a clear `ValueError` for invalid M (not divisible by 2, or < 3)
- [ ] `Simulation` raises a clear `ValueError` if a `ParticleType` has `lambda_trans=None` (no translational DOF) when no rotational support exists yet
- [ ] `sim.run(blocks)` completes without error for a zero-potential free-particle system
- [ ] `sim.acceptance_stats` reports nonzero attempt counts after `run()`; acceptance rate is 1.0 for a zero-potential system
- [ ] Observable callbacks fire at the correct blocks given `every` and `start_after`; do not fire before `start_after`
- [ ] `sim.remove_observable(id)` prevents subsequent firings
- [ ] `sim.run(blocks)` can be called multiple times on the same instance; block count accumulates correctly
- [ ] Observable callback receives `positions` and `orientations` arrays of shape (M, N, 3)
- [ ] `TranslationEndMove` and `TranslationInteriorMove` are importable and registerable
- [ ] After a rejected move, the main `positions` array is unchanged (buffer is not copied through on rejection)
- [ ] A second `sim.run()` call after `sim.add_observable()` fires the new observable correctly

## Blocked by

None — can start immediately.

## User stories addressed

- User story 1, 2, 3 (partial — `lambda_trans` only), 4 (partial — seed accepted, RNG type deferred to C++ slice)
- User story 6, 7 (partial — MIC deferred to `issues/004`), 9, 11, 12, 13
- User story 14, 15 (end and interior moves only), 18, 19, 20
- User story 21, 22, 23
- User story 24, 25
- User story 28, 29, 30, 31

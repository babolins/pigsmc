# 003 — C++ Sweep Engine

## Parent PRD

`issues/prd.md`

## What to build

Replace the Python sweep loop from `issues/001` with a C++ implementation in `pigsmc._engine`. This is a pure performance upgrade — the user-facing API does not change, and all tests from `issues/001` pass without modification. After this slice, the per-move Python dispatch overhead is eliminated for library-provided moves.

**Scope:**

- `PathState` C++ struct: Eigen-mapped views of `positions` and `orientations` arrays (no copy), `N`, `M`, `tau_prime`, per-particle `lambda_trans` array, `slice_kind` array (all `PHYSICAL` for primitive propagator), boundary info (free only in this slice)
- `MoveResult` C++ struct: `std::vector<std::pair<int,int>> changed` and `double log_ratio_contrib`
- `Move` abstract C++ base class: `virtual MoveResult propose(PathState&, RNG&) = 0`
- pybind11 trampoline so Python subclasses of `Move` work correctly (enabling custom Python moves to be passed to the C++ engine)
- C++ RNG: MT19937-64 and PCG64 implementations, selected by string at construction, seeded by user-supplied integer. RNG lives in C++ and is owned by the engine.
- C++ sweep loop (`run_sweep`): `sweeps_per_block × N × M` weighted-random move attempts; computes primitive propagator acceptance ratio (potential change + trial wavefunction change at endpoints + propagator kinetic ratio); calls Python potential and trial wavefunction callables via pybind11; applies Metropolis accept/reject; accumulates per-move acceptance statistics
- `TranslationEndMove` and `TranslationInteriorMove` ported to C++ as concrete `Move` subclasses
- `Simulation.run(blocks)` in Python calls `engine.run_sweep()` per block (one pybind11 call per block)
- `sim.acceptance_stats` populated from C++ engine stats

The potential and trial wavefunction remain Python callables in this slice — they are called from C++ via pybind11. The engine calls `V_ext` and `V_int` only for affected (particle, slice) pairs (O(N) per single-particle move).

**QA target (manual):** same harmonic oscillator as `issues/001` QA target — verify identical statistical results with the same seed, and measure the speedup over the Python sweep loop.

## Acceptance criteria

- [ ] All tests from `issues/001` pass without any test modifications
- [ ] `Simulation.run(blocks)` delegates the sweep loop to C++ (`engine.run_sweep()` called once per block)
- [ ] A custom Python `Move` subclass (implementing `propose` in Python) can be registered with `sim.add_move()` and called correctly from the C++ sweep loop via the pybind11 trampoline
- [ ] RNG type (`"mt19937_64"` or `"pcg64"`) and seed are accepted at `Simulation` construction and control the C++ RNG
- [ ] The engine calls the user's potential functions only for affected (particle, slice) pairs — verified by counting calls in a test with a mock potential
- [ ] The engine calls trial wavefunction functions only at endpoint slices for affected pairs — verified similarly
- [ ] `sim.acceptance_stats` matches manual counts for a deterministic seed and simple potential
- [ ] Measurable speedup over the Python sweep loop for N=10, M=10, 1000 blocks

## Blocked by

- `issues/002-build-system-setup.md`

## User stories addressed

- User story 4 (RNG type and seed fully implemented in C++)
- User story 21, 22, 23 (pybind11 trampoline enables Python Move subclasses)
- User story 26 (engine evaluates only affected pairs)
- User story 20 (acceptance_stats from C++ engine)

# PRD: pigsmc — Path Integral Ground State Quantum Monte Carlo Library

## Problem Statement

Researchers studying quantum many-body systems with translational and/or rotational degrees of freedom need to perform Path Integral Ground State Quantum Monte Carlo (PIGSMC) simulations. No existing Python library provides a flexible, high-performance, extensible framework for this class of simulation. Researchers are forced to write bespoke simulation codes from scratch for each project, duplicating infrastructure (move types, propagators, checkpointing, observable scheduling) and sacrificing either performance (pure Python) or flexibility (fully compiled codes that are hard to extend).

## Solution

`pigsmc` is a Python library that provides the simulation infrastructure for PIGSMC — move types, propagators, boundary conditions, checkpointing, observable scheduling, and a runtime C++ compilation facility — while leaving the physics (potentials, trial wavefunctions, observables) entirely to the user. The user can prototype entirely in Python and progressively replace hot functions with compiled C++ via a `compile_physics` function without changing any other simulation code. The performance-critical sweep loop runs in C++; the block-level loop and all scheduling logic run in Python for full extensibility.

## User Stories

### Getting Started

1. As a researcher, I want to construct a `Simulation` by specifying the number of particles, number of time slices, time step, propagator type, boundary conditions, and per-particle parameters, so that the library validates my configuration before I invest time in a run.
2. As a researcher, I want the library to enforce that M-1 is divisible by 2 (primitive) or 4 (Chin) and that M meets the minimum slice count, so that I get a clear error at construction rather than a silent wrong result.
3. As a researcher, I want to specify per-particle kinetic parameters via a `ParticleType` dataclass with `lambda_trans` and `lambda_rot` fields (either can be `None` to indicate absent DOF), so that heterogeneous systems and pure-translational or pure-rotational particles are supported naturally.
4. As a researcher, I want to choose between `"mt19937_64"` and `"pcg64"` RNG types and provide a seed at construction, so that my simulation is reproducible and restartable from a checkpoint.
5. As a researcher, I want to specify boundary conditions by constructing a `Boundary` object via factory methods (`Boundary.free()`, `Boundary.periodic_3d(Lx, Ly, Lz)`, `Boundary.quasi_2d(Lx, Ly)`, `Boundary.quasi_1d(Lz)`) — each accepting only the box lengths for its periodic dimensions — and passing it to `Simulation`, so that I can simulate free, fully periodic, slab, and tube geometries without implementing minimum-image logic myself.

### Providing Physics

6. As a researcher, I want to wire in my external potential via `sim.set_potential(V_ext, V_int)` using decomposed one-body and pair functions, so that the library handles pair summation and I only implement the per-particle and per-pair physics.
7. As a researcher, I want `V_int` to receive the minimum-image displacement vector `r_ij = r_i - r_j` (with PBC already applied) as its first argument, so that I do not have to re-implement boundary logic inside my potential function.
8. As a researcher, I want both `V_ext` and `V_int` to accept a `compute_grad=False` flag and return `(value, grad_pos, grad_ori)` when `True`, so that gradients and values are co-located in my implementation (avoiding duplicated math) and gradients are never computed unless required.
9. As a researcher using the primitive propagator, I want to provide only `V_ext` and `V_int` without implementing gradients, so that I can get started quickly without the overhead of gradient derivation.
10. As a researcher using a Chin propagator, I want the library to request gradients automatically (via `compute_grad=True`) only for intermediate slices, so that I implement one function with a flag rather than two separate functions.
11. As a researcher, I want to wire in my trial wavefunction via `sim.set_trial_wavefunction(f=None, h=None)` with decomposed one-body `f(r_i, u_i)` and pair `h(r_ij, u_i, u_j)` terms, so that I can implement only the terms I need (either can be omitted for zero contribution).
12. As a researcher, I want the library to call trial wavefunction functions only at the two endpoint slices and only for particles and pairs affected by a proposed move, so that the O(N) cost is not inflated to O(N²) unnecessarily.
13. As a researcher, I want to swap a Python potential or trial wavefunction for a compiled C++ version without changing any other simulation code, so that I can prototype in Python and accelerate incrementally.

### Configuring Moves

14. As a researcher, I want to register move objects with `sim.add_move(move, weight=1.0)`, so that I can compose any combination of move types with any relative frequencies.
15. As a researcher, I want the library to provide `TranslationEndMove`, `TranslationInteriorMove`, `TranslationRigidMove`, and `TranslationBisectionMove` as ready-to-use move classes, so that I can simulate purely translational systems without writing any move code.
16. As a researcher simulating linear rigid rotors, I want analogous `RotationEndMove`, `RotationInteriorMove`, `RotationRigidMove`, and `RotationBisectionMove` classes, so that rotational degrees of freedom are fully supported out of the box.
17. As a researcher, I want `TranslationBisectionMove(level=L)` to use Lévy bridge proposals over 2^L contiguous slices and to be agnostic to physical/intermediate slice type, so that bisection works correctly for both primitive and Chin propagators with a single move class.
18. As a researcher, I want end moves to randomly select one of the two path endpoints per attempt, so that both endpoints are sampled without requiring me to register two separate move objects.
19. As a researcher, I want `sweeps_per_block` to be tunable, so that I can adjust block granularity based on observed acceptance statistics and system autocorrelation.
20. As a researcher, I want `sim.acceptance_stats` to expose per-move-type and per-bead acceptance counts and attempt counts, so that I can diagnose poor mixing and tune step sizes.
21. As a researcher, I want to implement a custom move in pure Python by subclassing `Move` and implementing `propose(path_state, rng)`, so that I can prototype non-standard moves without writing C++.
45. As a researcher, I want `sim.boundary` to be a read-only property that returns the `Boundary` object I passed at construction, so that I can access it from observable callbacks and other user code without keeping a separate reference.
22. As a researcher, I want `PathState` (passed to moves) to expose positions, orientations, N, M, τ', per-particle λ values, the `Boundary` object, a `slice_kind` array, and writable `buffer_positions` and `buffer_orientations` arrays of shape (M, 1, 3), so that my custom move has all the information it needs to construct a valid proposal and a pre-allocated destination to write proposed coordinates into without touching the main path arrays.
23. As a researcher, I want `MoveResult` to carry the index of the affected particle (`particle: int`), the inclusive slice range updated in the move buffer (`m_lo: int`, `m_hi: int`), and a `log_ratio_contrib` scalar for any kinematic contribution the move owns (e.g. the Lévy bridge factor in bisection), so that the engine can read the buffer to compute potential and wavefunction differences and copy it into the main path arrays only on acceptance — leaving the main path arrays unchanged on rejection. Moves are constrained to affect exactly one particle; when multiple slices are updated they must form a consecutive range `[m_lo, m_hi]`.

### Running Simulations

24. As a researcher, I want `sim.run(blocks)` to execute the requested number of blocks and return, so that I can call it multiple times (e.g. after adjusting step sizes or registering new observables) without restarting.
25. As a researcher, I want the library to handle the logic of what constitutes a sweep (N×M move attempts) automatically, so that I only specify blocks and `sweeps_per_block`.
26. As a researcher, I want the engine to evaluate the potential only for pairs and slices affected by each proposed move (O(N) for single-particle moves), so that I do not pay O(N²) per move attempt.
27. As a researcher using a Chin propagator, I want the engine to automatically evaluate `V` at physical slices and `Ṽ = V + 2u₀τ'W` at intermediate slices, so that the correct propagator is used without any configuration on my part beyond selecting the propagator type.

### Observables

28. As a researcher, I want to register observable callbacks via `sim.add_observable(callback, every=N, start_after=B)`, so that observables are collected periodically after a burn-in period without manual block counting.
29. As a researcher, I want `add_observable` to return an ID I can pass to `sim.remove_observable(id)`, so that I can deregister observables mid-run (e.g. after burn-in is complete) or add new ones after the simulation has been running.
30. As a researcher, I want my observable callback to receive `(block, positions, orientations)` with the full (M, N, 3) path arrays, so that I can implement observables requiring any combination of slices — center slice diagonal estimators, endpoint estimators (local energy), virial estimators, or future off-diagonal estimators.
31. As a researcher, I want my observable callback to be responsible for its own state (accumulating samples, computing statistics, writing output), so that the library does not impose a particular data collection pattern.
32. As a researcher, I want to be able to save coordinate snapshots from within my observable callback, so that I have full control over snapshot frequency and format independently of checkpointing.
46. As a researcher, I want every `Boundary` object to expose `displacement(r_i, r_j)` (NumPy-vectorized, broadcastable) and `all_displacements(positions_slice)` methods that return MIC-corrected pair displacements, so that I can compute pair distances in observable callbacks (e.g. radial distribution function) without re-implementing minimum-image logic myself.

### Checkpointing

33. As a researcher, I want `sim.save_checkpoint(path)` to write a complete HDF5 file containing current coordinates, RNG state, acceptance statistics, and all scalar metadata, so that I can resume a simulation exactly from any block boundary.
34. As a researcher, I want `sim.load_checkpoint(path)` to restore the full simulation state, so that resumed runs are statistically indistinguishable from uninterrupted runs.
35. As a researcher, I want the checkpoint to include the current positions and orientations, so that I do not need to separately save coordinates for resumption (though I may still do so for trajectory analysis).
36. As a researcher, I want the checkpoint format to be HDF5, so that I can inspect checkpoint files with standard scientific tools (h5py, HDFView) without library-specific readers.

### Compiled C++ Extensions

37. As a researcher, I want to call `compile_physics(sources, extra_includes=[], extra_libraries=[], lto=False, cache_dir=".pigsmc_cache")` to compile my C++ physics functions into a Python module, so that I can accelerate hot functions without managing a build system manually.
38. As a researcher, I want `compile_physics` to cache compiled modules and skip recompilation when sources are unchanged, so that repeated imports are fast.
39. As a researcher, I want the library to provide C++ registration macros so that I can expose my functions to Python without writing pybind11 boilerplate, so that the barrier to writing compiled physics is low.
40. As a researcher, I want to pass `extra_includes` and `extra_libraries` to `compile_physics` so that I can use third-party C++ libraries (e.g. custom interaction models) in my physics implementations.
41. As a researcher, I want to implement a custom move in C++ by subclassing the `Move` virtual base class and registering it via `compile_physics`, so that custom moves have zero Python-dispatch overhead in production runs.
47. As a researcher implementing compiled C++ observables or potentials, I want the library header to expose a single `pigsmc::mic_displacement(r_i, r_j, boundary)` function that accepts a `BoundaryDescriptor` struct (obtained from `sim.boundary.descriptor` in Python) and dispatches at runtime, so that my compiled code computes correct MIC-corrected pair displacements regardless of which boundary type the simulation uses — without needing to recompile when I switch boundary types.

### Free Rotor Propagator

42. As a researcher, I want the free rotor propagator to be precomputed on a grid at simulation initialization, so that rotational propagator evaluations during the simulation are fast interpolation lookups rather than expensive Legendre series evaluations.
43. As a researcher, I want to control the Legendre series cutoff `L_max` and grid size, so that I can tune the accuracy/initialization-cost tradeoff for my system's rotational parameters.
44. As a researcher, I want the interpolation to use Fritsch-Carlson monotone cubic interpolation, so that the propagator is guaranteed non-negative (as required for a valid probability weight) and free of Gibbs-like overshoot near the forward-scattering peak.

## Implementation Decisions

### Module Architecture

**`pigsmc.simulation` — `Simulation` class**
The top-level user-facing class. Owns the Python block loop, observable registry and scheduling, checkpoint I/O, and the interface for registering moves and wiring physics. Delegates the sweep loop to the C++ engine. Validates configuration at construction (slice counts, propagator constraints, particle DOF consistency).

**`pigsmc._engine` — C++ pybind11 extension (sweep engine)**
The performance core. Contains: `PathState` (Eigen-mapped views of position/orientation arrays plus system parameters, plus two pre-allocated (M, 1, 3) move buffer arrays into which moves write proposed coordinates), the C++ `Move` virtual base class with pybind11 trampoline, `MoveResult`, the sweep loop (`run_sweep`), acceptance ratio computation (propagator ratio + potential change + trial wavefunction change computed from the move buffer against the current path), accept/reject logic (buffer copied to main path arrays on acceptance; main path arrays untouched on rejection), RNG (MT19937-64 or PCG64), and per-move/per-bead acceptance statistics. Built with scikit-build-core (CMake). Eigen used for array abstraction; `Eigen::Ref<MatrixXd>` for 2D slice views.

**`pigsmc.moves` — Standard move types**
Eight concrete move classes (four translational, four rotational): `EndMove`, `InteriorMove`, `RigidMove`, `BisectionMove` for each DOF type. All subclass the C++ `Move` base. Bisection moves implement Lévy bridge proposals; rigid moves implement uniform displacement/rotation proposals. Moves are slice-type-agnostic; the engine routes potential evaluation via `slice_kind`.

**`pigsmc.propagators` — Propagator implementations**
Encapsulates propagator-specific logic: acceptance ratio contributions for primitive, Chin 4A, and Chin 4B. Owns the free rotor propagator grid (initialization via Legendre sum with configurable `L_max`, Fritsch-Carlson monotone cubic interpolation, per-λ_rot caching). Exposes the `slice_kind` pattern for a given propagator and M.

**`pigsmc.boundaries` — Boundary conditions**
`Boundary` class hierarchy with four concrete subclasses, each constructed via a factory method: `Boundary.free()` (no box lengths), `Boundary.periodic_3d(Lx, Ly, Lz)`, `Boundary.quasi_2d(Lx, Ly)` (periodic x,y; free z), `Boundary.quasi_1d(Lz)` (free x,y; periodic z). Each subclass accepts only the box lengths for its periodic dimensions; passing extra lengths raises `TypeError` at construction. The concrete type is used at `Simulation` construction to select the C++ MIC function pointer, dispatched with zero runtime branching in the hot path; `Boundary.free()` dispatches to a no-op. Positions are stored in **unwrapped coordinates**: particle positions are not constrained to the primary simulation cell and are never wrapped on acceptance; only pair displacements `r_ij` are MIC-corrected. Each `Boundary` object exposes: `displacement(r_i, r_j)` (NumPy, broadcastable — applies MIC to periodic dimensions, returns raw difference for free dimensions) and `all_displacements(positions_slice)` (convenience wrapper for the all-pairs `(N, N, 3)` case given an `(N, 3)` slice). `box_lengths` property exposes only the lengths for periodic dimensions (absent on `FreeBoundary`). `descriptor` property returns a pybind11-wrapped `BoundaryDescriptor` struct for passing boundary information into compiled C++ functions.

**`pigsmc.particles` — `ParticleType`**
A simple dataclass: `lambda_trans: float | None`, `lambda_rot: float | None`. Used at construction to configure per-particle kinetic scales in the engine. `lambda_trans` is defined as λ_trans = ℏτ/(2m) and `lambda_rot` is defined as λ_rot = ℏτ/(2I), where m is the particle mass, I is its moment of inertia, and τ is the imaginary time step. These values already incorporate the time step and are used directly as the Gaussian width in free-propagator exponents (denominator 4λ, not 4λτ').

**`pigsmc.physics` — `compile_physics`**
Runtime C++ compilation using setuptools Extension machinery. Discovers the library's installed include paths and compiled objects. Produces a pybind11 extension module in `cache_dir`, keyed by source hashes. Provides C++ registration macros (via a library header) so users can expose functions without boilerplate. The same header exposes a `pigsmc::BoundaryDescriptor` struct and a single `pigsmc::mic_displacement(Vector3d r_i, Vector3d r_j, const BoundaryDescriptor& bd)` free function. The `BoundaryDescriptor` is obtained from Python via `sim.boundary.descriptor` and passed as an argument to the user's compiled function; dispatch is at runtime so compiled modules are boundary-type-agnostic and do not need to be recompiled when the boundary type changes.

**`pigsmc.checkpoint` — HDF5 checkpoint I/O**
Save and load logic using h5py. HDF5 layout: `/coords/positions`, `/coords/orientations`, `/rng/state`, `/stats/acceptance`, `/metadata` (attrs). Isolated from `Simulation` so it can be tested independently with mock data.

### Key Interface Contracts

- **Potential functions**: `V_ext(r_i, u_i, compute_grad=False)` and `V_int(r_ij, u_i, u_j, compute_grad=False)`. Return scalar when `compute_grad=False`; return `(value, grad_pos, grad_ori)` when `True`. `r_ij` is the MIC-corrected displacement vector.
- **Trial wavefunction functions**: `f(r_i, u_i) -> float` and `h(r_ij, u_i, u_j) -> float`. Both optional. Same `r_ij` convention as potential.
- **Observable callback**: `callback(block: int, positions: np.ndarray, orientations: np.ndarray)`. Arrays have shape (M, N, 3), dtype float64.
- **Move proposal**: `propose()` writes proposed coordinates into `PathState.buffer_positions` and `PathState.buffer_orientations` (pre-allocated (M, 1, 3) arrays, reused across all move attempts), then returns `MoveResult(particle: int, m_lo: int, m_hi: int, log_ratio_contrib: float)`. Moves are constrained to affect exactly one particle (`particle`) over a consecutive range of slices `[m_lo, m_hi]` (inclusive). The engine reads the buffer entries for slices `m_lo..m_hi` to compute kinetic, potential and wavefunction differences; on acceptance it copies those entries into the main path arrays; on rejection the main path arrays are left unchanged.
- **Path coordinate arrays**: Two (M, N, 3) C-contiguous float64 arrays, slice-major order. Always allocated for all N particles regardless of DOF.
- **Slice kind**: Length-M integer array in `PathState` with values `PHYSICAL`/`INTERMEDIATE`. Determines whether engine calls `V` or `Ṽ` when evaluating potential contributions.

### Propagator Constraints
- Primitive: M-1 divisible by 2, minimum M=3.
- Chin 4A / 4B: M-1 divisible by 4, minimum M=5. Pattern is P-I-P-I-P-...
- Chin coefficients: 4A: {v₀=1/6, v₁=2/3, u₀=1/72}; 4B: {v₀=0, v₁=1, u₀=1/24}; t₁=1/2 for both.

### Performance Decisions
- Sweep loop in C++, block loop in Python. One pybind11 call per block eliminates per-move Python overhead.
- Engine recomputes only O(N) affected pairs per move attempt (not O(N²)).
- Move buffer pre-allocated at construction as two (M, 1, 3) Eigen arrays alongside the main path arrays — no per-move heap allocation in the hot path; main path arrays written only on acceptance.
- MIC computation dispatched via C++ function pointer — zero runtime branching in the hot path.
- Free rotor propagator evaluated via precomputed monotone cubic grid — O(1) per evaluation.
- SIMD vectorization via compiler; no explicit multi-threading. Parallelism is embarrassingly parallel (multiple processes, each with its own `Simulation` instance).
- Gradients computed inside the potential function alongside values and never computed unless `compute_grad=True`.

### Build System
- Library C++ extension: scikit-build-core (CMake). Finds Eigen, pybind11, sets SIMD and LTO flags.
- User C++ extensions (`compile_physics`): setuptools Extension at runtime, using the library's installed include/library paths.

## Testing Decisions

**What makes a good test**: Tests should verify observable behavior through public interfaces only — never test internal C++ implementation details, array layout, or private methods. A good test sets up a physically meaningful or analytically tractable scenario, runs the public API, and asserts on the output. Prefer tests where the correct answer is known analytically (e.g. a free particle, a harmonic oscillator, or a known propagator value).

**Modules to test:**

- **Free rotor propagator grid**: Verify that the precomputed grid converges to the analytical sum at a set of x values as L_max increases. Verify that the interpolated values are non-negative across the full domain. Verify that the grid is correctly cached and reused for identical λ_rot values.

- **MIC / boundary conditions**: Construct each `Boundary` subclass and verify correct factory behaviour (e.g. `Boundary.free()` accepts no box lengths; `Boundary.quasi_2d(Lx, Ly)` rejects a third argument). For each type, verify that `r_ij` delivered by the engine is within the correct range (e.g. ‖r_ij‖_∞ ≤ L/2 for periodic dimensions) and that free-dimension components are unmodified. Verify that the Python-side `displacement()` and `all_displacements()` methods produce the same MIC-corrected results as the engine. Verify that particle positions are stored in unwrapped coordinates (positions may exceed `[0, L)` and are never snapped back into the primary cell).

- **Acceptance ratio / detailed balance**: Set up a toy system (e.g. two particles with a known harmonic pair potential, primitive propagator) and verify that the ratio of forward/reverse acceptance probabilities satisfies detailed balance for a set of random moves. This tests the engine's acceptance logic without testing internal implementation.

- **Free-particle path**: For a pure free-particle system (no potential, no trial wavefunction), verify that the sampled path distribution matches the known Gaussian free-propagator distribution analytically — a strong end-to-end correctness check.

- **Observable scheduling**: Verify that callbacks fire at the correct blocks (`every`, `start_after`) and not at others. Verify that `remove_observable` prevents future firings. Verify that callbacks added mid-run fire correctly.

- **Checkpointing**: Save a checkpoint after some blocks, restore into a fresh `Simulation`, continue running, and verify that RNG state, block count, and acceptance statistics are consistent. Verify that the HDF5 file contains the expected groups and attributes.

- **`compile_physics`**: Compile a trivial C++ potential (e.g. always returns 0), load it, wire it into a simulation, and verify the simulation runs without error. Verify that the cache is used on second call. Verify that `extra_includes` allows use of a custom header.

- **`Simulation` configuration validation**: Verify that invalid configurations (wrong M for propagator type, missing potential for Chin propagator, etc.) raise clear errors at construction or at `run()` time, not silently during the sweep.

## Out of Scope

- Long-range electrostatics / Ewald summation — the user is responsible for cutoffs and long-range corrections.
- Worm algorithm / off-diagonal estimators — the interface is designed to be forward-compatible (full path passed to observables), but the worm algorithm is not implemented.
- Non-linear rigid rotors / symmetric tops — only linear rigid rotors (orientation as a 3D unit vector) are supported.
- Non-orthorhombic simulation cells — only orthorhombic boxes are supported.
- GPU / hardware accelerator support.
- Multi-threaded parallelism within a single simulation — parallelism is embarrassingly parallel (multiple processes).
- Automatic step size tuning — the library tracks acceptance statistics; the user is responsible for tuning step sizes.
- Built-in observable implementations (e.g. energy estimators) — the user implements observables as callbacks.
- Trajectory file formats (e.g. XYZ, DCD) — the user is responsible for coordinate snapshot format and frequency.
- Identical-particle exchange / permutation moves — bosonic/fermionic statistics are out of scope.

## Further Notes

- The `λ_trans` and `λ_rot` parameters in `ParticleType` are defined as λ_trans = ℏτ/(2m) and λ_rot = ℏτ/(2I) respectively, where τ is the imaginary time step. These values already incorporate τ, so they appear bare (not multiplied by τ') in free-propagator exponent denominators. The user is responsible for computing them consistently with their chosen unit system and time step. This is distinct from supplying a mass or moment of inertia alone — the time-step dependence is the user's responsibility.
- The time step is supplied as `τ' = τ/ℏ` (i.e. ℏ=1 convention). Physical interpretation is the user's responsibility.
- The observable callback receives the full path (all M slices), not just the center slice, to support local energy estimators (requiring endpoint slices), virial estimators, and potential future off-diagonal estimators requiring adjacent slices.
- `compile_physics` is intended for production use after prototyping; Python implementations of potentials and moves are fully supported throughout and carry no correctness penalty, only performance overhead relative to compiled C++.
- The free rotor propagator grid is precomputed per `λ_rot` value and cached. For simulations with many particle species of different rotational constants, initialization cost scales with the number of distinct species.

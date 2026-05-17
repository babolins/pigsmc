# 007 — Rotational Degrees of Freedom

## Parent PRD

`issues/prd.md`

## What to build

Add support for rotational degrees of freedom (linear rigid rotors). This slice activates the `lambda_rot` field of `ParticleType`, implements the free rotor propagator as a precomputed interpolation grid, and provides all four standard rotation move classes. After this slice, systems of linear rigid rotors (or mixed translational+rotational particles) can be simulated.

**Scope:**

**ParticleType and validation:**
- `lambda_rot: float | None` activated in `ParticleType`; `None` means no rotational DOF
- `Simulation` validates: if any particle has `lambda_rot` set, all particles must have `lambda_rot` set (mixed pure-translational and rotational particles deferred; for now all-or-nothing)
- Orientation arrays `positions`-paired `orientations` (M, N, 3) are now live; initialized as unit vectors by the user (library does not set initial conditions)

**Free rotor propagator grid:**
- At `Simulation` construction, precompute a 1D grid over `x = û·û' ∈ [-1, 1]` for each unique `lambda_rot` value among particles
- Grid values: exact Legendre series `Σ_{l=0}^{L_max} (2l+1)/(4π) P_l(x) exp(-t_eff λ_rot l(l+1))` where `t_eff` accounts for propagator type and step fraction
- `L_max` user-configurable with a sane default (e.g. 100); grid size user-configurable (e.g. 1000 points)
- Interpolation: Fritsch-Carlson monotone cubic (guarantees non-negative values)
- Grid cached per `lambda_rot` value within a simulation instance

**Rotation move classes** (all C++ `Move` subclasses):
- `RotationEndMove(step_size: float)` — proposes a random small rotation of one particle at a randomly chosen endpoint slice; rotation drawn uniformly from a cone of half-angle `step_size` (radians) around the current orientation; result normalized to unit vector
- `RotationInteriorMove(step_size: float)` — same but at a randomly chosen interior slice
- `RotationRigidMove(step_size: float)` — applies the same random rotation to all M slices of one particle
- `RotationBisectionMove(level: int)` — Lévy bridge on the sphere: proposes new orientations for interior slices of a 2^level range using the free rotor propagator as the proposal distribution; `MoveResult.log_ratio_contrib` contains the spherical Lévy bridge contribution

**QA target (manual):** free rotor (no potential) — verify that the sampled distribution of `û·û'` between adjacent slices matches the free rotor propagator values from the precomputed grid.

## Acceptance criteria

- [ ] `ParticleType(lambda_trans=1.0, lambda_rot=0.5)` is accepted; `lambda_rot=None` continues to work for pure-translational particles
- [ ] The free rotor propagator grid is precomputed at construction and interpolated values are non-negative across all `x ∈ [-1, 1]`
- [ ] Grid values converge as `L_max` increases — verified in a test comparing small and large `L_max` values at a fixed `x`
- [ ] `RotationEndMove`, `RotationInteriorMove`, `RotationRigidMove`, `RotationBisectionMove` are importable and registerable
- [ ] For a free rotor (no potential), `RotationBisectionMove` acceptance rate is 1.0
- [ ] Orientation vectors remain unit vectors (within floating-point tolerance) after any move
- [ ] `L_max` and grid size are configurable at `Simulation` construction
- [ ] All prior tests pass (rotational DOF is opt-in; existing pure-translational tests unaffected)

## Blocked by

- `issues/003-cpp-sweep-engine.md`

## User stories addressed

- User story 3 (`lambda_rot` in `ParticleType`)
- User story 16 (all four rotation move classes)
- User story 17 (rotational bisection, slice-type-agnostic)
- User story 42, 43, 44 (free rotor propagator grid)

# 009 — Checkpointing

## Parent PRD

`issues/prd.md`

## What to build

Add HDF5-based checkpoint save and load so that long-running simulations can be interrupted and resumed exactly. The library owns the full checkpoint including current coordinates; the user is responsible for separate trajectory/snapshot output.

**Scope:**

- `sim.save_checkpoint(path: str | Path)` — writes a complete HDF5 file using h5py
- `sim.load_checkpoint(path: str | Path)` — restores full simulation state from an HDF5 file; raises a clear error if the checkpoint was produced by a simulation with incompatible configuration (different N, M, propagator type, etc.)
- HDF5 layout:
  - `/coords/positions` — (M, N, 3) float64
  - `/coords/orientations` — (M, N, 3) float64
  - `/rng/state` — serialized C++ RNG state (BitGenerator state stored as a byte array or structured dataset)
  - `/stats/acceptance` — per-move-type attempt and acceptance counts
  - `/metadata` attrs: `block_count`, `sweeps_per_block`, `tau_prime`, `N`, `M`, `propagator`, `boundary`, `box_lengths`, `rng_type`, library version string
- `sim.load_checkpoint()` restores: coordinates, RNG state, block count, acceptance stats, `sweeps_per_block`
- Validation on load: N, M, propagator type, and boundary type in the checkpoint must match the current `Simulation` configuration; mismatches raise a clear `ValueError` with a descriptive message
- `h5py` added as a package dependency

**QA target (manual):** run 50 blocks, save checkpoint, construct a new `Simulation` with the same configuration, load checkpoint, run 50 more blocks — verify that the final state is statistically identical to an uninterrupted 100-block run with the same seed.

## Acceptance criteria

- [ ] `sim.save_checkpoint(path)` creates a valid HDF5 file readable by h5py outside the library
- [ ] `sim.load_checkpoint(path)` restores positions, orientations, block count, acceptance stats, and `sweeps_per_block`
- [ ] After loading a checkpoint and running additional blocks, `sim.block_count` reflects the total (pre-checkpoint + post-load) block count
- [ ] Loading a checkpoint into a `Simulation` with mismatched N or M raises `ValueError`
- [ ] Loading a checkpoint into a `Simulation` with mismatched propagator type raises `ValueError`
- [ ] The HDF5 file contains `/coords/positions`, `/coords/orientations`, `/rng/state`, `/stats/acceptance`, and `/metadata` with all required attributes
- [ ] `sim.save_checkpoint()` is callable at any point during `run()` (i.e. from within an observable callback) without corrupting state
- [ ] All prior tests pass

## Blocked by

- `issues/003-cpp-sweep-engine.md`

## User stories addressed

- User story 4 (RNG restartability from checkpoint)
- User story 33, 34, 35, 36

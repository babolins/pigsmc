# 010 — compile_physics

## Parent PRD

`issues/prd.md`

## What to build

Add `compile_physics`: a runtime C++ compilation facility that lets users compile their potential, trial wavefunction, move, and observable implementations into a pybind11 extension module and use them with `Simulation` without any change to the simulation setup code. This is the production performance path after Python prototyping.

**Scope:**

- `compile_physics(sources: list[str | Path], extra_includes: list[str] = [], extra_libraries: list[str] = [], lto: bool = False, cache_dir: str | Path = ".pigsmc_cache") -> types.ModuleType`
- Compilation via setuptools `Extension` machinery invoked programmatically at runtime
- The library installs a C++ header (`pigsmc/physics.hpp` or similar) providing registration macros:
  - `PIGSMC_REGISTER_V_EXT(name, fn)` — registers `fn` as `V_ext`-compatible (scalar or grad-returning depending on signature)
  - `PIGSMC_REGISTER_V_INT(name, fn)` — registers `V_int`-compatible function
  - `PIGSMC_REGISTER_F(name, fn)` — registers one-body trial wavefunction term
  - `PIGSMC_REGISTER_H(name, fn)` — registers pair trial wavefunction term
  - `PIGSMC_REGISTER_MOVE(name, cls)` — registers a `Move` subclass
- The same library header exposes a `pigsmc::BoundaryDescriptor` struct and a single `pigsmc::mic_displacement(Vector3d r_i, Vector3d r_j, const BoundaryDescriptor& bd)` free function for computing MIC-corrected pair displacements in compiled C++ code. `BoundaryDescriptor` encapsulates boundary kind and box lengths; it is obtained from Python via `sim.boundary.descriptor` and passed as an argument to user-compiled functions. Because dispatch is at runtime, a compiled module is boundary-type-agnostic: switching from `Boundary.quasi_2d` to `Boundary.periodic_3d` in Python requires no C++ code change and no recompilation.
- `compile_physics` discovers the library's installed include paths and compiled object files automatically (no manual `-I` or `-L` flags required for library headers)
- `extra_includes` and `extra_libraries` allow users to link against third-party libraries
- Caching: compiled modules are stored in `cache_dir` keyed by a hash of source file contents and compiler flags; subsequent calls with unchanged sources return the cached module without recompilation
- `lto=True` enables link-time optimization between user code and the library (best-effort; silently ignored if the compiler does not support it)
- Returned module contains all registered functions/classes as Python-callable objects, usable directly with `sim.set_potential()`, `sim.set_trial_wavefunction()`, and `sim.add_move()`
- The compiled functions accept and return the same types as their Python equivalents (numpy arrays, floats, `MoveResult`) — pybind11 handles conversion

**QA target (manual):** implement a harmonic potential in C++, compile with `compile_physics`, wire into a simulation, and verify that results match the Python implementation with the same seed; measure speedup.

## Acceptance criteria

- [ ] `compile_physics(["my_potential.cpp"])` returns a Python module without error given a valid C++ source file using the registration macros
- [ ] Functions registered with `PIGSMC_REGISTER_V_EXT` and `PIGSMC_REGISTER_V_INT` are callable from Python and accepted by `sim.set_potential()`
- [ ] A class registered with `PIGSMC_REGISTER_MOVE` is instantiable from Python and accepted by `sim.add_move()`
- [ ] A second call to `compile_physics` with unchanged sources returns the cached module (no recompilation) — verified by checking that the `.so` file modification time does not change
- [ ] A call with changed sources triggers recompilation
- [ ] `extra_includes` allows a user header to be included in the compiled source
- [ ] `pigsmc::mic_displacement(r_i, r_j, sim.boundary.descriptor)` is callable from user C++ code with any `Boundary` type and returns the correct MIC-corrected vector — verified by a compiled test that uses the same module with two different boundary types without recompiling
- [ ] A simulation using compiled C++ functions produces identical results to the Python equivalent given the same seed
- [ ] A clear error is raised if compilation fails, with the compiler error message included
- [ ] All prior tests pass

## Blocked by

- `issues/003-cpp-sweep-engine.md`

## User stories addressed

- User story 13 (swap Python for compiled C++ without changing simulation code)
- User story 37, 38, 39, 40, 41
- User story 47 (C++ `mic_displacement` overloads in the library header)

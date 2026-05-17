# 002 — Build System Setup

## Parent PRD

`issues/prd.md`

## What to build

Configure scikit-build-core (CMake) as the library's C++ build system. This slice produces a compilable but minimal pybind11 extension — a "hello world" C++ module that imports successfully — proving the full toolchain works end-to-end: CMake finds Eigen and pybind11, compiles a `.so`, and `import pigsmc._engine` succeeds. No simulation logic is implemented here; that is `issues/003`.

**Scope:**

- Replace `uv_build` with `scikit-build-core` in `pyproject.toml`
- `CMakeLists.txt` at the project root configuring:
  - C++17 minimum
  - `find_package(pybind11)` (or fetch via CMake FetchContent)
  - `find_package(Eigen3)` (or fetch via CMake FetchContent)
  - A `pigsmc._engine` pybind11 extension target with a stub module (single `hello()` function or version string)
  - Compiler flags: `-O3`, `-march=native` (or equivalent), `-fvisibility=hidden`
  - Optional LTO flag (off by default, enabled via CMake option)
- CI-ready: `pip install -e .` succeeds in a clean environment
- Existing pure-Python tests from `issues/001` continue to pass (the extension is additive)

**QA target:** `python -c "import pigsmc._engine; print(pigsmc._engine.__doc__)"` succeeds in a fresh virtualenv after `pip install -e .`.

## Acceptance criteria

- [ ] `pip install -e .` succeeds from a clean checkout with only Python and a C++ compiler available
- [ ] `import pigsmc._engine` succeeds in Python
- [ ] CMake correctly locates or fetches Eigen3 and pybind11
- [ ] The extension compiles with `-O3` and hidden visibility
- [ ] `pip install -e .` is re-runnable (incremental builds work)
- [ ] All existing Python tests from `issues/001` pass after the build system change

## Blocked by

- `issues/001-minimal-working-simulation.md`

## User stories addressed

- Foundational infrastructure for user stories 37–41 (`compile_physics`)
- Enables user story 26 (C++ performance) via `issues/003`

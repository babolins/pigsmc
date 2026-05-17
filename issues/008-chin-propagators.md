# 008 — Chin Propagators (4A and 4B)

## Parent PRD

`issues/prd.md`

## What to build

Add support for the fourth-order Chin 4A and Chin 4B short-time propagator approximations. These require an expanded path with alternating physical (P) and intermediate (I) slices (P-I-P-I-P pattern), modified potential evaluation at I-slices using `Ṽ = V + 2u₀τ'W`, and gradient computation from the user's potential functions.

**Scope:**

**Propagator selection and validation:**
- `propagator` parameter accepts `"primitive"` (existing), `"chin4a"`, and `"chin4b"`
- Chin coefficients: 4A: `{v₀=1/6, v₁=2/3, u₀=1/72, t₁=1/2}`; 4B: `{v₀=0, v₁=1, u₀=1/24, t₁=1/2}`
- Validation: for Chin propagators, M-1 must be divisible by 4 and M ≥ 5; clear `ValueError` at construction otherwise

**slice_kind array:**
- `PathState.slice_kind`: length-M array with values `PHYSICAL` (even indices 0,2,4,...) and `INTERMEDIATE` (odd indices 1,3,5,...)
- Exposed to custom moves so they can query slice type if needed

**Potential routing:**
- Engine calls `V_ext(r_i, u_i, compute_grad=False)` and `V_int(r_ij, u_i, u_j, compute_grad=False)` for P-slices (as before)
- Engine calls `V_ext(r_i, u_i, compute_grad=True)` and `V_int(r_ij, u_i, u_j, compute_grad=True)` for I-slices to obtain `(value, grad_pos, grad_ori)`
- Engine computes `W = Σ_i [λ_trans_i |∇_{r_i} V|² + λ_rot_i |û_i × ∇_{û_i} V|²]` at I-slices
- Engine evaluates `Ṽ = V + 2u₀τ'W` at I-slices for use in the acceptance ratio
- If `compute_grad=True` is called but the user's potential does not return gradients (returns a scalar), a clear `RuntimeError` is raised at first use

**Bisection move elementary step:**
- `TranslationBisectionMove` and `RotationBisectionMove` use `t₁τ'` (= `τ'/2`) as the elementary step size between consecutive slices for Chin propagators, `τ'` for primitive — determined by the engine at sweep time, not by the move class

**QA target (manual):** harmonic oscillator with Chin 4A vs primitive — same system, same τ', verify that Chin 4A achieves better convergence of the ground state energy with fewer slices.

## Acceptance criteria

- [ ] `Simulation(propagator="chin4a", M=9, ...)` constructs without error; `M=7` (not divisible by 4 minus 1) raises `ValueError`
- [ ] `Simulation(propagator="chin4a", M=3, ...)` raises `ValueError` (M < 5)
- [ ] `PathState.slice_kind` has value `PHYSICAL` at even indices and `INTERMEDIATE` at odd indices
- [ ] Engine calls `compute_grad=True` only for I-slices — verified with a mock potential that records the `compute_grad` flag for each call
- [ ] Engine calls `compute_grad=False` for all P-slices
- [ ] Using a Chin propagator with a potential that does not support `compute_grad=True` raises a clear `RuntimeError`
- [ ] `TranslationBisectionMove` uses `t₁τ'` as the elementary step for Chin propagators — verified by checking that free-particle acceptance rate remains 1.0
- [ ] All prior tests (primitive propagator) pass unmodified

## Blocked by

- `issues/003-cpp-sweep-engine.md`
- `issues/007-rotational-degrees-of-freedom.md` (for rotational Ṽ gradient term `λ_rot_i |û_i × ∇_{û_i} V|²`)

## User stories addressed

- User story 2 (Chin M constraints)
- User story 8 (`compute_grad` flag, gradient co-location)
- User story 10 (gradients requested only at I-slices)
- User story 27 (Ṽ routing at intermediate slices)
- User story 17 (bisection uses correct elementary step for Chin)

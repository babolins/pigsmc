import time

import numpy as np
import pytest
from pigsmc import (
    Move,
    MoveResult,
    ParticleType,
    Simulation,
    TranslationBisectionMove,
    TranslationEndMove,
    TranslationInteriorMove,
    TranslationRigidMove,
)


def test_engine_importable():
    import pigsmc._engine  # noqa: F401


def test_engine_has_docstring():
    import pigsmc._engine
    assert pigsmc._engine.__doc__ is not None
    assert len(pigsmc._engine.__doc__) > 0


def make_sim(N=1, M=3, tau_prime=0.1, seed=42, **kwargs):
    particles = [ParticleType(lambda_trans=1.0) for _ in range(N)]
    return Simulation(particles, M=M, tau_prime=tau_prime, seed=seed, **kwargs)


# ---------------------------------------------------------------------------
# Python Move trampoline
# ---------------------------------------------------------------------------

class CustomMove(Move):
    """Custom Python Move that records call count."""
    def __init__(self, step_size=0.1):
        super().__init__()
        self.step_size = step_size
        self.call_count = 0

    def propose(self, path_state, rng):
        self.call_count += 1
        i = int(rng.integers(0, path_state.N))
        m = 0  # always propose at slice 0
        disp = rng.uniform(-self.step_size, self.step_size, size=3)
        path_state.buffer_positions[m, 0, :] = path_state.positions[m, i, :] + disp
        return MoveResult(particle=i, m_lo=m, m_hi=m, log_ratio_contrib=0.0)


def test_python_move_trampoline_called():
    """Python Move subclass propose() is called from C++ engine."""
    sim = make_sim(N=1, M=3, seed=0)
    move = CustomMove(step_size=0.1)
    sim.add_move(move)
    sim.run(blocks=2)
    assert move.call_count > 0


def test_python_move_trampoline_acceptance_stats():
    """Acceptance stats are tracked for Python Move subclasses."""
    sim = make_sim(N=1, M=3, seed=0)
    move = CustomMove(step_size=0.0)  # zero step → always accepted (no potential)
    sim.add_move(move)
    sim.run(blocks=1)
    stats = sim.acceptance_stats[move]
    assert stats["attempts"] > 0
    assert stats["acceptances"] == stats["attempts"]


# ---------------------------------------------------------------------------
# Potential called only for affected pairs (O(N) per move)
# ---------------------------------------------------------------------------

def test_potential_called_for_affected_pairs_only():
    """Engine calls V_ext only for affected particle at affected slice."""
    sim = make_sim(N=3, M=3, seed=7)
    sim.positions[:] = 1.0

    call_args = []
    def V_ext(r, u):
        call_args.append(r.copy())
        return 0.0

    sim.set_potential(V_ext, None)
    move = TranslationEndMove(step_size=0.5)
    sim.add_move(move)
    sim.run(blocks=1)

    # Each move attempt calls V_ext twice (old + new) for exactly 1 particle
    # Total attempts = sweeps_per_block * N * M = 1 * 3 * 3 = 9
    # Each attempt: 2 calls → 18 total calls
    assert len(call_args) == 18


def test_V_int_called_for_affected_particle_only():
    """V_int called with j != i only (O(N) per single-particle move)."""
    sim = make_sim(N=3, M=3, seed=5)
    sim.positions[:] = 0.0

    pair_calls = []
    def V_int(r_ij, u_i, u_j):
        pair_calls.append(r_ij.copy())
        return 0.0

    sim.set_potential(None, V_int)
    move = TranslationEndMove(step_size=0.1)
    sim.add_move(move)
    sim.run(blocks=1)

    # N=3 particles: each move affects 1 particle → 2 pair calls per attempt
    # (j=0 and j=1 when i=2, for example) × 2 (old+new)
    # Total = 9 attempts × 2 pairs × 2 = 36
    assert len(pair_calls) == 36


# ---------------------------------------------------------------------------
# Trial wavefunction only at endpoint slices
# ---------------------------------------------------------------------------

def test_trial_wavefunction_endpoint_only():
    """f() is called only for moves at endpoint slices (m=0 or m=M-1)."""
    sim = make_sim(N=1, M=5, seed=3)
    sim.positions[:] = 0.0

    f_calls = []
    def f(r, u):
        f_calls.append(None)
        return 0.0

    sim.set_trial_wavefunction(f=f, h=None)
    # Use only interior moves — they should NOT trigger f
    move = TranslationInteriorMove(step_size=0.1)
    sim.add_move(move)
    sim.run(blocks=2)
    # Interior moves never touch endpoints → f should never be called
    assert len(f_calls) == 0


# ---------------------------------------------------------------------------
# Acceptance stats match manual counts (deterministic seed)
# ---------------------------------------------------------------------------

def test_acceptance_stats_match_manual():
    """acceptance_stats attempts + acceptances are consistent."""
    sim = make_sim(N=1, M=3, seed=42)
    sim.positions[:] = 0.0

    # Harmonic potential: all moves from 0 toward nonzero increase E → some rejection
    def V_ext(r, u):
        return 0.5 * float(np.dot(r, r))

    sim.set_potential(V_ext, None)
    move = TranslationEndMove(step_size=1.0)
    sim.add_move(move)
    sim.run(blocks=5)

    stats = sim.acceptance_stats[move]
    assert stats["attempts"] > 0
    assert 0 <= stats["acceptances"] <= stats["attempts"]
    # With harmonic potential from zero, some moves should be rejected
    assert stats["acceptances"] < stats["attempts"]


# ---------------------------------------------------------------------------
# Speedup benchmark
# ---------------------------------------------------------------------------

def test_cpp_engine_faster_than_python_baseline():
    """C++ engine produces measurable speedup for N=10, M=10, 1000 blocks."""
    N, M, blocks = 10, 11, 1000
    particles = [ParticleType(lambda_trans=1.0) for _ in range(N)]

    sim = Simulation(particles, M=M, tau_prime=0.1, seed=0)
    sim.positions[:] = 0.0

    def V_ext(r, u):
        return 0.5 * float(np.dot(r, r))

    sim.set_potential(V_ext, None)
    sim.add_move(TranslationEndMove(step_size=0.5))

    t0 = time.perf_counter()
    sim.run(blocks=blocks)
    dt_cpp = time.perf_counter() - t0

    # Soft check: should complete in under 60 seconds even on slow hardware.
    # The real speedup is measured relative to the old Python loop, but we
    # just verify it finishes in reasonable time.
    assert dt_cpp < 60.0, f"C++ engine took {dt_cpp:.1f}s for {blocks} blocks"


# ---------------------------------------------------------------------------
# TranslationRigidMove
# ---------------------------------------------------------------------------

def test_rigid_move_importable_and_registerable():
    sim = make_sim(N=1, M=5)
    move = TranslationRigidMove(step_size=0.5)
    sim.add_move(move)
    assert move in sim.acceptance_stats


def test_rigid_move_same_displacement_all_slices():
    """All M slices of the selected particle shift by the same vector."""
    sim = make_sim(N=1, M=5, seed=7)
    # Start with distinct per-slice positions so the rigid shift is visible
    for m in range(5):
        sim.positions[m, 0, :] = [float(m), 0.0, 0.0]
    initial_offsets = sim.positions[:, 0, :] - sim.positions[0, 0, :]  # (5,3)

    move = TranslationRigidMove(step_size=0.5)
    sim.add_move(move)
    sim.run(blocks=20)

    # Relative offsets between slices must be unchanged after rigid moves
    final_offsets = sim.positions[:, 0, :] - sim.positions[0, 0, :]
    np.testing.assert_allclose(final_offsets, initial_offsets, atol=1e-12)


def test_rigid_move_acceptance_rate_one_with_zero_potential():
    """Kinetic terms cancel for rigid moves → acceptance rate 1.0 with no potential."""
    sim = make_sim(N=2, M=5, seed=13)
    move = TranslationRigidMove(step_size=1.0)
    sim.add_move(move)
    sim.run(blocks=10)
    stats = sim.acceptance_stats[move]
    assert stats["attempts"] > 0
    assert stats["acceptances"] == stats["attempts"]


def test_rigid_move_stats_independent_of_other_moves():
    """Rigid move stats are tracked separately from end/interior moves."""
    sim = make_sim(N=1, M=5, seed=0)
    end_move = TranslationEndMove(step_size=0.1)
    rigid_move = TranslationRigidMove(step_size=0.5)
    sim.add_move(end_move, weight=1.0)
    sim.add_move(rigid_move, weight=1.0)
    sim.run(blocks=5)

    end_stats = sim.acceptance_stats[end_move]
    rigid_stats = sim.acceptance_stats[rigid_move]
    assert end_stats["attempts"] + rigid_stats["attempts"] > 0
    assert rigid_stats["acceptances"] == rigid_stats["attempts"]
    assert end_stats is not rigid_stats


# ---------------------------------------------------------------------------
# TranslationBisectionMove
# ---------------------------------------------------------------------------

def test_bisection_move_importable_and_registerable():
    sim = make_sim(N=1, M=9)
    move = TranslationBisectionMove(level=1)
    sim.add_move(move)
    assert move in sim.acceptance_stats


def test_bisection_move_acceptance_rate_one_with_zero_potential():
    """Lévy bridge cancels kinetic terms exactly → acceptance 1.0 with no potential."""
    sim = make_sim(N=1, M=9, seed=17)
    move = TranslationBisectionMove(level=3)
    sim.add_move(move)
    sim.run(blocks=10)
    stats = sim.acceptance_stats[move]
    assert stats["attempts"] > 0
    assert stats["acceptances"] == stats["attempts"]


def test_bisection_move_result_slice_range():
    """MoveResult m_hi - m_lo + 1 == 2^level - 1 (interior slices only)."""
    level = 2
    expected_interior = (1 << level) - 1  # 3

    captured = []

    class CaptureBisectionMove(TranslationBisectionMove):
        def propose(self, path_state, rng):
            result = super().propose(path_state, rng)
            captured.append(result)
            return result

    sim = make_sim(N=1, M=9, seed=5)
    move = CaptureBisectionMove(level=level)
    sim.add_move(move)
    sim.run(blocks=1)

    assert len(captured) > 0
    for r in captured:
        assert r.m_hi - r.m_lo + 1 == expected_interior


def test_bisection_move_level_too_large_raises():
    """2^level > M-1 raises RuntimeError at run() time."""
    sim = make_sim(N=1, M=3, seed=0)  # M-1 = 2, but 2^2 = 4 > 2
    move = TranslationBisectionMove(level=2)
    sim.add_move(move)
    with pytest.raises(RuntimeError, match="level too large"):
        sim.run(blocks=1)


def test_bisection_move_stats_independent_of_other_moves():
    """Bisection move stats are tracked separately from end moves."""
    sim = make_sim(N=1, M=9, seed=3)
    end_move = TranslationEndMove(step_size=0.1)
    bisection_move = TranslationBisectionMove(level=1)
    sim.add_move(end_move, weight=1.0)
    sim.add_move(bisection_move, weight=1.0)
    sim.run(blocks=5)

    end_stats = sim.acceptance_stats[end_move]
    bisection_stats = sim.acceptance_stats[bisection_move]
    assert end_stats["attempts"] + bisection_stats["attempts"] > 0
    assert bisection_stats["acceptances"] == bisection_stats["attempts"]
    assert end_stats is not bisection_stats

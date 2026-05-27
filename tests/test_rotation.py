"""Tests for rotational degrees of freedom (issue 007)."""
import numpy as np
import pytest
from pigsmc import (
    ParticleType,
    Simulation,
    RotationBisectionMove,
    RotationEndMove,
    RotationInteriorMove,
    RotationRigidMove,
)
from pigsmc.propagators import FreeRotorGrid


def make_rot_sim(N=1, M=3, tau_prime=0.1, lambda_rot=0.5, seed=42, **kwargs):
    particles = [ParticleType(lambda_trans=1.0, lambda_rot=lambda_rot) for _ in range(N)]
    return Simulation(particles, M=M, tau_prime=tau_prime, seed=seed, **kwargs)


# ---------------------------------------------------------------------------
# ParticleType and Simulation validation
# ---------------------------------------------------------------------------

def test_particle_type_lambda_rot_accepted():
    p = ParticleType(lambda_trans=1.0, lambda_rot=0.5)
    assert p.lambda_rot == 0.5
    assert p.lambda_trans == 1.0


def test_particle_type_lambda_rot_none_unchanged():
    p = ParticleType(lambda_trans=1.0)
    assert p.lambda_rot is None


def test_simulation_all_lambda_rot_set():
    """Simulation with all particles having lambda_rot is accepted."""
    sim = make_rot_sim(N=2)
    assert sim is not None


def test_simulation_mixed_lambda_rot_raises():
    """Mixed translational + rotational particles raise ValueError."""
    particles = [
        ParticleType(lambda_trans=1.0, lambda_rot=0.5),
        ParticleType(lambda_trans=1.0),
    ]
    with pytest.raises(ValueError, match="lambda_rot"):
        Simulation(particles, M=3, tau_prime=0.1)


# ---------------------------------------------------------------------------
# Free rotor propagator grid
# ---------------------------------------------------------------------------

def test_free_rotor_grid_log_finite():
    """log G is finite across x ∈ [-1, 1]."""
    grid = FreeRotorGrid(lambda_rot=0.5, L_max=50, grid_size=200)
    xs = np.linspace(-1.0, 1.0, 500)
    vals = grid.log_eval(xs)
    assert np.all(np.isfinite(vals)), f"non-finite log G at some x"


def test_free_rotor_grid_converges_with_L_max():
    """log G values converge as L_max increases."""
    x = 0.5
    small = FreeRotorGrid(lambda_rot=0.5, L_max=10, grid_size=500).log_eval(x)
    large = FreeRotorGrid(lambda_rot=0.5, L_max=100, grid_size=500).log_eval(x)
    assert abs(float(large) - float(small)) < 1e-4


def test_free_rotor_grid_log_eval():
    """log_eval is finite and monotone-increasing toward x=1."""
    grid = FreeRotorGrid(lambda_rot=0.5, L_max=50, grid_size=500)
    log_at_0 = float(grid.log_eval(0.0))
    log_at_1 = float(grid.log_eval(1.0))
    assert np.isfinite(log_at_0)
    assert log_at_1 > log_at_0


def test_free_rotor_grid_configurable():
    """L_max and grid_size are configurable."""
    g1 = FreeRotorGrid(lambda_rot=0.5, L_max=20, grid_size=100)
    g2 = FreeRotorGrid(lambda_rot=0.5, L_max=100, grid_size=1000)
    assert g1 is not g2  # different objects


# ---------------------------------------------------------------------------
# Simulation-level L_max / grid_size passthrough
# ---------------------------------------------------------------------------

def test_simulation_L_max_grid_size_configurable():
    """Simulation accepts L_max and grid_size kwargs when particles have lambda_rot."""
    sim = make_rot_sim(L_max=50, grid_size=200)
    assert sim is not None


# ---------------------------------------------------------------------------
# Rotation move imports and registration
# ---------------------------------------------------------------------------

def test_rotation_moves_importable():
    """All four rotation move classes are importable."""
    assert RotationEndMove is not None
    assert RotationInteriorMove is not None
    assert RotationRigidMove is not None
    assert RotationBisectionMove is not None


def test_rotation_end_move_registerable():
    sim = make_rot_sim(N=1, M=5)
    move = RotationEndMove(step_size=0.3)
    sim.add_move(move)
    assert move in sim.acceptance_stats


def test_rotation_interior_move_registerable():
    sim = make_rot_sim(N=1, M=5)
    move = RotationInteriorMove(step_size=0.3)
    sim.add_move(move)
    assert move in sim.acceptance_stats


def test_rotation_rigid_move_registerable():
    sim = make_rot_sim(N=1, M=5)
    move = RotationRigidMove(step_size=0.3)
    sim.add_move(move)
    assert move in sim.acceptance_stats


def test_rotation_bisection_move_registerable():
    sim = make_rot_sim(N=1, M=9)
    move = RotationBisectionMove(level=1)
    sim.add_move(move)
    assert move in sim.acceptance_stats


# ---------------------------------------------------------------------------
# Rotation move acceptance ratio for free rotor (no potential)
# ---------------------------------------------------------------------------

def test_rotation_rigid_acceptance_rate_one_free_rotor():
    """Rigid move has acceptance 1.0 for free rotor: all dot products are invariant."""
    sim = make_rot_sim(N=1, M=5, seed=7)
    sim.orientations[:] = 0.0
    sim.orientations[:, :, 2] = 1.0

    move = RotationRigidMove(step_size=0.5)
    sim.add_move(move)
    sim.run(blocks=50)
    stats = sim.acceptance_stats[move]
    assert stats["attempts"] > 0
    assert stats["acceptances"] == stats["attempts"], (
        f"Expected 100% acceptance, got {stats['acceptances']}/{stats['attempts']}"
    )


def test_rotation_bisection_acceptance_rate_free_rotor():
    """Bisection move has acceptance < 1 for free rotor: kinetic terms enter the ratio."""
    sim = make_rot_sim(N=1, M=9, seed=42)
    sim.orientations[:] = 0.0
    sim.orientations[:, :, 2] = 1.0

    move = RotationBisectionMove(level=3)
    sim.add_move(move)
    sim.run(blocks=100)
    stats = sim.acceptance_stats[move]
    assert stats["attempts"] > 0
    assert stats["acceptances"] > 0, "All proposals rejected"
    assert stats["acceptances"] < stats["attempts"], (
        f"Expected acceptance < 1 (kinetic terms not cancelled), "
        f"got {stats['acceptances']}/{stats['attempts']}"
    )


# ---------------------------------------------------------------------------
# Unit vector invariant after moves
# ---------------------------------------------------------------------------

def _is_unit(u, atol=1e-10):
    norms = np.linalg.norm(u, axis=-1)
    return np.allclose(norms, 1.0, atol=atol)


def test_rotation_end_move_preserves_unit_vectors():
    sim = make_rot_sim(N=2, M=5, seed=1)
    sim.orientations[:] = 0.0
    sim.orientations[:, :, 0] = 1.0
    move = RotationEndMove(step_size=0.5)
    sim.add_move(move)
    sim.run(blocks=10)
    assert _is_unit(sim.orientations)


def test_rotation_interior_move_preserves_unit_vectors():
    sim = make_rot_sim(N=2, M=5, seed=2)
    sim.orientations[:] = 0.0
    sim.orientations[:, :, 0] = 1.0
    move = RotationInteriorMove(step_size=0.5)
    sim.add_move(move)
    sim.run(blocks=10)
    assert _is_unit(sim.orientations)


def test_rotation_rigid_move_preserves_unit_vectors():
    sim = make_rot_sim(N=2, M=5, seed=3)
    sim.orientations[:] = 0.0
    sim.orientations[:, :, 0] = 1.0
    move = RotationRigidMove(step_size=0.5)
    sim.add_move(move)
    sim.run(blocks=10)
    assert _is_unit(sim.orientations)


def test_rotation_bisection_move_preserves_unit_vectors():
    sim = make_rot_sim(N=1, M=9, seed=4)
    sim.orientations[:] = 0.0
    sim.orientations[:, :, 2] = 1.0
    move = RotationBisectionMove(level=1)
    sim.add_move(move)
    sim.run(blocks=10)
    assert _is_unit(sim.orientations)

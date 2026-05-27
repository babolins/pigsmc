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

def test_free_rotor_grid_non_negative():
    """Interpolated propagator values are non-negative across x ∈ [-1, 1]."""
    grid = FreeRotorGrid(lambda_rot=0.5, L_max=50, grid_size=200)
    xs = np.linspace(-1.0, 1.0, 500)
    vals = grid.eval(xs)
    assert np.all(vals >= 0.0), f"min={vals.min()}"


def test_free_rotor_grid_converges_with_L_max():
    """Grid values converge as L_max increases."""
    x = 0.5
    small = FreeRotorGrid(lambda_rot=0.5, L_max=10, grid_size=500).eval(x)
    large = FreeRotorGrid(lambda_rot=0.5, L_max=100, grid_size=500).eval(x)
    assert abs(float(large) - float(small)) < 1e-4


def test_free_rotor_grid_log_eval():
    """log_eval returns log of G, consistent with eval."""
    grid = FreeRotorGrid(lambda_rot=0.5, L_max=50, grid_size=500)
    x = 0.3
    log_g = float(grid.log_eval(x))
    g = float(grid.eval(x))
    assert abs(np.exp(log_g) - g) < 1e-10


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
# Acceptance rate: bisection = 1.0 for free rotor (no potential)
# ---------------------------------------------------------------------------

def test_rotation_bisection_acceptance_rate_one_free_rotor():
    """Lévy bridge cancels kinetic terms → acceptance 1.0 for free rotor."""
    sim = make_rot_sim(N=1, M=9, seed=42)
    # Initialize with valid unit vectors
    sim.orientations[:] = 0.0
    sim.orientations[:, :, 2] = 1.0  # all pointing in z direction

    move = RotationBisectionMove(level=3)
    sim.add_move(move)
    sim.run(blocks=20)
    stats = sim.acceptance_stats[move]
    assert stats["attempts"] > 0
    assert stats["acceptances"] == stats["attempts"], (
        f"Expected 100% acceptance, got {stats['acceptances']}/{stats['attempts']}"
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

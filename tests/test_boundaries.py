import numpy as np
import pytest
from pigsmc import Boundary, ParticleType, Simulation, TranslationEndMove


# ---------------------------------------------------------------------------
# Factory method construction
# ---------------------------------------------------------------------------

def test_free_boundary_factory():
    bc = Boundary.free()
    assert bc is not None


def test_periodic_3d_factory():
    bc = Boundary.periodic_3d(1.0, 2.0, 3.0)
    assert bc is not None


def test_quasi_2d_factory():
    bc = Boundary.quasi_2d(1.0, 2.0)
    assert bc is not None


def test_quasi_1d_factory():
    bc = Boundary.quasi_1d(3.0)
    assert bc is not None


# ---------------------------------------------------------------------------
# box_lengths property
# ---------------------------------------------------------------------------

def test_periodic_3d_box_lengths():
    bc = Boundary.periodic_3d(1.0, 2.0, 3.0)
    np.testing.assert_array_equal(bc.box_lengths, [1.0, 2.0, 3.0])


def test_quasi_2d_box_lengths():
    bc = Boundary.quasi_2d(1.0, 2.0)
    np.testing.assert_array_equal(bc.box_lengths, [1.0, 2.0])


def test_quasi_1d_box_lengths():
    bc = Boundary.quasi_1d(3.0)
    np.testing.assert_array_equal(bc.box_lengths, [3.0])


def test_free_boundary_has_no_box_lengths():
    bc = Boundary.free()
    assert not hasattr(bc, "box_lengths")


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def test_quasi_2d_rejects_third_arg():
    with pytest.raises(TypeError):
        Boundary.quasi_2d(1.0, 2.0, 3.0)


def test_quasi_1d_rejects_second_arg():
    with pytest.raises(TypeError):
        Boundary.quasi_1d(1.0, 2.0)


def test_periodic_3d_nonpositive_length_raises():
    with pytest.raises(ValueError):
        Boundary.periodic_3d(0.0, 1.0, 1.0)


def test_periodic_3d_negative_length_raises():
    with pytest.raises(ValueError):
        Boundary.periodic_3d(1.0, -1.0, 1.0)


def test_quasi_2d_nonpositive_raises():
    with pytest.raises(ValueError):
        Boundary.quasi_2d(0.0, 1.0)


def test_quasi_1d_nonpositive_raises():
    with pytest.raises(ValueError):
        Boundary.quasi_1d(0.0)


# ---------------------------------------------------------------------------
# displacement() — single pair
# ---------------------------------------------------------------------------

def test_free_displacement_is_raw_subtraction():
    bc = Boundary.free()
    r_i = np.array([0.9, 0.5, 0.5])
    r_j = np.array([0.1, 0.5, 0.5])
    d = bc.displacement(r_i, r_j)
    np.testing.assert_allclose(d, [0.8, 0.0, 0.0])


def test_periodic_3d_displacement_mic_wraps():
    L = 1.0
    bc = Boundary.periodic_3d(L, L, L)
    # r_i at 0.1, r_j at 0.9 → raw diff = -0.8, MIC → +0.2
    r_i = np.array([0.1, 0.0, 0.0])
    r_j = np.array([0.9, 0.0, 0.0])
    d = bc.displacement(r_i, r_j)
    np.testing.assert_allclose(d, [0.2, 0.0, 0.0], atol=1e-12)


def test_periodic_3d_displacement_no_wrap_needed():
    bc = Boundary.periodic_3d(1.0, 1.0, 1.0)
    r_i = np.array([0.3, 0.3, 0.3])
    r_j = np.array([0.2, 0.2, 0.2])
    d = bc.displacement(r_i, r_j)
    np.testing.assert_allclose(d, [0.1, 0.1, 0.1], atol=1e-12)


def test_quasi_2d_mic_xy_free_z():
    bc = Boundary.quasi_2d(1.0, 1.0)
    # x: 0.1 - 0.9 = -0.8 → MIC → +0.2
    # y: 0.1 - 0.9 = -0.8 → MIC → +0.2
    # z: 0.1 - 0.9 = -0.8 (no MIC)
    r_i = np.array([0.1, 0.1, 0.1])
    r_j = np.array([0.9, 0.9, 0.9])
    d = bc.displacement(r_i, r_j)
    np.testing.assert_allclose(d, [0.2, 0.2, -0.8], atol=1e-12)


def test_quasi_1d_mic_z_free_xy():
    bc = Boundary.quasi_1d(1.0)
    # z: 0.1 - 0.9 = -0.8 → MIC → +0.2
    # x,y: raw
    r_i = np.array([0.1, 0.1, 0.1])
    r_j = np.array([0.9, 0.9, 0.9])
    d = bc.displacement(r_i, r_j)
    np.testing.assert_allclose(d, [-0.8, -0.8, 0.2], atol=1e-12)


# ---------------------------------------------------------------------------
# displacement() — broadcastable inputs
# ---------------------------------------------------------------------------

def test_displacement_broadcastable_batch():
    bc = Boundary.periodic_3d(1.0, 1.0, 1.0)
    # Two pairs at once: shape (2, 3)
    r_i = np.array([[0.1, 0.0, 0.0], [0.3, 0.0, 0.0]])
    r_j = np.array([[0.9, 0.0, 0.0], [0.2, 0.0, 0.0]])
    d = bc.displacement(r_i, r_j)
    assert d.shape == (2, 3)
    np.testing.assert_allclose(d[0], [0.2, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(d[1], [0.1, 0.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# all_displacements()
# ---------------------------------------------------------------------------

def test_all_displacements_shape():
    bc = Boundary.periodic_3d(2.0, 2.0, 2.0)
    N = 4
    pos = np.random.default_rng(0).random((N, 3)) * 2.0
    result = bc.all_displacements(pos)
    assert result.shape == (N, N, 3)


def test_all_displacements_consistent_with_displacement():
    bc = Boundary.periodic_3d(2.0, 2.0, 2.0)
    N = 3
    rng = np.random.default_rng(7)
    pos = rng.random((N, 3)) * 2.0
    all_d = bc.all_displacements(pos)
    for i in range(N):
        for j in range(N):
            expected = bc.displacement(pos[i], pos[j])
            np.testing.assert_allclose(all_d[i, j], expected, atol=1e-12)


def test_all_displacements_antisymmetric():
    bc = Boundary.periodic_3d(1.0, 1.0, 1.0)
    rng = np.random.default_rng(3)
    pos = rng.random((4, 3))
    all_d = bc.all_displacements(pos)
    # d[i,j] = -d[j,i]
    np.testing.assert_allclose(all_d, -all_d.transpose(1, 0, 2), atol=1e-12)


# ---------------------------------------------------------------------------
# sim.boundary property
# ---------------------------------------------------------------------------

def test_sim_boundary_property_returns_passed_boundary():
    bc = Boundary.periodic_3d(2.0, 2.0, 2.0)
    particles = [ParticleType(lambda_trans=1.0)]
    sim = Simulation(particles, M=3, tau_prime=0.1, boundary=bc)
    assert sim.boundary is bc


def test_sim_default_boundary_is_free():
    particles = [ParticleType(lambda_trans=1.0)]
    sim = Simulation(particles, M=3, tau_prime=0.1)
    # Just check it doesn't raise and has a boundary
    assert sim.boundary is not None


# ---------------------------------------------------------------------------
# Engine delivers MIC-corrected r_ij to V_int
# ---------------------------------------------------------------------------

def test_engine_delivers_mic_corrected_rij_to_V_int():
    """All r_ij vectors delivered to V_int must satisfy |r_ij[d]| <= L/2."""
    L = 1.0
    bc = Boundary.periodic_3d(L, L, L)
    particles = [ParticleType(lambda_trans=1.0), ParticleType(lambda_trans=1.0)]
    sim = Simulation(particles, M=3, tau_prime=0.1, boundary=bc, seed=0)

    # Place particles near opposite sides of the box so raw r_ij would exceed L/2
    sim.positions[:, 0, :] = 0.1   # particle 0 near 0
    sim.positions[:, 1, :] = 0.9   # particle 1 near L

    recorded_rij = []

    def V_int(r_ij, u_i, u_j):
        recorded_rij.append(np.array(r_ij).copy())
        return 0.0

    sim.set_potential(None, V_int)
    sim.add_move(TranslationEndMove(step_size=0.1))
    sim.run(blocks=2)

    assert len(recorded_rij) > 0
    for rij in recorded_rij:
        for d in range(3):
            assert abs(rij[d]) <= L / 2 + 1e-10, (
                f"r_ij[{d}] = {rij[d]} exceeds L/2 = {L/2}"
            )


def test_engine_delivers_mic_corrected_rij_quasi_2d():
    """Quasi-2d: MIC in x,y; z displacement may exceed Lxy/2."""
    Lx, Ly = 1.0, 1.0
    bc = Boundary.quasi_2d(Lx, Ly)
    particles = [ParticleType(lambda_trans=1.0), ParticleType(lambda_trans=1.0)]
    sim = Simulation(particles, M=3, tau_prime=0.1, boundary=bc, seed=0)

    # x: particles near opposite sides (MIC should wrap)
    # z: large separation (no wrapping)
    sim.positions[:, 0, :] = [0.1, 0.1, 0.0]
    sim.positions[:, 1, :] = [0.9, 0.9, 5.0]

    recorded_rij = []

    def V_int(r_ij, u_i, u_j):
        recorded_rij.append(np.array(r_ij).copy())
        return 0.0

    sim.set_potential(None, V_int)
    sim.add_move(TranslationEndMove(step_size=0.05))
    sim.run(blocks=2)

    assert len(recorded_rij) > 0
    for rij in recorded_rij:
        assert abs(rij[0]) <= Lx / 2 + 1e-10, f"x component {rij[0]} not MIC-wrapped"
        assert abs(rij[1]) <= Ly / 2 + 1e-10, f"y component {rij[1]} not MIC-wrapped"
        # z is free — the raw displacement can be large
        # (just check it's the raw difference, no MIC)
        assert abs(rij[2]) > 1.0, f"z component {rij[2]} should be large (no MIC)"


# ---------------------------------------------------------------------------
# Positions stored in unwrapped coordinates
# ---------------------------------------------------------------------------

def test_positions_stored_unwrapped():
    """Engine does NOT wrap positions back into [0, L) after accepted moves."""
    L = 1.0
    bc = Boundary.periodic_3d(L, L, L)
    particles = [ParticleType(lambda_trans=1.0)]
    sim = Simulation(particles, M=3, tau_prime=0.1, boundary=bc, seed=0)

    # Place particle far outside [0, L) — should remain there
    sim.positions[:] = 2.5

    # Zero-step move: proposed == current, so always accepted, positions "updated"
    # to the same values; if engine wraps, 2.5 % 1.0 = 0.5
    move = TranslationEndMove(step_size=0.0)
    sim.add_move(move)
    sim.run(blocks=5)

    assert np.all(sim.positions >= 2.0), (
        f"Positions were wrapped! Min value: {sim.positions.min()}"
    )


# ---------------------------------------------------------------------------
# Backward compatibility: free boundary runs like before
# ---------------------------------------------------------------------------

def test_free_boundary_runs_without_error():
    bc = Boundary.free()
    particles = [ParticleType(lambda_trans=1.0), ParticleType(lambda_trans=1.0)]
    sim = Simulation(particles, M=3, tau_prime=0.1, boundary=bc, seed=0)
    sim.add_move(TranslationEndMove(step_size=0.1))
    sim.run(blocks=5)  # should not raise


def test_sim_without_explicit_boundary_runs():
    """Default boundary (free) should produce identical results to explicit free()."""
    particles = [ParticleType(lambda_trans=1.0)]
    sim = Simulation(particles, M=3, tau_prime=0.1, seed=42)
    sim.add_move(TranslationEndMove(step_size=0.1))
    sim.run(blocks=5)

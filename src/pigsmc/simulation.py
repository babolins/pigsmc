from __future__ import annotations

import numpy as np

from .boundaries import Boundary
from .moves import Move, SliceKind
from .particles import ParticleType
from pigsmc._engine import Engine, FreeRotorGridC


class Simulation:
    def __init__(
        self,
        particles: list[ParticleType],
        M: int,
        tau_prime: float,
        propagator: str = "primitive",
        boundary: Boundary = None,
        rng: str = "mt19937_64",
        seed: int = 0,
        sweeps_per_block: int = 1,
        L_max: int = 100,
        grid_size: int = 1000,
    ):
        if boundary is None:
            boundary = Boundary.free()

        if M < 3:
            raise ValueError(f"M must be >= 3, got {M}")
        if (M - 1) % 2 != 0:
            raise ValueError(f"M-1 must be divisible by 2, got M={M} (M-1={M-1})")
        for p in particles:
            if p.lambda_trans is None:
                raise ValueError(
                    "All particles must have lambda_trans set; got lambda_trans=None"
                )
        has_rot = [p.lambda_rot is not None for p in particles]
        if any(has_rot) and not all(has_rot):
            raise ValueError(
                "All particles must have lambda_rot set if any do; mixed "
                "translational+rotational particles are not supported"
            )

        self._particles = particles
        self._N = len(particles)
        self._M = M
        self._tau_prime = tau_prime
        self._propagator = propagator
        self._boundary = boundary
        self._sweeps_per_block = sweeps_per_block

        _bit_generators = {"mt19937_64": np.random.MT19937, "pcg64": np.random.PCG64}
        bg_cls = _bit_generators.get(rng, np.random.MT19937)
        self._rng = np.random.Generator(bg_cls(seed))

        self.positions = np.zeros((M, self._N, 3), dtype=np.float64)
        self.orientations = np.zeros((M, self._N, 3), dtype=np.float64)

        self._buf_pos = np.zeros((M, 1, 3), dtype=np.float64)
        self._buf_ori = np.zeros((M, 1, 3), dtype=np.float64)

        self._lambda_trans = np.array(
            [p.lambda_trans for p in particles], dtype=np.float64
        )
        self._lambda_rot = np.array(
            [p.lambda_rot if p.lambda_rot is not None else 0.0 for p in particles],
            dtype=np.float64,
        )
        self._slice_kind = np.full(M, SliceKind.PHYSICAL, dtype=np.int32)

        # Build per-particle C++ rotational propagator grids (one per unique lambda_rot).
        grids: list[FreeRotorGridC] = []
        if any(has_rot):
            unique_lam_rot = set(p.lambda_rot for p in particles if p.lambda_rot is not None)
            grids_by_lam = {
                lam: FreeRotorGridC(lam, L_max=L_max, grid_size=grid_size)
                for lam in unique_lam_rot
            }
            grids = [grids_by_lam[p.lambda_rot] for p in particles]

        _box_half = np.asarray(boundary._box_half, dtype=np.float64)
        self._engine = Engine(
            self.positions, self.orientations,
            self._buf_pos, self._buf_ori,
            self._lambda_trans, self._lambda_rot,
            self._slice_kind,
            self._N, self._M, self._tau_prime,
            self._rng,
            boundary._kind, _box_half, boundary,
            grids,
        )

        self._moves: list[Move] = []
        self._move_weights: list[float] = []
        self._acceptance_data: dict[int, dict] = {}
        self._bead_acceptance_data: dict[int, dict] = {
            m: {"attempts": 0, "acceptances": 0} for m in range(M)
        }

        self._observables: dict[int, dict] = {}
        self._next_obs_id = 0
        self._block_count = 0

    # ------------------------------------------------------------------
    # Physics wiring
    # ------------------------------------------------------------------

    def set_potential(self, V_ext, V_int) -> None:
        self._engine.set_potential(
            V_ext if V_ext is not None else None,
            V_int if V_int is not None else None,
        )

    def set_trial_wavefunction(self, f=None, h=None) -> None:
        self._engine.set_trial_wavefunction(
            f if f is not None else None,
            h if h is not None else None,
        )

    # ------------------------------------------------------------------
    # Move registration
    # ------------------------------------------------------------------

    def add_move(self, move: Move, weight: float = 1.0) -> None:
        self._moves.append(move)
        self._move_weights.append(weight)
        self._acceptance_data[id(move)] = {"attempts": 0, "acceptances": 0}
        self._engine.add_move(move, weight)

    # ------------------------------------------------------------------
    # Observable registration
    # ------------------------------------------------------------------

    def add_observable(self, callback, every: int, start_after: int = 0) -> int:
        obs_id = self._next_obs_id
        self._next_obs_id += 1
        self._observables[obs_id] = {
            "callback": callback,
            "every": every,
            "start_after": start_after,
        }
        return obs_id

    def remove_observable(self, obs_id: int) -> None:
        self._observables.pop(obs_id, None)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def boundary(self) -> Boundary:
        return self._boundary

    @property
    def acceptance_stats(self) -> dict[Move, dict]:
        return {move: self._acceptance_data[id(move)] for move in self._moves}

    @property
    def bead_acceptance_stats(self) -> dict[int, dict]:
        return dict(self._bead_acceptance_data)

    @property
    def block_count(self) -> int:
        return self._block_count

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, blocks: int) -> None:
        if not self._moves:
            raise ValueError("No moves registered; call add_move() first")

        for _ in range(blocks):
            move_stats, bead_stats = self._engine.run_sweep(self._sweeps_per_block)

            for idx, move in enumerate(self._moves):
                attempts, acceptances = move_stats[idx]
                self._acceptance_data[id(move)]["attempts"] += attempts
                self._acceptance_data[id(move)]["acceptances"] += acceptances

            for m in range(self._M):
                attempts, acceptances = bead_stats[m]
                self._bead_acceptance_data[m]["attempts"] += attempts
                self._bead_acceptance_data[m]["acceptances"] += acceptances

            self._block_count += 1
            self._fire_observables()

    def _fire_observables(self) -> None:
        b = self._block_count
        for obs in list(self._observables.values()):
            if b > obs["start_after"] and (b - obs["start_after"]) % obs["every"] == 0:
                obs["callback"](b, self.positions, self.orientations)

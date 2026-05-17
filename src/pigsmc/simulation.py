from __future__ import annotations

import numpy as np

from .boundaries import Boundary
from .moves import Move, MoveResult, PathState, SliceKind
from .particles import ParticleType


class Simulation:
    def __init__(
        self,
        particles: list[ParticleType],
        M: int,
        tau_prime: float,
        propagator: str = "primitive",
        boundary: Boundary = Boundary.FREE,
        rng: str = "mt19937_64",
        seed: int = 0,
        sweeps_per_block: int = 1,
    ):
        if M < 3:
            raise ValueError(f"M must be >= 3, got {M}")
        if (M - 1) % 2 != 0:
            raise ValueError(f"M-1 must be divisible by 2, got M={M} (M-1={M-1})")
        for p in particles:
            if p.lambda_trans is None:
                raise ValueError(
                    "All particles must have lambda_trans set; got lambda_trans=None"
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
        self._slice_kind = np.full(M, SliceKind.PHYSICAL, dtype=np.int32)

        self._V_ext = None
        self._V_int = None
        self._f = None
        self._h = None

        self._moves: list[Move] = []
        self._move_weights: list[float] = []
        self._acceptance_data: dict[int, dict] = {}  # id(move) → stats
        self._bead_acceptance_data: dict[int, dict] = {  # bead index → stats
            m: {"attempts": 0, "acceptances": 0} for m in range(M)
        }

        self._observables: dict[int, dict] = {}
        self._next_obs_id = 0
        self._block_count = 0

    # ------------------------------------------------------------------
    # Physics wiring
    # ------------------------------------------------------------------

    def set_potential(self, V_ext, V_int) -> None:
        self._V_ext = V_ext
        self._V_int = V_int

    def set_trial_wavefunction(self, f=None, h=None) -> None:
        self._f = f
        self._h = h

    # ------------------------------------------------------------------
    # Move registration
    # ------------------------------------------------------------------

    def add_move(self, move: Move, weight: float = 1.0) -> None:
        self._moves.append(move)
        self._move_weights.append(weight)
        self._acceptance_data[id(move)] = {"attempts": 0, "acceptances": 0}

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

        total_w = sum(self._move_weights)
        probs = [w / total_w for w in self._move_weights]

        pos_view = self.positions.view()
        pos_view.flags.writeable = False
        ori_view = self.orientations.view()
        ori_view.flags.writeable = False

        path_state = PathState(
            positions=pos_view,
            orientations=ori_view,
            buffer_positions=self._buf_pos,
            buffer_orientations=self._buf_ori,
            N=self._N,
            M=self._M,
            tau_prime=self._tau_prime,
            lambda_trans=self._lambda_trans,
            slice_kind=self._slice_kind,
        )

        attempts_per_block = self._sweeps_per_block * self._N * self._M
        n_moves = len(self._moves)

        for _ in range(blocks):
            for _ in range(attempts_per_block):
                idx = int(self._rng.choice(n_moves, p=probs))
                move = self._moves[idx]

                result = move.propose(path_state, self._rng)
                log_a = self._log_acceptance(result)

                accepted = log_a >= 0.0 or np.log(float(self._rng.random())) < log_a

                self._acceptance_data[id(move)]["attempts"] += 1
                changed_beads = {m for (_, m) in result.changed}
                for m in changed_beads:
                    self._bead_acceptance_data[m]["attempts"] += 1
                if accepted:
                    self._acceptance_data[id(move)]["acceptances"] += 1
                    for m in changed_beads:
                        self._bead_acceptance_data[m]["acceptances"] += 1
                    for (i, m) in result.changed:
                        self.positions[m, i, :] = self._buf_pos[m, 0, :]

            self._block_count += 1
            self._fire_observables()

    def _fire_observables(self) -> None:
        b = self._block_count
        for obs in list(self._observables.values()):
            if b > obs["start_after"] and (b - obs["start_after"]) % obs["every"] == 0:
                obs["callback"](b, self.positions, self.orientations)

    # ------------------------------------------------------------------
    # Acceptance ratio (primitive propagator)
    # ------------------------------------------------------------------

    def _log_acceptance(self, result: MoveResult) -> float:
        log_a = result.log_ratio_contrib

        for (i, m) in result.changed:
            r_old = self.positions[m, i, :]
            r_new = self._buf_pos[m, 0, :]
            lam = self._lambda_trans[i]
            tau = self._tau_prime

            # Kinetic terms from the free propagator
            if m > 0:
                r_prev = self.positions[m - 1, i, :]
                d_old = r_old - r_prev
                d_new = r_new - r_prev
                log_a += (np.dot(d_old, d_old) - np.dot(d_new, d_new)) / (4.0 * lam * tau)
            if m < self._M - 1:
                r_next = self.positions[m + 1, i, :]
                d_old = r_old - r_next
                d_new = r_new - r_next
                log_a += (np.dot(d_old, d_old) - np.dot(d_new, d_new)) / (4.0 * lam * tau)

            # Potential terms
            if self._V_ext is not None or self._V_int is not None:
                factor = 0.5 if (m == 0 or m == self._M - 1) else 1.0
                u_i = self.orientations[m, i, :]
                V_old = V_new = 0.0

                if self._V_ext is not None:
                    V_old += float(self._V_ext(r_old, u_i))
                    V_new += float(self._V_ext(r_new, u_i))

                if self._V_int is not None:
                    for j in range(self._N):
                        if j != i:
                            r_j = self.positions[m, j, :]
                            u_j = self.orientations[m, j, :]
                            V_old += float(self._V_int(r_old - r_j, u_i, u_j))
                            V_new += float(self._V_int(r_new - r_j, u_i, u_j))

                log_a -= tau * factor * (V_new - V_old)

            # Trial wavefunction terms (endpoint slices only)
            if (m == 0 or m == self._M - 1) and (
                self._f is not None or self._h is not None
            ):
                u_i = self.orientations[m, i, :]
                psi_old = psi_new = 0.0

                if self._f is not None:
                    psi_old += float(self._f(r_old, u_i))
                    psi_new += float(self._f(r_new, u_i))

                if self._h is not None:
                    for j in range(self._N):
                        if j != i:
                            r_j = self.positions[m, j, :]
                            u_j = self.orientations[m, j, :]
                            psi_old += float(self._h(r_old - r_j, u_i, u_j))
                            psi_new += float(self._h(r_new - r_j, u_i, u_j))

                log_a += psi_new - psi_old

        return float(log_a)

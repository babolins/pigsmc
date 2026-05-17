from dataclasses import dataclass


@dataclass
class ParticleType:
    lambda_trans: float | None
    lambda_rot: float | None = None

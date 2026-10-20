"""Policy conventions, independent of Isaac Sim and Isaac Gym."""

from dataclasses import dataclass

JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
) + tuple(f"joint{i}" for i in range(1, 7))
DEFAULT_POS = (0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5,
               0.0, 0.3, 0.5, 0.0, 0.0, 0.0)
KP = (40.0,) * 12 + (100.0, 100.0, 100.0, 20.0, 20.0, 5.0)
KD = (1.0,) * 12 + (3.0, 3.0, 3.0, 2.0, 1.0, 0.5)
TORQUE_LIMIT = (35.278, 35.278, 44.4) * 4 + (20.0, 20.0, 15.0, 7.0, 5.0, 5.0)
FOOT_NAMES = tuple(f"{leg}_foot" for leg in ("FR", "FL", "RR", "RL"))
RANDOM_BODY_NAMES = ("base", "link1", "link3", "link5", "link6")
TARGET_TIMES = (0.02, 0.04, 0.06, 1.0)
POS_CURRICULUM = ((100.0, 2.0), (1.0, 1.0), (0.8, 0.5), (0.5, 0.1),
                  (0.4, 0.05), (0.2, 0.01), (0.1, 0.005))
ORN_CURRICULUM = ((100.0, 8.0), (1.0, 4.0), (0.8, 2.0), (0.6, 1.0), (0.2, 0.5))


@dataclass
class PPOConfig:
    rollout_steps: int = 24
    epochs: int = 32
    mini_batches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    learning_rate: float = 0.001
    entropy_coef: float = 0.01


def ordered_indices(available, requested):
    """Explicitly map policy order to simulator order; fail on missing names."""
    if len(set(available)) != len(available):
        raise ValueError("Simulator names must be unique")
    missing = set(requested) - set(available)
    if missing:
        raise ValueError(f"Robot is missing required names: {sorted(missing)}")
    return [available.index(name) for name in requested]

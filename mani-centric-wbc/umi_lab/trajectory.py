"""Original ee_pos / ee_axis_angle dataset format, with CPU storage and GPU queries."""

import pickle
from pathlib import Path

import numpy as np
import torch

from .math import axis_angle_to_quat, quat_to_matrix


class TrajectoryLibrary:
    def __init__(self, filename, dt=0.005, episode_length_s=None):
        if dt <= 0:
            raise ValueError("Trajectory sample dt must be positive")
        self.dt = dt
        max_frames = int(episode_length_s / dt) if episode_length_s is not None else None
        if max_frames is not None and max_frames < 1:
            raise ValueError("Episode must contain at least one trajectory sample")
        # Original UMI datasets are pickle files. Only load trusted datasets.
        with Path(filename).expanduser().open("rb") as stream:
            episodes = pickle.load(stream)
        if not isinstance(episodes, (list, tuple)) or not episodes:
            raise ValueError("Expected a nonempty list of trajectory dictionaries")
        positions, rotations, lengths = [], [], []
        for episode in episodes:
            pos = torch.as_tensor(np.asarray(episode["ee_pos"]), dtype=torch.float32).clone()
            aa = torch.as_tensor(np.asarray(episode["ee_axis_angle"]), dtype=torch.float32)
            if pos.ndim != 2 or pos.shape[-1] != 3 or aa.shape != pos.shape or len(pos) == 0:
                raise ValueError("ee_pos and ee_axis_angle must both have shape (T, 3), T > 0")
            if not torch.isfinite(pos).all() or not torch.isfinite(aa).all():
                raise ValueError("Trajectory contains non-finite values")
            pos, aa = pos[:max_frames], aa[:max_frames]
            # Match upstream's mean of frames 1, 2, 3, including short sequences.
            frames = torch.tensor([1, 2, 3]).clamp_max(len(pos) - 1)
            pos[:, :2] -= pos[frames, :2].mean(0)
            positions.append(pos)
            rotations.append(quat_to_matrix(axis_angle_to_quat(aa)))
            lengths.append(len(pos))
        self.lengths = torch.tensor(lengths)
        self.pos = torch.nn.utils.rnn.pad_sequence(positions, batch_first=True)
        self.rot = torch.nn.utils.rnn.pad_sequence(rotations, batch_first=True)

    def query(self, trajectory_ids, times, device):
        ids = trajectory_ids.detach().cpu()
        frames = (times.detach().cpu() / self.dt).long().clamp_min(0)
        ids = ids.reshape((-1,) + (1,) * (frames.ndim - 1)).expand_as(frames)
        frames = torch.minimum(frames, self.lengths[ids] - 1)
        return self.pos[ids, frames].to(device), self.rot[ids, frames].to(device)

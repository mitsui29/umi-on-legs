"""Use the repository's PPO, network layout and checkpoint keys."""

import torch
from torch import nn

from legged_gym.rsl_rl.algorithms.ppo import PPO
from legged_gym.rsl_rl.modules import ActorCritic

from .config import JOINT_NAMES, TARGET_TIMES, PPOConfig


def mlp(inputs, outputs):
    sizes = (inputs, 128, 64, 32, outputs)
    layers = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < 3:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def make_ppo(device, num_envs, cfg: PPOConfig):
    policy = ActorCritic(mlp(96, 18), mlp(225, 1), num_actions=18)
    alg = PPO(policy, num_learning_epochs=cfg.epochs, num_mini_batches=cfg.mini_batches,
              clip_param=0.2, gamma=cfg.gamma, lam=cfg.lam, value_loss_coef=1.0,
              entropy_coef=cfg.entropy_coef, learning_rate=cfg.learning_rate, max_grad_norm=1.0,
              use_clipped_value_loss=True, schedule="adaptive", desired_kl=0.01,
              max_lr=0.01, min_lr=0.0001, device=device)
    alg.init_storage(num_envs, cfg.rollout_steps, 96, 225, 18, 1)
    return alg


def save_checkpoint(path, alg, iteration, env_cfg, ppo_cfg):
    torch.save({"model_state_dict": alg.get_model_state_dict(),
                "optimizer_state_dict": alg.get_optimizer_state_dict(), "iter": iteration,
                "learning_rate": alg.learning_rate, "backend": "isaaclab-1.2",
                "joint_names": JOINT_NAMES, "target_times": TARGET_TIMES,
                "env_cfg": env_cfg, "ppo_cfg": vars(ppo_cfg)}, path)


def load_checkpoint(path, alg, resume=True):
    checkpoint = torch.load(path, map_location=alg.device, weights_only=False)
    if tuple(checkpoint.get("joint_names", ())) != JOINT_NAMES:
        raise ValueError("Checkpoint lacks this backend's joint-order contract; use an Isaac Lab checkpoint")
    alg.actor_critic.load_state_dict(checkpoint["model_state_dict"])
    if resume:
        alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        alg.learning_rate = checkpoint["learning_rate"]
    return checkpoint["iter"]

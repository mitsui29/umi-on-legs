"""Train, resume, evaluate or smoke-test Go2+ARX5 on Isaac Lab 1.2."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=20000)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--trajectory", type=str)
parser.add_argument("--trajectory_dt", type=float, default=0.005)
parser.add_argument("--synthetic", action="store_true", help="Explicit synthetic trajectory smoke test")
parser.add_argument("--no_randomization", action="store_true")
parser.add_argument("--no_noise", action="store_true")
parser.add_argument("--rollout_steps", type=int, default=24)
parser.add_argument("--epochs", type=int, default=32)
parser.add_argument("--mini_batches", type=int, default=4)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--play", action="store_true", help="Evaluate checkpoint and export actor.pt")
parser.add_argument("--steps", type=int, default=500, help="Number of evaluation control steps")
parser.add_argument("--save_interval", type=int, default=500)
parser.add_argument("--output", type=Path, default=Path("logs/isaaclab"))
parser.add_argument("--cache_dir", type=Path)

from omni.isaac.lab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.iterations < 1 or args.steps < 1 or args.save_interval < 1:
    parser.error("Environment, iteration, step and save counts must be positive")
if args.rollout_steps * args.num_envs < args.mini_batches or min(args.epochs, args.mini_batches, args.rollout_steps) < 1:
    parser.error("PPO needs positive counts and at least one sample per mini-batch")
if args.play and args.checkpoint is None:
    parser.error("--play requires --checkpoint")
if not args.trajectory and not args.synthetic:
    parser.error("Provide --trajectory PATH or explicitly select --synthetic")
if args.trajectory and args.synthetic:
    parser.error("Choose either --trajectory or --synthetic")

app = AppLauncher(args).app

import torch

from umi_lab.config import PPOConfig
from umi_lab.env import ReachingEnv
from umi_lab.env_cfg import make_cfg
from umi_lab.ppo import load_checkpoint, make_ppo, save_checkpoint


def main():
    torch.set_num_threads(4)
    options = dict(num_envs=args.num_envs, device=args.device, seed=args.seed,
                   trajectory_path=args.trajectory, trajectory_dt=args.trajectory_dt, synthetic=args.synthetic,
                   randomization=not args.no_randomization, observation_noise=not args.no_noise,
                   cache_dir=str(args.cache_dir.resolve()) if args.cache_dir else None)
    env = ReachingEnv(make_cfg(**options))
    cfg = PPOConfig(args.rollout_steps, args.epochs, args.mini_batches)
    alg = make_ppo(env.device, env.num_envs, cfg)
    iteration = load_checkpoint(args.checkpoint, alg, resume=not args.play) if args.checkpoint else 0
    obs, _ = env.reset()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(dict(env=options, ppo=vars(cfg)), indent=2) + "\n")
    if args.play:
        alg.actor_critic.eval()
        errors = []
        with torch.no_grad():
            for _ in range(args.steps):
                obs, reward, terminated, truncated, info = env.step(alg.actor_critic.act_inference(obs["policy"]))
                check_tensors(obs, reward)
                errors.append(float(info["log"]["tracking/position_error"]))
        actor = torch.jit.script(alg.actor_critic.actor)
        actor.save(str(args.output / "actor.pt"))
        print(json.dumps(dict(evaluation_steps=args.steps, mean_position_error=sum(errors)/len(errors),
                              resets=env.total_resets, timeouts=env.total_timeouts, exported="actor.pt")), flush=True)
        env.close()
        return
    with (args.output / "metrics.jsonl").open("a") as metrics:
        for iteration in range(iteration + 1, iteration + args.iterations + 1):
            start = time.monotonic()
            with torch.no_grad():
                for _ in range(cfg.rollout_steps):
                    action = alg.act(obs["policy"], obs["critic"])
                    obs, reward, terminated, truncated, info = env.step(action)
                    check_tensors(obs, reward)
                    alg.process_env_step(reward, terminated | truncated, {"time_outs": truncated & ~terminated})
                alg.compute_returns(obs["critic"])
            stats = alg.update(iteration)
            alg.storage.clear()
            stats.update({k: float(v) for k, v in info["log"].items()})
            stats.update(iteration=iteration, seconds=time.monotonic()-start,
                         resets=env.total_resets, timeouts=env.total_timeouts)
            line = json.dumps(stats, allow_nan=False)
            print(line, flush=True)
            metrics.write(line + "\n")
            metrics.flush()
            if iteration % args.save_interval == 0:
                save_checkpoint(args.output / f"model_{iteration}.pt", alg, iteration, options, cfg)
        save_checkpoint(args.output / "model_last.pt", alg, iteration, options, cfg)
    env.close()


def check_tensors(obs, reward):
    if not all(torch.isfinite(x).all() for x in (*obs.values(), reward)):
        raise FloatingPointError("Non-finite simulation observations or rewards")


if __name__ == "__main__":
    main()
    app.close()

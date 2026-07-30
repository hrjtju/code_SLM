"""
T15: single-instance sanity check for the Double-DQN pipeline.

The point of this script is to answer one question before any long run:

    *Can the agent overfit ONE instance?*

If it cannot beat a random policy on a single fixed instance, no amount of
hyper-parameter tuning on the full training set will help -- there is still a bug.

Usage:
    python sanity_overfit.py                          # default instance, 400 episodes
    python sanity_overfit.py --instance instances_json/ec_20-1.json --episodes 800
    python sanity_overfit.py --per                    # same check with prioritized replay
"""

import argparse
import os
import shutil
import statistics
import tempfile
import warnings

os.environ.setdefault("WANDB_MODE", "disabled")

import torch

from doubledqn_dflat import DoubleDQN, evaluate, train
from slm_model.slm_env import SingleSLMEnvParallel1D
from training_utils.utils import IntstanceAvgMeter

warnings.filterwarnings("ignore")


def build_args(train_dir: str, eval_dir: str, episodes: int, per: bool) -> argparse.Namespace:
    return argparse.Namespace(
        mask=True,
        sample=False,
        lr=2e-4,
        penalty=0.1,
        reward_scale=10.0,
        num_episodes=episodes,
        hidden_dim=128,
        gamma=0.99,
        epsilon=1.0,
        epsilon_final=0.05,
        epsilon_decay_steps=max(1, episodes * 10),
        target_update=200,
        buffer_size=20000,
        minimal_size=500,
        batch_size=64,
        max_part_type=20,
        max_orientation_num=7,
        max_batch_num=20,
        clip_grad_norm=10.0,
        per=per,
        per_alpha=0.6,
        per_beta=0.4,
        obs_norm=True,
        obs_norm_warmup=2000,
        train_dir=train_dir,
        eval_dir=eval_dir,
        eval_every=max(1, episodes // 10),
        trial_name="sanity_overfit",
    )


def random_policy_energy(instance_path: str, args, trials: int = 30) -> float:
    energies = []
    for _ in range(trials):
        env = SingleSLMEnvParallel1D(
            in_path=instance_path, phase="Test",
            max_part_type=args.max_part_type, max_batch_num=args.max_batch_num,
            max_orientation_num=args.max_orientation_num, penalty=args.penalty,
        )
        done = False
        while not done:
            action = torch.rand(env.action_dim)
            _, _, terminate, truncate, _ = env.step(action)
            done = terminate or truncate
        energies.append(env.solution.calculate_energy())
    return statistics.mean(energies)


def greedy_energy(agent: DoubleDQN, args, device: torch.device) -> float:
    meter = IntstanceAvgMeter(window_size=1)
    evaluate(agent, args, device, meter)
    values = list(meter.dict_avg().values())
    return values[0] if values else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-instance overfit sanity check")
    parser.add_argument("--instance", type=str, default="./instances_json/ec_20-1.json")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--per", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    opts = parser.parse_args()

    torch.manual_seed(opts.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workdir = tempfile.mkdtemp(prefix="slm_sanity_")

    try:
        shutil.copy(opts.instance, workdir)
        args = build_args(train_dir=workdir + "/", eval_dir=workdir, episodes=opts.episodes, per=opts.per)

        print(f"instance : {opts.instance}")
        print(f"device   : {device}   episodes: {opts.episodes}   per: {opts.per}")

        base_energy = random_policy_energy(opts.instance, args)
        print(f"[baseline] random-policy mean energy : {base_energy:.6f}")

        agent = DoubleDQN(device, args)
        energy_before = greedy_energy(agent, args, device)
        print(f"[before]   greedy energy (untrained) : {energy_before:.6f}")

        history = train(args, device, use_wandb=False, agent=agent, progress=True, save_model=False)

        losses = [x for x in history["loss"] if x == x]  # drop NaN (pre-warmup episodes)
        half = max(1, len(losses) // 4)
        loss_first, loss_last = statistics.mean(losses[:half]), statistics.mean(losses[-half:])

        energy_after = greedy_energy(agent, args, device)
        train_first = statistics.mean(history["energy"][:max(1, len(history["energy"]) // 4)])
        train_last = statistics.mean(history["energy"][-max(1, len(history["energy"]) // 4):])

        print("-" * 68)
        print(f"loss   : first quartile {loss_first:.6e} -> last quartile {loss_last:.6e}")
        print(f"train  : energy {train_first:.6f} -> {train_last:.6f}  (epsilon {history['epsilon'][0]:.2f} -> {history['epsilon'][-1]:.2f})")
        print(f"greedy : energy {energy_before:.6f} -> {energy_after:.6f}  (random baseline {base_energy:.6f})")
        print(f"fails  : {history['fail'][0]} -> {history['fail'][-1]} (invalid allocation attempts per episode)")
        if history["eval"]:
            trace = "  ".join(f"{ep}:{val:.4f}" for ep, val in history["eval"])
            print(f"greedy trace : {trace}")
        print("-" * 68)

        checks = {
            "loss decreased": loss_last < loss_first,
            "greedy beats random baseline": energy_after <= base_energy,
            "training energy improved": train_last <= train_first,
        }
        for name, ok in checks.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"  [INFO] untrained greedy energy = {energy_before:.6f} "
              f"(a random net + action mask already behaves like a heuristic, so this is not a pass/fail gate)")

        ok = all(checks.values())
        print(("ALL CHECKS PASSED" if ok else "SANITY CHECK FAILED"))
        return 0 if ok else 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

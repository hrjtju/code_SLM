"""
Ruijie He

`doubledqn_flat.py` -- legacy single-trunk variant of the flat Double-DQN agent.

T16: every learning-side fix (T1-T19, see TODO_RL_FIX.md) lives in
`doubledqn_dflat.py`. To avoid two diverging copies of the same buggy pipeline, this
module now *reuses* that implementation and only keeps the legacy network
architecture (one 683-wide trunk) so that checkpoints trained with this file can
still be loaded by `test_dqnflat.py`.

* Version: 0.1.0: original implementation
* Version: 0.2.0: reuse the fixed DoubleDQN from doubledqn_dflat
"""

import datetime
import os
import sys
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
from torch import Tensor
import wandb

from doubledqn_dflat import (  # noqa: F401  (re-exported for backwards compatibility)
    Arguments,
    ClassicalReplayBuffer,
    DoubleDQN as _DoubleDQNBase,
    PrioritizedReplayBuffer,
    UniformReplayBuffer,
    evaluate,
    parse_args,
    repack,
    state_dim,
    to_device,
    train,
)


class QNet(nn.Module):
    """Legacy architecture: a single MLP over the whole flat state."""

    version = "0.2.0"

    def __init__(self,
                 max_part: int = 20,
                 max_ori: int = 7,
                 max_batch: int = 20,
                 device: torch.device = "cpu",
                 ) -> None:
        super(QNet, self).__init__()

        self.max_part = max_part
        self.max_ori = max_ori
        self.max_batch = max_batch
        self.device = device

        # T9: 683 / 47 used to be hard-coded here; both are derived now.
        self.in_dim = state_dim(max_part, max_ori, max_batch)
        self.out_dim = max_part + max_ori + max_batch

        self.policy = nn.Sequential(
            nn.Linear(in_features=self.in_dim, out_features=512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(inplace=True),
            nn.LayerNorm(128),
            nn.Linear(in_features=128, out_features=self.out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)  # T3: always return a [B, A] tensor
        return self.policy(x)


class DoubleDQN(_DoubleDQNBase):
    """The fixed agent with the legacy trunk."""

    def build_qnet(self) -> nn.Module:
        return QNet(self.max_part, self.max_ori, self.max_batch, self.device)


if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="slmflat-doubledqn",
        name=args.trial_name,
        config={
            "name": args.trial_name,
            "arch": "flat-legacy-trunk",
            "actions_type": "part-orientation-batch, tensor (executed-action labels)",
            "criterion": "energy-diff",
            "lr": args.lr,
            "num_episodes": args.num_episodes,
            "gamma": args.gamma,
            "epsilon_start": args.epsilon,
            "epsilon_final": args.epsilon_final,
            "epsilon_decay_steps": args.epsilon_decay_steps,
            "target_update": args.target_update,
            "buffer_size": args.buffer_size,
            "minimal_size": args.minimal_size,
            "batch_size": args.batch_size,
            "device": str(device),
            "max_part_type": args.max_part_type,
            "max_orientation_num": args.max_orientation_num,
            "max_batch_num": args.max_batch_num,
            "clip_grad_norm": args.clip_grad_norm,
            "penalty": args.penalty,
            "reward_scale": args.reward_scale,
            "per": args.per,
            "obs_norm": args.obs_norm,
            "train_dir": args.train_dir,
            "eval_dir": args.eval_dir,
            "mask": args.mask,
            "time": datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S"),
        },
    )

    train(args, device, use_wandb=True, agent=DoubleDQN(device, args))

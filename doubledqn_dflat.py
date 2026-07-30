"""
Ruijie He

Reference: https://hrl.boyuai.com/chapter/2/dqn%E6%94%B9%E8%BF%9B%E7%AE%97%E6%B3%95

* Version: 0.1.0: 20250415, by Ruijie He, Double DQN with prioritized replay buffer
* Version: 0.2.0: convergence fixes, see TODO_RL_FIX.md (T1-T17)

"""


import datetime
from functools import reduce
import os
import random
import collections
from typing import List, Literal, Optional, Tuple
import warnings
import sys
import argparse

from training_utils.model_utils import (
    DQN_Log,
    RunningMeanStd,
    calculate_gradient_norm,
    init_weights_normal,
)

# Add the slm_model directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import wandb
from slm_model.slm_env import SingleSLMEnvParallel1D
from training_utils.utils import IntstanceAvgMeter


def to_device(x, device):
    try:
        x.to(device)
    except AttributeError:
        return x

    return x.to(device)


def repack(t: Tuple) -> Tensor:
    return torch.stack(t, dim=0)


def state_dim(max_part: int, max_ori: int, max_batch: int) -> int:
    """
    T9: single source of truth for the flat state layout produced by
    `SingleSLMEnvParallel1D`:
        [ batch_view (max_batch * 3) | build lwh (3) | parts state (max_part * (3 + 4*max_ori)) ]
    """
    return max_batch * 3 + 3 + max_part * (3 + 4 * max_ori)


class Arguments:
    mask: bool
    sample: bool
    lr: float
    penalty: float
    reward_scale: float
    num_episodes: int
    hidden_dim: int
    gamma: float
    epsilon: float
    epsilon_final: float
    epsilon_decay_steps: int
    target_update: int
    buffer_size: int
    minimal_size: int
    batch_size: int
    max_part_type: int
    max_orientation_num: int
    max_batch_num: int
    clip_grad_norm: float
    per: bool
    per_alpha: float
    per_beta: float
    obs_norm: bool
    obs_norm_warmup: int
    train_dir: str
    eval_dir: str
    trial_name: str


class UniformReplayBuffer:
    """
    T5: plain FIFO + uniform sampling replay buffer (the safe default).

    `sample` returns importance-sampling weights as well so that the training loop
    is agnostic to which buffer is in use.
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, next_mask, abs_td_error: float = 0.0):
        self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, len(self.buffer), size=batch_size)
        transitions = [self.buffer[i] for i in idx]
        state, action, reward, next_state, done, next_mask = zip(*transitions)

        # T1: reward / done must be stacked too, otherwise they stay 1-D and
        # broadcast against the [B, 1] Q values into a [B, B] target matrix.
        return dict(
            states=repack(state),
            actions=repack(action),
            rewards=repack(reward),
            next_states=repack(next_state),
            dones=repack(done),
            next_masks=repack(next_mask),
            weights=torch.ones(batch_size, 1),
            indices=idx,
        )

    def update_priorities(self, indices, abs_td_errors) -> None:
        return

    def size(self) -> int:
        return len(self.buffer)

    def get_priority_dist(self) -> Tuple[float, float]:
        return 0.0, 0.0


class PrioritizedReplayBuffer:
    """
    T5: standard proportional prioritized experience replay (Schaul et al. 2016).

    The previous implementation was a min-heap that *kept only* large-TD-error
    entries, sampled from them **uniformly**, never refreshed a priority and had no
    importance-sampling correction. That turns the replay distribution into a
    frozen set of early outliers and biases every gradient. This version:

      * ring buffer -> FIFO eviction (recency is handled by the ring, not by a
        random 10% purge);
      * sampling probability proportional to ``priority ** alpha``;
      * importance-sampling weights ``(N * p) ** -beta`` (max-normalised);
      * priorities of sampled transitions are written back after each update.
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, eps: float = 1e-3) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        self.buffer: List[Tuple] = []
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.pos = 0

    def add(self, state, action, reward, next_state, done, next_mask, abs_td_error: float = 0.0):
        tup = (state, action, reward, next_state, done, next_mask)
        priority = abs(float(abs_td_error)) + self.eps

        if len(self.buffer) < self.capacity:
            self.buffer.append(tup)
        else:
            self.buffer[self.pos] = tup

        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> dict:
        n = len(self.buffer)
        probs = self.priorities[:n] ** self.alpha
        probs = probs / probs.sum()

        idx = np.random.choice(n, size=batch_size, p=probs)
        transitions = [self.buffer[i] for i in idx]
        state, action, reward, next_state, done, next_mask = zip(*transitions)

        weights = (n * probs[idx]) ** (-self.beta)
        weights = weights / weights.max()

        return dict(
            states=repack(state),
            actions=repack(action),
            rewards=repack(reward),
            next_states=repack(next_state),
            dones=repack(done),
            next_masks=repack(next_mask),
            weights=torch.as_tensor(weights, dtype=torch.float32).view(-1, 1),
            indices=idx,
        )

    def update_priorities(self, indices, abs_td_errors) -> None:
        self.priorities[np.asarray(indices)] = np.abs(np.asarray(abs_td_errors, dtype=np.float64)) + self.eps

    def size(self) -> int:
        return len(self.buffer)

    def get_priority_dist(self) -> Tuple[float, float]:
        n = len(self.buffer)
        if n == 0:
            return 0.0, 0.0
        arr = self.priorities[:n]
        return float(arr.mean()), float(arr.std())


# Backwards-compatible alias (the old name was used elsewhere / in notebooks)
ClassicalReplayBuffer = UniformReplayBuffer


class QNet(nn.Module):
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

        # T9: derive every dimension from the arguments instead of hard-coding 47 /
        # `max_part * 3`, which only happened to be right when
        # max_part_type == max_batch_num == 20.
        self.V = max_batch * 3                      # flattened batch view
        self.M = 3                                  # build volume lwh
        self.P = (3 + max_ori * 4) * max_part       # parts state
        self.stride = 3 + 4 * max_ori               # per-part-type record length
        self.in_dim = self.V + self.M + self.P
        self.out_dim = max_part + max_ori + max_batch

        self.input_1 = nn.Sequential(
            nn.Linear(in_features=max_part, out_features=256),
            nn.LeakyReLU(inplace=True),
        )
        self.input_2 = nn.Sequential(
            nn.Linear(in_features=self.in_dim - max_part, out_features=256),
            nn.LeakyReLU(inplace=True),
        )

        self.policy = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(inplace=True),
            nn.LayerNorm(128),
            nn.Linear(in_features=128, out_features=self.out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        assert x.shape[-1] == self.in_dim, \
            f"state dim mismatch: got {x.shape[-1]}, expected {self.in_dim}"

        # remaining count of each part type
        a = x[:, self.V + self.M::self.stride]

        all_cols = torch.arange(x.shape[1], device=x.device)
        mask = ~torch.isin(all_cols, all_cols[self.V + self.M::self.stride])
        b = x[:, mask]

        a = self.input_1(a)
        b = self.input_2(b)

        x = torch.concat([a, b], dim=-1)
        x = x.view(x.shape[0], -1)  # Flatten the input
        dist = self.policy(x)
        return dist


class DoubleDQN:
    def __init__(self,
                 device: torch.device,
                 args: Arguments = None,
                 ) -> None:

        self.max_part = args.max_part_type
        self.max_ori = args.max_orientation_num
        self.max_batch = args.max_batch_num
        self.n_branches = 3

        self.gamma = args.gamma
        self.target_update = args.target_update
        self.clip_grad_norm = args.clip_grad_norm

        # T4: linear epsilon annealing driven by the number of environment steps.
        # `getattr` keeps legacy arg namespaces (e.g. test_dqnflat.py) working.
        self.epsilon_start = args.epsilon
        self.epsilon_final = getattr(args, "epsilon_final", 0.05)
        self.epsilon_decay_steps = max(1, getattr(args, "epsilon_decay_steps", 50000))
        self.epsilon = self.epsilon_start
        self.env_steps = 0

        self.update_count = 0
        self.device = device

        self.q_net = self.build_qnet().to(self.device)
        self.q_net.apply(init_weights_normal)

        self.target_q_net = self.build_qnet().to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.mask = args.mask
        self.sample = getattr(args, "sample", False)

        # T8: observation normalisation
        self.obs_norm = getattr(args, "obs_norm", True)
        self.obs_norm_warmup = getattr(args, "obs_norm_warmup", 10000)
        self.obs_rms = RunningMeanStd(
            state_dim(self.max_part, self.max_ori, self.max_batch), device=self.device
        )

    def load_checkpoint(self, path: str) -> None:
        """
        Load either the new checkpoint format (`{"q_net": ..., "obs_rms": ...}`) or a
        bare `state_dict` saved by the pre-fix code.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(ckpt, dict) and "q_net" in ckpt:
            self.q_net.load_state_dict(ckpt["q_net"])
            if ckpt.get("obs_rms") is not None:
                self.obs_rms.load_state_dict(ckpt["obs_rms"])
        else:
            self.q_net.load_state_dict(ckpt)
            # legacy checkpoints were trained on raw observations
            self.obs_norm = False

        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.obs_rms.frozen = True

    # ------------------------------------------------------------------ helpers

    def build_qnet(self) -> nn.Module:
        """Hook so that variants (see `doubledqn_flat.py`) can swap the architecture."""
        return QNet(self.max_part, self.max_ori, self.max_batch, self.device)

    def _prep(self, x: Tensor) -> Tensor:
        x = x.float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.obs_rms.normalize(x) if self.obs_norm else x

    def anneal_epsilon(self) -> None:
        """T4: called once per environment step."""
        self.env_steps += 1
        frac = min(1.0, self.env_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_final - self.epsilon_start)

    def _mask_q(self, q: Tensor, mask: Tensor) -> Tensor:
        """
        T18: push the Q value of infeasible actions down to a large *finite* negative
        number. A finite value (not -inf) is required because terminal transitions
        multiply the bootstrap by `(1 - done) == 0` and `-inf * 0` is NaN. Heads with
        no feasible entry at all (terminal states) are left untouched.
        """
        mask = mask.to(q.device).bool().reshape(q.shape)
        keep = torch.concat(
            [m | ~m.any(dim=-1, keepdim=True) for m in self.split_actions(mask)], dim=-1
        )
        return q.masked_fill(~keep, -1e9)

    def split_actions(self, a: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Split action tensor into part, orientation and batch part.
        """
        # a: [batch_size, MAX_PART_TYPE + MAX_ORIENTATION_NUM + MAX_BATCH_NUM]

        return (
            a.narrow(-1, 0, self.max_part),
            a.narrow(-1, self.max_part, self.max_ori),
            a.narrow(-1, self.max_part + self.max_ori, self.max_batch)
        )

    def batch_kronecker_product_flatten(self, tensors) -> Tensor:
        return reduce(lambda x, y: torch.einsum("ia,ib->iab", x.flatten(1), y.flatten(1)).flatten(-1),
                      tensors)

    # ------------------------------------------------------------------ acting

    @torch.no_grad()
    def take_action(self, state: torch.Tensor, curr_mask: torch.Tensor = None, test: bool = False) -> Tensor:
        """
        Returns a *score* vector [1, max_part + max_ori + max_batch]. The env reads it
        as a priority ordering; the executed triple comes back through `info`.

        T10: no autograd graph is built here, so the actions stored in the replay
        buffer no longer keep the whole graph alive.
        """
        action_dim = self.max_part + self.max_ori + self.max_batch

        if (not test) and np.random.random() < self.epsilon:
            # T3: keep the same [1, A] shape as the network output
            action = (torch.randn(1, action_dim).abs() + 0.01).to(self.device)
        else:
            action = self.q_net(self._prep(state))

        part_d, ori_d, batch_d = self.split_actions(action)

        # T3: mask along the feature dim, with guards for the "nothing feasible" case
        if self.mask and curr_mask is not None:
            curr_mask = curr_mask.to(action.device).bool()

            part_valid = curr_mask.any(dim=-1).any(dim=-1)                  # [P]
            if part_valid.any():
                part_d[:, ~part_valid] = -torch.inf
                part = int(part_d.argmax(dim=-1))

                ori_valid = curr_mask[part].any(dim=-1)                     # [O]
                if ori_valid.any():
                    ori_d[:, ~ori_valid] = -torch.inf
                    ori = int(ori_d.argmax(dim=-1))

                    batch_valid = curr_mask[part, ori]                      # [B]
                    if batch_valid.any():
                        batch_d[:, ~batch_valid] = -torch.inf

        if self.sample and test:
            part_d1 = torch.zeros_like(part_d)
            ori_d1 = torch.zeros_like(ori_d)
            batch_d1 = torch.zeros_like(batch_d)

            part_id = torch.distributions.Categorical(F.softmax(part_d, dim=-1)).sample().item()
            ori_id = torch.distributions.Categorical(F.softmax(ori_d, dim=-1)).sample().item()
            batch_id = torch.distributions.Categorical(F.softmax(batch_d, dim=-1)).sample().item()

            part_d1[0, part_id] = 1.0
            ori_d1[0, ori_id] = 1.0
            batch_d1[0, batch_id] = 1.0

            part_d, ori_d, batch_d = part_d1, ori_d1, batch_d1

        return torch.concat([part_d, ori_d, batch_d], dim=-1)

    # ------------------------------------------------------------------ learning

    def get_td_target(self, state, action, reward, next_state, done, next_mask=None, test=False) -> Tuple[Tensor, Tensor]:
        state, next_state = self._prep(state), self._prep(next_state)

        reward = reward.float().reshape(-1, 1)
        done = done.float().reshape(-1, 1)

        net_state_output = self.q_net(state)
        splitted_max_actions = self.split_actions(net_state_output)
        splitted_done_actions = self.split_actions(action)

        max_action_sets = self.q_net(next_state)
        target_action_sets = self.target_q_net(next_state)

        # T18: the bootstrapped target must not maximise over actions that are
        # infeasible at s' (~10 of the 47 per-head candidates on average, far more
        # near the end of an episode); otherwise the target stays optimistic and the
        # TD error never settles.
        if next_mask is not None:
            max_action_sets = self._mask_q(max_action_sets, next_mask)
            target_action_sets = self._mask_q(target_action_sets, next_mask)

        splitted_max_actions_next = self.split_actions(max_action_sets)
        splitted_max_actions_target = self.split_actions(target_action_sets)

        # T19: BDQ-style factorisation (Tavakoli et al. 2018). Each branch keeps its
        # OWN Q estimate and is regressed onto the SAME bootstrapped target. Averaging
        # the three head values into a single scalar before the MSE (the previous
        # behaviour) only constrains their mean, so the individual heads -- which are
        # what the greedy policy argmaxes over -- were left underdetermined and drifted
        # apart while the loss looked fine.
        q_values = torch.concat(
            [q.gather(1, a.max(1)[1].view(-1, 1))
             for q, a in zip(splitted_max_actions, splitted_done_actions)],
            dim=-1,
        )                                                       # [B, n_branches]

        max_next_q = sum(
            q.gather(1, a.max(1)[1].view(-1, 1))
            for q, a in zip(splitted_max_actions_target, splitted_max_actions_next)
        ) / self.n_branches                                     # [B, 1]

        q_targets = reward + self.gamma * max_next_q * (1 - done)

        # T1: guard the exact bug that was silently broadcasting [B,1] x [B] -> [B,B]
        assert q_targets.shape == reward.shape and q_values.shape[0] == reward.shape[0], \
            f"shape mismatch: q={q_values.shape}, target={q_targets.shape}, reward={reward.shape}"

        if test:
            return q_values.detach(), q_targets.detach()
        else:
            return q_values, q_targets.detach()

    def update(self, transition_dict) -> Tuple[DQN_Log, Tensor]:
        states = transition_dict["states"].float()
        actions = transition_dict["actions"].float()
        next_states = transition_dict["next_states"].float()
        rewards = transition_dict["rewards"].float().reshape(-1, 1)
        dones = transition_dict["dones"].float().reshape(-1, 1)
        next_masks = transition_dict.get("next_masks")
        weights = transition_dict.get("weights")
        weights = torch.ones_like(rewards) if weights is None else weights.float().reshape(-1, 1)

        # T8: refresh the normaliser on real training data, then freeze it
        if self.obs_norm and not self.obs_rms.frozen:
            self.obs_rms.update(states)
            if self.update_count >= self.obs_norm_warmup:
                self.obs_rms.frozen = True

        q_values, q_targets = self.get_td_target(states, actions, rewards, next_states, dones,
                                                 next_mask=next_masks)

        # T19: per-branch TD errors, averaged over branches for PER priorities
        td_errors = (q_targets - q_values).detach().abs().mean(dim=-1).reshape(-1)

        # T5: importance-sampling weighted loss (weights are all-ones for uniform replay)
        # T19: every branch is regressed onto the shared target
        dqn_loss = (weights * F.mse_loss(q_values, q_targets.expand_as(q_values),
                                         reduction="none").mean(dim=-1, keepdim=True)).mean()

        self.optimizer.zero_grad()

        dqn_loss.backward()

        clip_grad_norm_(self.q_net.parameters(), self.clip_grad_norm)
        grad_norm = calculate_gradient_norm(self.q_net)

        self.optimizer.step()

        # T7: target_update is counted in *gradient steps*; use a large value
        if (self.update_count + 1) % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.update_count += 1

        return DQN_Log(
            loss=dqn_loss.item(),
            grad_norm=grad_norm,
            q_mean=q_values.mean().item(),
            td_error=td_errors.mean().item(),
        ), td_errors


def parse_args() -> Arguments:

    parser = argparse.ArgumentParser(description="DQN training")

    parser.add_argument("--mask", action='store_true', help="Use mask for action selection")
    parser.add_argument("--sample", action='store_true', help="you shall never use this here")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--penalty", type=float, default=0.1, help="Penalty for invalid actions")
    parser.add_argument("--reward_scale", type=float, default=10.0,
                        help="Scale applied to the reward for learning only (logs stay unscaled)")
    parser.add_argument("--num_episodes", type=int, default=100000, help="Number of episodes")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon for epsilon-greedy")
    parser.add_argument("--epsilon_final", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--epsilon_decay_steps", type=int, default=50000,
                        help="Env steps over which epsilon is linearly annealed")
    parser.add_argument("--target_update", type=int, default=1000,
                        help="Target network update frequency, counted in gradient steps")
    parser.add_argument("--buffer_size", type=int, default=50000, help="Replay buffer size")
    parser.add_argument("--minimal_size", type=int, default=1000, help="Minimal size for sampling")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--max_part_type", type=int, default=20, help="Max part type")
    parser.add_argument("--max_orientation_num", type=int, default=7, help="Max orientation number")
    parser.add_argument("--max_batch_num", type=int, default=20, help="Max batch number")
    parser.add_argument("--clip_grad_norm", type=float, default=10.0, help="Gradient clipping norm")
    parser.add_argument("--per", action='store_true', help="Use prioritized replay (default: uniform)")
    parser.add_argument("--per_alpha", type=float, default=0.6, help="PER priority exponent")
    parser.add_argument("--per_beta", type=float, default=0.4, help="PER importance-sampling exponent")
    parser.add_argument("--obs_norm", action='store_true', default=True, help="Normalize observations")
    parser.add_argument("--no_obs_norm", action='store_false', dest="obs_norm")
    parser.add_argument("--obs_norm_warmup", type=int, default=10000,
                        help="Gradient steps after which observation stats are frozen")
    parser.add_argument("--train_dir", type=str, default="./instances_json/", help="Directory for training instances")
    parser.add_argument("--eval_dir", type=str, default="./instances_json/", help="Directory for testing instances")
    parser.add_argument("--eval_every", type=int, default=500, help="Evaluate every N episodes")
    parser.add_argument("--trial_name", type=str, default="None", help="Trial name for saving model")

    return parser.parse_args()


def evaluate(agent: DoubleDQN, args: Arguments, device: torch.device, test_meter: IntstanceAvgMeter) -> None:
    """
    T6: greedy evaluation. The previous version updated `state` instead of `state_`,
    so the agent kept acting on the *initial* observation for the whole episode and
    the reported per-instance energies were meaningless.
    """
    sample_backup = agent.sample
    agent.sample = False  # force greedy

    try:
        for instance_f in sorted(filter(lambda x: x.endswith(".json"), os.listdir(args.eval_dir))):
            test_env = SingleSLMEnvParallel1D(
                in_path=os.path.join(args.eval_dir, instance_f), phase="Test",
                max_part_type=args.max_part_type, max_batch_num=args.max_batch_num,
                max_orientation_num=args.max_orientation_num, penalty=args.penalty,
            )

            state_ = test_env.curr_state
            done = False

            while not done:
                action = agent.take_action(state_.to(device), test_env.mask_tensor.to(device), test=True)
                next_state, _, terminate, truncate, _ = test_env.step(action)
                done = terminate or truncate

                state_ = next_state  # T6: was `state = next_state`

            test_meter.update(instance_f, test_env.solution.calculate_energy())
    finally:
        agent.sample = sample_backup


def train(args: Arguments,
          device: torch.device,
          use_wandb: bool = True,
          agent: Optional["DoubleDQN"] = None,
          progress: bool = True,
          save_model: bool = True) -> dict:
    """
    Shared training loop, used both by `__main__` and by `sanity_overfit.py`
    so that the sanity check exercises exactly the same code path.

    Returns a history dict (per-episode return / energy / loss) for assertions.
    """

    num_episodes = args.num_episodes
    minimal_size = args.minimal_size
    batch_size = args.batch_size

    MAX_PART_TYPE = args.max_part_type
    MAX_ORIENTATION_NUM = args.max_orientation_num
    MAX_BATCH_NUM = args.max_batch_num

    trial_name = args.trial_name

    replay_buffer = (
        PrioritizedReplayBuffer(args.buffer_size, alpha=args.per_alpha, beta=args.per_beta)
        if args.per else UniformReplayBuffer(args.buffer_size)
    )

    agent = DoubleDQN(device, args) if agent is None else agent

    return_meter = IntstanceAvgMeter(window_size=200)
    energy_meter = IntstanceAvgMeter(window_size=200)
    loss_meter = IntstanceAvgMeter(window_size=2000)
    grad_norm_meter = IntstanceAvgMeter(window_size=2000)
    q_meter = IntstanceAvgMeter(window_size=2000)
    fail_meter = IntstanceAvgMeter(window_size=200)

    test_meter = IntstanceAvgMeter(window_size=10)

    history = {"return": [], "energy": [], "loss": [], "grad_norm": [], "epsilon": [], "fail": [],
               "eval": []}

    env = SingleSLMEnvParallel1D(in_path=args.train_dir, phase="Train",
                                 max_part_type=MAX_PART_TYPE, max_batch_num=MAX_BATCH_NUM,
                                 max_orientation_num=MAX_ORIENTATION_NUM,
                                 penalty=args.penalty)

    outer = 10 if num_episodes >= 10 else 1
    inner = max(1, int(num_episodes / outer))

    for i in range(outer):
        with tqdm(total=inner, desc=f"Iteration {i}", leave=False, position=0, disable=not progress) as pbar:
            for i_episode in range(inner):
                episode_return = 0.0
                episode_fails = 0
                episode_losses = []
                state, instance = env.reset()
                done = False

                while not done:

                    action = agent.take_action(state.to(device), env.mask_tensor.to(device), test=False)
                    next_state, reward, terminate, truncate, info = env.step(action)
                    done = terminate or truncate

                    agent.anneal_epsilon()  # T4
                    episode_fails += info.get("fail_allocate_num", 0)

                    # T2: learn on the action the environment *executed*, not on the
                    # argmax of the score vector (they differed ~40% of the time).
                    executed = info.get("executed_action")

                    if executed is not None:
                        stored_action = env.onehot_action(executed)
                        next_mask = env.head_masks().float()  # T18: feasibility at s'
                        scaled_reward = reward * args.reward_scale  # T14

                        # T12: only a real terminal state stops the bootstrap;
                        # `truncate` (dead end) must not be treated as terminal.
                        done_flag = float(terminate)

                        with torch.no_grad():  # T10
                            q_value, q_target = agent.get_td_target(
                                state.to(device).reshape(1, -1),
                                stored_action.to(device).reshape(1, -1),
                                torch.tensor([[scaled_reward]], device=device),
                                next_state.to(device).reshape(1, -1),
                                torch.tensor([[done_flag]], device=device),
                                next_mask=next_mask.to(device).reshape(1, -1),
                                test=True,
                            )
                        td_err = torch.abs(q_value - q_target).mean().item()

                        replay_buffer.add(
                            state.detach().cpu().float().reshape(-1),
                            stored_action.detach().cpu().float().reshape(-1),
                            torch.tensor([scaled_reward], dtype=torch.float32),
                            next_state.detach().cpu().float().reshape(-1),
                            torch.tensor([done_flag], dtype=torch.float32),
                            next_mask.detach().cpu().reshape(-1),
                            td_err,
                        )

                    state = next_state
                    episode_return += reward  # unscaled, for logging

                    if replay_buffer.size() > minimal_size:
                        batch = replay_buffer.sample(batch_size)
                        tmp_log, td_abs = agent.update(
                            transition_dict={
                                k: to_device(v, device)
                                for k, v in batch.items() if k != "indices"
                            }
                        )
                        replay_buffer.update_priorities(batch["indices"], td_abs.cpu().numpy())

                        loss_meter.update(instance, tmp_log.loss)
                        grad_norm_meter.update(instance, tmp_log.grad_norm)
                        q_meter.update(instance, tmp_log.q_mean)
                        episode_losses.append(tmp_log.loss)

                episode_energy = env.solution.calculate_energy()

                return_meter.update(instance, episode_return)
                energy_meter.update(instance, episode_energy)
                fail_meter.update(instance, episode_fails)

                history["return"].append(episode_return)
                history["energy"].append(episode_energy)
                history["loss"].append(sum(episode_losses) / len(episode_losses) if episode_losses else float("nan"))
                history["grad_norm"].append(grad_norm_meter.all_avg())
                history["epsilon"].append(agent.epsilon)
                history["fail"].append(episode_fails)

                episode_id = int(inner * i + i_episode + 1)

                pbar.set_postfix({
                    "episode": f"{episode_id:4d}",
                    "eps": f"{agent.epsilon:.3f}",
                    "return": f"{f'{return_meter.all_avg():.6e}':12s}",
                    "energy": f"{f'{energy_meter.all_avg():.6e}':12s}",
                    "loss": f"{f'{loss_meter.all_avg():.6e}':12s}",
                    "grad_norm": f"{f'{grad_norm_meter.all_avg():.6e}':12s}",
                })

                td_err_mean, td_err_std = replay_buffer.get_priority_dist()

                log_dict = {
                    "AvgEnergy/mean": energy_meter.all_avg(),
                    "AvgEnergy/max": energy_meter.all_max_avg(),
                    "AvgEnergy/min": energy_meter.all_min_avg(),
                    "AvgReturn/mean": return_meter.all_avg(),
                    "AvgReturn/max": return_meter.all_max_avg(),
                    "AvgReturn/min": return_meter.all_min_avg(),
                    "epsilon": agent.epsilon,
                    "DQN/loss": loss_meter.all_avg(),
                    "DQN/grad_norm": grad_norm_meter.all_avg(),
                    "DQN/q_mean": q_meter.all_avg(),
                    "DQN/fail_allocate_num": fail_meter.all_avg(),
                    "ReplayBuffer/size": replay_buffer.size(),
                    "ReplayBuffer/TD_Error_Mean": td_err_mean,
                    "ReplayBuffer/TD_Error_Std": td_err_std,
                }

                if getattr(args, "eval_every", 0) > 0 and i_episode % args.eval_every == 0:
                    evaluate(agent, args, device, test_meter)
                    eval_dict = test_meter.dict_avg("Instances/")
                    log_dict.update(eval_dict)
                    history["eval"].append((episode_id, test_meter.all_avg()))

                # T6: a single log call per episode with an explicit step
                if use_wandb:
                    wandb.log(log_dict, step=episode_id)

                pbar.update(1)

        if save_model:
            now_str = datetime.datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
            os.makedirs("./model_params", exist_ok=True)
            torch.save(
                {
                    "q_net": agent.q_net.state_dict(),
                    "obs_rms": agent.obs_rms.state_dict(),
                    "args": vars(args) if not isinstance(args, dict) else args,
                },
                f"./model_params/{trial_name}_{i}_{now_str}.pt",
            )

    history["agent"] = agent
    return history


if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="slmflat-doubledqn",
        config={
            "name": args.trial_name,
            "actions_type": "part-orientation-batch, tensor (executed-action labels)",
            "criterion": "energy-diff",
            "lr": args.lr,
            "num_episodes": args.num_episodes,
            "hidden_dim": args.hidden_dim,
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
            "time": datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")
        },
    )

    train(args, device, use_wandb=True)

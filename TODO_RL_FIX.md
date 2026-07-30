# RL 不收敛修复清单 (TODO_RL_FIX)

主目标文件：`doubledqn_dflat.py`（当前主训练脚本）、`slm_model/slm_env.py`
连带修复：`doubledqn_flat.py`（同源 sibling，含相同 P0 bug）
**不在范围内**：`doubledqn.py` / `dqn.py`（旧的 2D-view 版本，`q_values` 已 reshape 成 `[B]`，无广播 bug，属 legacy）

诊断依据均已在 `.venv` 中实测复现，见每项的「症状」。

---

## P0 — 必须修，否则数学上不可能收敛

- [x] **T1 修 TD 目标广播错误**
  - 位置：`doubledqn_dflat.py:139` (`PrioritizedReplayBuffer.sample`)、`:343-345`、`:333`
  - 症状：`rewards`/`dones` 为 `[B]`，`q_values` 为 `[B,1]` → `q_targets` 变成 `[B,B]`，MSE 变 B×B 全交叉回归，Q 只能收敛到 batch 均值。
  - 做法：`sample()` 对 reward/done 也 `repack`；`update()` 里统一 `.float().view(-1,1)`；加 shape 断言。

- [x] **T2 修「训练用 action 标签 ≠ 环境实际执行 action」**
  - 位置：`slm_model/slm_env.py: SingleSLMEnvParallel1D.step`、`doubledqn_dflat.py` 训练主循环
  - 症状：实测 40% 的 step，环境按优先级回退执行的 `(part, ori, batch)` 与 argmax 三元组不同，Q 更新打在未执行的动作上。
  - 做法：env 记录 `self.executed_action`，`step` 通过 `info` 返回；新增 `env.onehot_action()`；replay buffer 存**执行动作的 one-hot**（argmax 即执行动作），env 仍保留"优先级排序"语义。

- [x] **T3 修 `--mask` 分支崩溃（掩码从未生效）**
  - 位置：`doubledqn_dflat.py:271-283`
  - 症状：`IndexError: mask [20] vs indexed tensor [1,20]`；且随机分支输出 `[47]`、贪心分支输出 `[1,47]`，两分支 shape 不一致。
  - 做法：两分支统一为 `[1, A]`；掩码沿 `dim=-1` 赋 `-inf`；对"全不可行"加 `.any()` 保护。

## P1 — 严重影响收敛质量

- [x] **T4 epsilon 退火失效**
  - 位置：`doubledqn_dflat.py:271-273`
  - 症状：衰减嵌在探索分支内且再套一层 `random()<eps`，因子 0.99999 → 实测 10 万步后 eps≈0.41，全程一半动作是噪声。
  - 做法：改为按 env step 计数的线性退火 `epsilon_start → epsilon_final`，新增 `--epsilon_final` / `--epsilon_decay_steps`。

- [x] **T5 重写 replay buffer**
  - 位置：`doubledqn_dflat.py:79-176`
  - 症状：(a) `self.num += 1` 重复两次；(b) 只保留大 TD-error 且优先级永不更新 + **均匀采样、无 IS 权重** → 有偏梯度、训练集被早期离群样本占据；(c) 每次 add 有 10% 概率全量 sort+heapify；(d) `ClassicalReplayBuffer` 重复定义 `sample`/`size`，第二份会崩。
  - 做法：`PrioritizedReplayBuffer` 改为标准 proportional PER（ring buffer + FIFO 淘汰 + `p^alpha` 采样 + IS 权重 + 采样后更新优先级）；清理 `ClassicalReplayBuffer`；新增 `--per` 开关，**默认走均匀 buffer**。

- [x] **T6 修评估循环变量名 bug**
  - 位置：`doubledqn_dflat.py:556`（及 `doubledqn_flat.py:531`）
  - 症状：`state = next_state` 应为 `state_ = next_state` → 评估时 agent 永远只看初始状态，`Instances/*` 曲线无意义。
  - 做法：改名；评估强制 greedy（不走 softmax 采样）；测试指标与训练指标用同一个 `step=episode_id`。

## P2 — 超参 / 数值尺度 / 工程健壮性

- [x] **T7 超参默认值**：`lr 0.01→2e-4`、`gamma 1.00→0.99`、`target_update 5→1000`（按梯度步）、`epsilon 0.5→1.0` + 退火。
- [x] **T8 状态归一化**：state 原始值最大 1e5（`rest_area=61504` 等）且首层无归一化 → 加 per-feature `RunningMeanStd`（clamp ±10，warmup 后冻结，`--obs_norm` / `--obs_norm_warmup`）。
- [x] **T9 QNet 硬编码维度**：`out_features=47` 与 `self.V = max_part*3`（应为 `max_batch*3`）仅在 20==20 时偶然成立 → 全部由参数推导 + 输入维度断言。
- [x] **T10 `take_action` / 单步 td_err 缺 `no_grad`**：带 `grad_fn` 的 action 进 buffer → 计算图被长期持有，内存泄漏。
- [x] **T11 `env.reset()` 重置全局随机种子**：`random.seed(random.random())` 每 episode 执行，与 buffer 的 `random.sample` 共用流 → 采样模式结构性重复。做法：`seed=None` 时不 seed，`reset()` 不再重播种。
- [x] **T12 truncation 当成 terminal**：装不下(truncated)时不 bootstrap，人为压低价值 → buffer 的 done 只存 `terminated`，`truncated` 仅用于跳出循环。
- [x] **T13 三分支 Q 用 sum 聚合**：改为 mean（branching-DQN 惯例），target 侧同步改，保持 Bellman 一致、缩小 TD 误差尺度。
- [x] **T14 reward 尺度**：单步 reward ~0.05、return ~-2.9，与 Q 初始尺度不匹配 → 新增 `--reward_scale`（默认 10.0），仅用于学习，日志仍记录原始 return/energy。

## 验证

- [x] **T15 新增 sanity check 脚本** `sanity_overfit.py`：在**单个固定 instance**上过拟合，检查 loss 单调下降、energy 优于随机策略。
- [x] **T16 把 T1/T4/T6/T10 同步到 `doubledqn_flat.py`**
- [x] **T17 回归测试**：`test_env.py` / 现有测试仍可跑；`--mask` 开启后 `fail_allocate_num → 0`。

## P0（第二轮）— 由 sanity check 暴露出的新问题

第一轮 T1~T14 修完后 `sanity_overfit.py` 仍 FAIL（loss 上升、greedy energy 随训练**变差** 0.554→0.712），
继续定位出下面两个结构性错误，修完后三项检查全部 PASS。

- [x] **T18 bootstrap 目标未屏蔽 `s'` 的不可行动作**
  - 位置：`doubledqn_dflat.py: get_td_target` / `slm_model/slm_env.py: head_masks()`
  - 症状：实测 `s'` 平均有 **10.5 / 47** 个 per-head 候选不可行（episode 末尾更多），`max_a Q(s',a)` 会挑到这些不可执行动作 → 目标系统性偏乐观，TD 误差不收敛。
  - 做法：env 新增 `head_masks()`（把 `mask_tensor` 边缘化成 `[P|O|B]`）；buffer 存 `next_mask`；target 侧用 `_mask_q()` 把不可行动作压到 `-1e9`（**有限值**，否则 terminal 的 `(1-done)=0` 会算出 `-inf*0=NaN`），整头全不可行时不屏蔽。

- [x] **T19 三个 head 只有「均值」被回归，per-head Q 欠定**
  - 位置：`doubledqn_dflat.py: get_td_target` / `update`
  - 症状：`Q(s,a) = mean(Q_p, Q_o, Q_b)` 后再做 MSE，只约束了三者之和；而策略是**逐 head argmax**，所以各 head 可以任意漂移而 loss 照样下降 → 学得越久 greedy 策略越差（实测 greedy energy 单调恶化到 0.712）。
  - 做法：改成标准 BDQ（Tavakoli et al. 2018）：`q_values` 保持 `[B, 3]`，三个 head **各自**回归到同一个共享目标 `y = r + γ·mean_i max_a Q_i^-(s',a)·(1-done)`；PER 优先级用 per-branch TD 误差的均值。

- [x] **T20 `max_*` 配置小于 instance 维度时报错信息不可读**
  - 位置：`slm_model/slm_env.py: init_mask` / `update_mask` / 新增 `state_dim` + `check_state_dim`
  - 症状：`--max_orientation_num 5`、`--max_part_type 15` 会在很远的地方抛 `RuntimeError: The size of tensor a (7) must match ... b (5)` 或 `stack expects each tensor to be equal size, but got [31] and [23]`。
  - 做法：`init_mask` / `update_mask` / `reset` 处加断言，直接指出是哪个 `max_*` 太小、哪个 instance。

---

## 执行记录

全部实测于 `.venv`（macOS / CPU），分支 `fix/dqn-convergence`。

| 项 | 状态 | 备注（实测证据） |
|---|---|---|
| T1 | ✅ | `sample()` 统一 `repack` 全部字段并返回 dict；`get_td_target` 内加 shape 断言。旧 `[B,B]` UserWarning 消失 |
| T2 | ✅ | `env.step` 返回 `info["executed_action"]`，buffer 存 `env.onehot_action()`；onehot roundtrip mismatch **0/160** |
| T3 | ✅ | 探索/贪心两分支统一 `[1,47]`；`--mask` 可正常跑，`fail_allocate_num` 为 0 |
| T4 | ✅ | 按 env step 线性退火，实测 `eps 1.00 → 0.05`（到达退火步数后精确等于 `epsilon_final`） |
| T5 | ✅ | 新 `UniformReplayBuffer`（默认）+ 标准 proportional `PrioritizedReplayBuffer`（ring buffer / `p^α` / IS 权重 / 采样后回写优先级）；`--per` 路径已跑通 sanity check |
| T6 | ✅ | `state_ = next_state`；评估强制 greedy（临时关掉 softmax 采样）；每 episode 单次 `wandb.log(step=episode_id)`。同样修了 `test_dqnflat.py` |
| T7 | ✅ | `lr 2e-4` / `gamma 0.99` / `target_update 1000`（梯度步）/ `epsilon 1.0` |
| T8 | ✅ | `RunningMeanStd`（per-feature、clamp ±10、warmup 后冻结）；checkpoint 一并保存 `obs_rms` |
| T9 | ✅ | 维度全部由参数推导；`--max_part_type 25 --max_batch_num 12` 现在可跑（改前 47 / `max_part*3` 硬编码必崩） |
| T10 | ✅ | `take_action` 加 `@torch.no_grad()`，单步 td_err 用 `no_grad`；实测 `action.requires_grad == False` |
| T11 | ✅ | `seed=None` 时不再调 `random.seed()`，`reset()` 不重播种 |
| T12 | ✅ | buffer 的 done 只存 `terminated`；装不下时置 `truncated=True`，仅用于跳出循环 |
| T13 | ✅ | 被 T19 取代：不再把三 head 聚合成标量（sum / mean 都不对），改为 per-branch 回归 |
| T14 | ✅ | `--reward_scale`（默认 10.0）只作用于学习，日志的 return / energy 仍是原始值 |
| T15 | ✅ | `sanity_overfit.py`：600 ep 三项检查全 PASS；2500 ep：loss `3.0e-2 → 9.5e-3`，greedy energy `0.639 → 0.550`（随机策略基线 1.049，未训练 masked-greedy 0.554） |
| T16 | ✅ | `doubledqn_flat.py` 改为复用修好的 `DoubleDQN`（`build_qnet()` 钩子保留 legacy 单干网络，旧 checkpoint 仍可加载）；`agent.load_checkpoint()` 兼容新旧两种格式，均实测通过 |
| T17 | ✅ | `doubledqn_dflat.py` / `doubledqn_flat.py` / `test_dqnflat.py`（32 instance greedy 全跑完）/ `doubledqn.py` / `dqn.py` 均正常；`--mask` 下 `fail_allocate_num = 0`。`test_env.py` 在 HEAD 上就已坏（`ModuleNotFoundError: slm_classes`），与本次改动无关 |
| T18 | ✅ | `head_masks()` + `_mask_q()`；实测 `s'` 平均 10.5/47 个 per-head 候选不可行，现已被屏蔽；terminal（整头不可行）不产生 NaN，`torch.isfinite(td).all() == True` |
| T19 | ✅ | 改为 BDQ per-branch 回归后，loss 才真正随训练下降、greedy 策略随训练变好（改前 greedy energy 单调恶化到 0.712） |
| T20 | ✅ | `max_*` 过小时给出可读断言：`state length 504 != expected 384 for max_part_type=15, ...` |

## 复现命令

```bash
source .venv/bin/activate

# 单 instance 过拟合 sanity check（先跑这个，600 ep 约 1 min）
python sanity_overfit.py --episodes 600
python sanity_overfit.py --episodes 2500          # 更长：greedy energy 0.639 → 0.550
python sanity_overfit.py --episodes 400 --per     # PER 路径

# 正式训练（默认 uniform buffer + 掩码 + 观测归一化）
python doubledqn_dflat.py --mask --trial_name fix_v2
```

## 遗留 / 后续建议（不属于本次 bug 修复）

1. 未训练的「随机网络 + 掩码 argmax」本身就相当于一个不错的启发式（energy 0.554），要明显超过它需要更长训练 / 调参，不是 bug。
2. 掩码只做面积可行性（必要非充分），2D 装箱仍可能失败并回退，因此 `fail_allocate_num` 偶尔 > 0；若要严格掩码，需在 mask 内做真实装箱试探。
3. `doubledqn.py` / `dqn.py` 仍是 legacy，未同步这些修复。
4. 可选进一步改进：dueling head、n-step return、soft target update（τ=0.005）、对批次编号置换不变的编码（当前 batch head 对批次编号敏感）。

"""实验监控：从 wandb 拉取当前 3 个 run 的最新指标。用法: python watch_runs.py [run_name_prefix]"""
import sys
import wandb

prefix = sys.argv[1] if len(sys.argv) > 1 else "fix_v2_"
api = wandb.Api()
runs = [r for r in api.runs("gravitastju-/slmflat-doubledqn") if (r.name or "").startswith(prefix)]

hdr = f"{'run':22s} {'state':9s} {'ep':>6s} {'train_E':>8s} {'eval_E':>8s} {'best_E':>8s} {'loss':>10s} {'q_mean':>8s} {'eps':>6s} {'fail':>5s}"
print(hdr); print("-" * len(hdr))
for r in sorted(runs, key=lambda x: x.name):
    s = r.summary
    g = lambda k, p=4: (f"{s[k]:.{p}f}" if isinstance(s.get(k), (int, float)) else "-")
    print(f"{r.name[:22]:22s} {r.state:9s} {str(s.get('_step','-')):>6s} "
          f"{g('AvgEnergy/mean'):>8s} {g('Eval/energy'):>8s} {g('Eval/best_energy'):>8s} "
          f"{g('DQN/loss',6):>10s} {g('DQN/q_mean',3):>8s} {g('epsilon',3):>6s} {g('DQN/fail_allocate_num',2):>5s}")

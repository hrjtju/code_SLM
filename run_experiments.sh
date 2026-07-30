#!/usr/bin/env bash
# 修复后 (T1-T20) 的正式实验：主配置 + 两个消融。
# 用法： ./run_experiments.sh [num_episodes]      (默认 20000)
set -u

cd "$(dirname "$0")"
source .venv/bin/activate

EP=${1:-20000}
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs/${STAMP}"
mkdir -p "$LOGDIR"

# 3 个 run 并行跑在 8 核机器上，限制每个 run 的线程数避免互相抢核
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

COMMON="--mask --num_episodes ${EP} --epsilon_decay_steps 200000 --buffer_size 100000 \
        --eval_every 1000 --train_dir ./instances_json/ --eval_dir ./instances_json/"

launch () {              # launch <trial_name> <extra args...>
    local name=$1; shift
    echo "[launch] ${name} $*"
    nohup python -u doubledqn_dflat.py $COMMON --trial_name "${name}" "$@" \
        > "${LOGDIR}/${name}.log" 2>&1 &
    echo "$!" > "${LOGDIR}/${name}.pid"
}

launch "fix_v2_base_${STAMP}"                          # A: 主配置（uniform buffer + obs_norm）
launch "fix_v2_per_${STAMP}"        --per              # B: 消融 —— 优先经验回放
launch "fix_v2_noobsnorm_${STAMP}"  --no_obs_norm      # C: 消融 —— 关掉观测归一化

echo
echo "logs   : ${LOGDIR}"
echo "watch  : tail -f ${LOGDIR}/*.log"
echo "stop   : kill \$(cat ${LOGDIR}/*.pid)"
wait

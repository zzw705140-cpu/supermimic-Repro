#!/usr/bin/env bash
# SuperMimic 训练入口, 修改以下配置后通过一行命令启动.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ------------------------- 任务与环境 ------------------------- #
TASK_NAME="tracking"                       # 实验名称中的任务标识
MOTION_DIR=""                              # 动作数据目录, 留空使用项目默认目录
DEVICE="cuda:0"                            # 仿真与网络设备
REPLAY_DEVICE="cuda:0"                     # 长期经验池设备, 可改为 cpu
EVAL_DEVICE="cpu"                          # 独立评估设备, 可改为另一张 GPU
NUM_ENVS=4096                                    # 并行仿真环境数  4096
CONTROL_DT=0.02                            # 控制周期, 秒
MAX_EPISODE_STEPS=1000                     # 一个训练回合最多控制步数
SHOW_VIEWER=false                               # 训练是否显示 MuJoCo 窗口
EVAL_VIEWER=false                               # 周期评估是否显示窗口
SEED=0

# ------------------------- 数据采集与经验池 ------------------------- #
REPLAY_STEPS=1000                          # 每个环境保存的最大时间步数
WARMUP_STEPS=64                            # 启动时先收集的数据步数, 不单独训练世界模型
UTD=0.4                                    # 每次训练调度对应的更新/采集步数比
ACTION_NOISE=0.1                           # 真实环境动作的高斯探索噪声标准差

# ------------------------- 世界模型、策略与 Critic ------------------------- #
WM_HIDDEN="2048,2048,2048"                 # 世界模型各隐藏层宽度
POLICY_HIDDEN="2048,1024,512,256"          # 策略编码器与解码器隐藏层宽度
CRITIC_HIDDEN="256,128"                    # Critic 隐藏层宽度
HISTORY_LENGTH=10                          # 策略观测历史帧数
GOAL_FRAMES=10                             # 未来参考目标帧数
LATENT_DIM=32                              # 参考目标潜变量维度
WM_WINDOW=8                                # 世界模型训练窗口长度
HORIZON=32                                 # 每条短期预测轨迹长度
WM_UPDATES=10                              # 每轮世界模型更新次数
WM_BATCH=2048                                  # 世界模型每次采样的窗口数  2048  32
POLICY_BATCH=4096                              # 每轮短期预测轨迹数量  4096  32
LEARNING_RATE=0.0002                       # 三个网络的初始学习率
GAMMA=0.99                                 # 奖励折扣系数
GAE_LAMBDA=0.95                            # Critic 回报目标的 lambda
CRITIC_EPOCHS=5                            # 同一批 Critic 数据的训练次数
CRITIC_MINIBATCHES=4                       # 每次遍历划分的批次数
LR_DECAY_ITERATIONS=250000                 # 学习率衰减轮数

# ------------------------- 训练、评估与保存 ------------------------- #
ITERATIONS=16000                                # 目标训练轮数 12700
CHECKPOINT_ROOT="checkpoints/supermimic"   # 新实验目录的根目录
CHECKPOINT_EVERY=2000                           # 保存模型参数的轮数间隔
CHECKPOINT_KEEP=2                               # 仅保留最近几个 checkpoint, 必须为正整数
EVAL_EVERY=4000                                 # 评估当前策略的轮数间隔
LOG_EVERY=100                                   # W&B 指标记录间隔
MEMORY_EVERY=100                                # 显存检查间隔
MEMORY_LIMIT=0.9                                # 整卡占用比例上限, 0 表示仅记录
WANDB=true                                 # 是否启用在线 W&B
WANDB_PROJECT="supermimic"                 # 项目名称, 账户使用本机登录信息

FLAGS=()
[[ "$SHOW_VIEWER" == true ]] && FLAGS+=(--show-viewer)
[[ "$EVAL_VIEWER" == true ]] && FLAGS+=(--eval-viewer)
[[ "$WANDB" == true ]] && FLAGS+=(--wandb)

exec python -m supermimic.workflows.train \
    --task="$TASK_NAME" --motion-dir="$MOTION_DIR" --device="$DEVICE" --replay-device="$REPLAY_DEVICE" \
    --eval-device="$EVAL_DEVICE" --num-envs="$NUM_ENVS" --control-dt="$CONTROL_DT" --max-episode-steps="$MAX_EPISODE_STEPS" --seed="$SEED" \
    --replay-steps="$REPLAY_STEPS" --warmup-steps="$WARMUP_STEPS" --utd="$UTD" --action-noise="$ACTION_NOISE" \
    --wm-hidden="$WM_HIDDEN" --policy-hidden="$POLICY_HIDDEN" --critic-hidden="$CRITIC_HIDDEN" \
    --history-length="$HISTORY_LENGTH" --goal-frames="$GOAL_FRAMES" --latent-dim="$LATENT_DIM" --wm-window="$WM_WINDOW" --horizon="$HORIZON" \
    --wm-updates="$WM_UPDATES" --wm-batch="$WM_BATCH" --policy-batch="$POLICY_BATCH" --learning-rate="$LEARNING_RATE" \
    --gamma="$GAMMA" --gae-lambda="$GAE_LAMBDA" --critic-epochs="$CRITIC_EPOCHS" --critic-minibatches="$CRITIC_MINIBATCHES" --lr-decay-iterations="$LR_DECAY_ITERATIONS" \
    --iterations="$ITERATIONS" --checkpoint-root="$CHECKPOINT_ROOT" --checkpoint-every="$CHECKPOINT_EVERY" --checkpoint-keep="$CHECKPOINT_KEEP" --eval-every="$EVAL_EVERY" --log-every="$LOG_EVERY" \
    --memory-every="$MEMORY_EVERY" --memory-limit="$MEMORY_LIMIT" --wandb-project="$WANDB_PROJECT" "${FLAGS[@]}"

#!/usr/bin/env bash
# SuperMimic 独立评估入口: 修改本文件中的配置后, 用 bash 启动.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ------------------------- 模型来源 ------------------------- #

# 指定 checkpoint 文件, 留空时自动查找
CHECKPOINT_PATH="checkpoints/supermimic/tracking__supermimic__20261112_041550/epoch_004000.ckpt"
RUN_DIR=""                                   # 指定实验目录, 与 CHECKPOINT_PATH 二选一
CHECKPOINT_ROOT="checkpoints/supermimic"     # 自动查找实验的根目录
TASK_NAME="tracking"                         # 自动查找实验时使用的任务名称

# ------------------------- 评估设置 ------------------------- #
MOTION_DIR=""                                # 动作数据目录, 留空使用项目默认数据
DEVICE="cuda:0"                              # 仿真与策略设备, 也可设置为 cpu
SHOW_VIEWER=true                             # true 时打开 MuJoCo 预览窗口
PRINT_ACTIONS=false                          # true 时输出策略每步的 29 维动作
REFERENCE_OFFSET_Y=1.0                       # 蓝色参考机器人的侧向距离, 窗口中按 P 切换显示
CAMERA_PAN_STEP=0.1                          # 点按或长按重复事件的相机地面平移距离, 单位为米
EVAL_MODE="final"                           # final 顺序评估全部动作, training 随机抽取指定回合数
MAX_STEPS=1500                              # 每回合最多 1500 步, 50 Hz 下为 30 秒
NUM_EPISODES=20                             # 仅 training 模式使用, 每回合独立抽取动作, 可重复
SEED=0                                      # 随机评估的种子

exec python -m supermimic.workflows.eval \
    --checkpoint="$CHECKPOINT_PATH" \
    --run-dir="$RUN_DIR" \
    --checkpoint-root="$CHECKPOINT_ROOT" \
    --task="$TASK_NAME" \
    --motion-dir="$MOTION_DIR" \
    --device="$DEVICE" \
    --show-viewer="$SHOW_VIEWER" \
    --print-actions="$PRINT_ACTIONS" \
    --reference-offset-y="$REFERENCE_OFFSET_Y" \
    --camera-pan-step="$CAMERA_PAN_STEP" \
    --mode="$EVAL_MODE" --max-steps="$MAX_STEPS" --num-episodes="$NUM_EPISODES" --seed="$SEED"

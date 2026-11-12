#!/usr/bin/env bash
# SuperMimic 独立评估入口: 修改本文件中的配置后, 用 bash 启动.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ------------------------- 模型来源 ------------------------- #
CHECKPOINT_PATH=""                           # 指定 checkpoint 文件, 留空时自动查找
RUN_DIR=""                                   # 指定实验目录, 与 CHECKPOINT_PATH 二选一
CHECKPOINT_ROOT="checkpoints/supermimic"     # 自动查找实验的根目录
TASK_NAME="tracking"                         # 自动查找实验时使用的任务名称

# ------------------------- 评估设置 ------------------------- #
MOTION_DIR=""                                # 动作数据目录, 留空使用项目默认数据
DEVICE="cuda:0"                              # 仿真与策略设备, 也可设置为 cpu
SHOW_VIEWER=false                            # true 时打开 MuJoCo 预览窗口
PRINT_ACTIONS=false                          # true 时输出策略每步的 29 维动作

exec python -m supermimic.workflows.eval \
    --checkpoint="$CHECKPOINT_PATH" \
    --run-dir="$RUN_DIR" \
    --checkpoint-root="$CHECKPOINT_ROOT" \
    --task="$TASK_NAME" \
    --motion-dir="$MOTION_DIR" \
    --device="$DEVICE" \
    --show-viewer="$SHOW_VIEWER" \
    --print-actions="$PRINT_ACTIONS"

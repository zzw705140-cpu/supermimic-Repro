"""管理本地训练目录, 运行配置和 checkpoint 路径, 模型状态由算法层保存和恢复.

目录格式: task__algorithm__YYYYMMDD_HHMMSS, 同秒重名时追加序号.
训练入口传入保存根目录; dataclass 配置需先用 asdict 转为字典.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

##############################################################################
#                         目录命名与配置转换                                  #
##############################################################################

_CONFIG_NAME = "run_config.json"                       # 每次训练对应的配置文件
_RUN_TIMESTAMP = r"\d{8}_\d{6}"                      # 本地时间, 精确到秒
_CHECKPOINT_PATTERNS = (
    re.compile(r"checkpoint_(\d+)$"),                  # 按训练步数编号, 文件或目录均可
    re.compile(r"epoch_(\d+)\.ckpt$"),                # 保留原工具按 epoch 编号的兼容格式
)


def _name_component(value: str, field: str) -> str:
    """验证任务和算法名称, 仅允许可安全组成运行目录的字符."""

    if not isinstance(value, str) or "__" in value or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"{field} must use letters, digits, '_' or '-', without '__': {value!r}")
    return value


def _run_prefix(task: str, algorithm: str) -> str:
    """生成 task__algorithm__ 前缀, 区分同一根目录下不同任务和算法的运行."""

    return f"{_name_component(task, 'task')}__{_name_component(algorithm, 'algorithm')}__"


def _json_config(value: Any) -> Any:
    """递归将路径, 配置字典和标量转为 JSON 数据, 不支持的类型直接报错."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_config(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return _json_config(value.to_dict())
    if hasattr(value, "item"):
        return _json_config(value.item())
    raise TypeError(f"Unsupported configuration value: {type(value).__name__}")


##############################################################################
#                         新建训练与查找恢复路径                              #
##############################################################################


# 新训练调用本函数建立独立目录, root 由训练脚本传入, 不在工具层写死.
def create_run_dir(root: str | Path, task: str, algorithm: str) -> Path:
    """按本地时间创建训练目录并返回 Path, 同秒重名时追加 _01 等序号.

    输入 root 为保存根目录, task/algorithm 为任务和算法名称.
    例如 task='tracking', algorithm='supermimic' 时, 目录前缀为 tracking__supermimic__.
    创建过程中不会复用已有运行目录, 不创建模型或启动 W&B 实验.
    """

    root = Path(root).expanduser().resolve()
    prefix = _run_prefix(task, algorithm)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while True:
        name = f"{prefix}{timestamp}" + (f"_{suffix:02d}" if suffix else "")
        run_dir = root / name
        try:
            run_dir.mkdir(exist_ok=False)
            return run_dir
        except FileExistsError:
            suffix += 1


# 恢复训练先选运行目录, 再通过 find_latest_checkpoint 选模型保存点.
def find_latest_run_dir(root: str | Path, task: str, algorithm: str) -> Path:
    """按时间戳和同秒序号返回指定任务及算法的最新运行目录 Path.

    不跳过缺少 checkpoint 的最新运行, 后续查找保存点时由检查明确报错.
    """

    root = Path(root).expanduser().resolve()
    prefix = _run_prefix(task, algorithm)
    pattern = re.compile(rf"{re.escape(prefix)}({_RUN_TIMESTAMP})(?:_(\d+))?$")
    if not root.is_dir():
        raise FileNotFoundError(f"Training root does not exist: {root}")
    runs: list[tuple[str, int, Path]] = []
    for path in root.iterdir():
        match = pattern.fullmatch(path.name)
        if path.is_dir() and match:
            runs.append((match.group(1), int(match.group(2) or 0), path))
    if not runs:
        raise FileNotFoundError(f"No run found for task={task!r}, algorithm={algorithm!r} in {root}")
    return max(runs, key=lambda item: (item[0], item[1]))[2]


def find_latest_checkpoint(run_dir: str | Path) -> Path:
    """返回运行目录直接子项中编号最大的 checkpoint 路径, 不读取模型内容.

    支持 checkpoint_N 和 epoch_N.ckpt, 两种格式同时出现时报错, 避免混用步数与轮数.
    """

    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    matches: list[list[tuple[int, Path]]] = [[], []]
    for path in run_dir.iterdir():
        for index, pattern in enumerate(_CHECKPOINT_PATTERNS):
            match = pattern.fullmatch(path.name)
            if match and (path.is_file() or path.is_dir()):
                matches[index].append((int(match.group(1)), path))
                break
    if all(matches):
        raise ValueError(f"Mixed checkpoint naming formats in: {run_dir}")
    checkpoints = matches[0] or matches[1]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in latest run: {run_dir}")
    return max(checkpoints, key=lambda item: item[0])[1]


##############################################################################
#                         配置保存与恢复兼容检查                              #
##############################################################################


def save_run_config(run_dir: str | Path, config: Mapping[str, Any]) -> Path:
    """将配置字典写入已有运行目录的 run_config.json, 返回文件 Path.

    先写临时文件再替换, 已有配置时报错; 原始训练配置不随恢复训练覆盖.
    """

    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    data = _json_config(config)
    target = run_dir / _CONFIG_NAME
    if target.exists():
        raise FileExistsError(f"Run config already exists: {target}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{_CONFIG_NAME}.", dir=run_dir)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_run_config(run_dir: str | Path) -> dict[str, Any]:
    """从运行目录读取 run_config.json 并返回配置字典, 不自动创建模型或环境."""

    path = Path(run_dir).expanduser().resolve() / _CONFIG_NAME
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Run config must contain a JSON object: {path}")
    return config


# 需要保持不变的配置项由算法层指定, 例如模型结构和动作维度; 不把训练总步数强制列入.
def check_resume_config(saved: Mapping[str, Any], current: Mapping[str, Any], immutable_fields: Sequence[str]) -> None:
    """比较保存配置与当前配置中的指定顶层字段, 不兼容时报错, 通过返回 None."""

    differences = []
    for field in immutable_fields:
        if field not in saved or field not in current:
            differences.append(f"{field}: missing from saved or current config")
        elif _json_config(saved[field]) != _json_config(current[field]):
            differences.append(f"{field}: saved={saved[field]!r}, current={current[field]!r}")
    if differences:
        raise ValueError("Resume configuration is incompatible: " + "; ".join(differences))


__all__ = ["create_run_dir", "find_latest_run_dir", "find_latest_checkpoint", "save_run_config", "load_run_config", "check_resume_config"]

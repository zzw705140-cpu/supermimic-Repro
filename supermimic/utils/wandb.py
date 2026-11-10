"""封装 SuperMimic 的 W&B 实验配置和指标记录, 由训练入口创建和关闭."""

from __future__ import annotations

import datetime
import tempfile
from collections.abc import Mapping
from copy import copy
from pathlib import Path
from socket import gethostname
from typing import Any

import absl.flags as flags
import ml_collections
import wandb

##############################################################################
#                         日志指标工具                                        #
##############################################################################


def _recursive_flatten_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    """将嵌套指标字典展平为 a/b/c 键名, 保留原始指标值."""

    flattened: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            nested = _recursive_flatten_dict(value)
            flattened.update({f"{key}/{nested_key}": nested_value for nested_key, nested_value in nested.items()})
        else:
            flattened[str(key)] = value
    return flattened


##############################################################################
#                         W&B 实验日志管理                                    #
##############################################################################


# 创建或续接实验, 同步配置并写入指标; 本地模型保存由算法层和训练入口负责.
class WandBLogger:
    """管理一个 W&B run, 不负责生成训练数据或更新模型.

    构造输入:
        wandb_config 为 W&B 配置, variant 为训练参数字典.
        wandb_output_dir 指定本地日志目录, 未指定时创建临时目录.
        run_name 为页面显示名, run_id 为实验唯一标识, resume 控制续接规则.
        debug=True 使用 disabled 模式, 不上传也不保存离线实验日志.

    接口:
        log(data, step) 接收指标字典和可选训练步数, 嵌套键展平后写入实验.
        finish() 正常结束实验, 本类不保存模型参数或经验池.

    默认项目为 supermimic, 实验描述为 tracking, 脚本均可覆盖.
    账户和密钥使用本机 W&B 登录配置或环境变量, 不在代码中写死.
    """

    # ---------------------------------- 默认配置与初始化 ---------------------------------- #

    @staticmethod
    def get_default_config() -> ml_collections.ConfigDict:
        """返回 SuperMimic 默认 W&B 配置, 不启动实验或连接服务器."""

        config = ml_collections.ConfigDict()
        config.project = "supermimic"            # W&B 页面中的项目名称
        config.entity = ml_collections.config_dict.FieldReference(None, field_type=str)
        config.exp_descriptor = "tracking"       # 实验描述, 参与生成默认实验标识
        config.unique_identifier = ""            # 留空时使用本地时间戳
        config.group = None
        config.tag = None
        return config

    def __init__(self, wandb_config: ml_collections.ConfigDict, variant: Mapping[str, Any], wandb_output_dir: str | Path | None = None, debug: bool = False, run_name: str | None = None, run_id: str | None = None, resume: str | None = None) -> None:
        """准备实验标识与日志目录, 初始化 W&B run, 同步已经解析的命令行参数."""

        self.config = wandb_config

        # 未指定标识时以实验描述和当前时间组成 run id, 恢复训练时由脚本传入原 run_id.
        if self.config.unique_identifier == "":
            self.config.unique_identifier = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = run_id or f"{self.config.exp_descriptor}_{self.config.unique_identifier}"
        self.config.experiment_id = self.experiment_id

        # 日志目录可以由训练入口设在本次运行目录中, 未传入时使用独立临时目录.
        if wandb_output_dir is None:
            wandb_output_dir = tempfile.mkdtemp(prefix="supermimic_wandb_")
        else:
            output_path = Path(wandb_output_dir).expanduser()
            output_path.mkdir(parents=True, exist_ok=True)
            wandb_output_dir = str(output_path)

        self._variant = copy(dict(variant))
        self._variant.setdefault("hostname", gethostname())

        # debug 只禁用实验记录与上传, 不影响调用方执行相同的 log/finish 流程.
        mode = "disabled" if debug else "online"
        tags = None if self.config.tag is None else [str(self.config.tag)]
        # run_name 仅控制显示名, run_id 才用于标识和续接同一次实验.
        self.run = wandb.init(config=self._variant, project=self.config.project, entity=self.config.entity, group=self.config.group, tags=tags, dir=wandb_output_dir, id=self.experiment_id, name=self.experiment_id if run_name is None else run_name, save_code=True, mode=mode, resume=resume)

        # 保留原工具对 absl 参数的支持, 未使用或尚未解析 flags 时跳过.
        if flags.FLAGS.is_parsed():
            flag_dict = {key: getattr(flags.FLAGS, key) for key in flags.FLAGS}
        else:
            flag_dict = {}
        for key, value in flag_dict.items():
            if isinstance(value, ml_collections.ConfigDict):
                flag_dict[key] = value.to_dict()
        if flag_dict:
            self.run.config.update(flag_dict, allow_val_change=True)

    # ---------------------------------- 指标记录与关闭 ---------------------------------- #

    def log(self, data: Mapping[str, Any], step: int | None = None) -> None:
        """将指标写入当前实验, 例如 loss/world_model, step 为脚本传入的训练步数."""

        self.run.log(_recursive_flatten_dict(data), step=step)

    def finish(self) -> None:
        """等待剩余指标写入完成并正常结束实验, 应在训练入口的 finally 中调用."""

        self.run.finish()


##############################################################################
#                         训练入口创建接口                                    #
##############################################################################


def make_wandb_logger(project: str = "supermimic", description: str = "tracking", *, debug: bool = False, entity: str | None = None, group: str | None = None, variant: Mapping[str, Any] | None = None, wandb_output_dir: str | Path | None = None, run_name: str | None = None, run_id: str | None = None, resume: str | None = None) -> WandBLogger:
    """创建并返回 WandBLogger, 项目名称和实验描述均可由启动脚本覆盖.

    variant 为训练配置字典, dataclass 配置需先由调用方用 asdict 转换.
    默认使用在线模式, debug=True 禁用记录与上传; 恢复实验时传原 run_id 和 resume.
    """

    wandb_config = WandBLogger.get_default_config()
    wandb_config.update({"project": project, "entity": entity, "group": group, "exp_descriptor": description, "tag": description})
    return WandBLogger(wandb_config=wandb_config, variant={} if variant is None else variant, wandb_output_dir=wandb_output_dir, debug=debug, run_name=run_name, run_id=run_id, resume=resume)


__all__ = ["WandBLogger", "make_wandb_logger"]

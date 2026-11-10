"""长期仿真交互经验池, 按时间和环境编号保存数据, 不管理任务或调用模型."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import os
from pathlib import Path
import tempfile

import torch

from supermimic.data.validation import ACTION_DIM, REPLAY_FIELDS, REPLAY_OPTIONAL_FIELDS, STATE_WIDTHS, validate_replay_data


##############################################################################
#                         公共参数与张量工具                                  #
##############################################################################


def _positive_int(value, name):
    """检查容量或采样数量, 不接受 bool 或非正整数."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须为正整数")
    return value


def _map_tensors(data, function):
    """递归处理嵌套状态字典中的张量, 不改变字典字段结构."""

    return {key: _map_tensors(value, function) if isinstance(value, Mapping) else function(value) for key, value in data.items()}


##############################################################################
#                         长期仿真交互经验池                                  #
##############################################################################


# 保存所有环境的真实交互, 为世界模型提供连续窗口, 为策略展开提供起点和历史.
class ReplayBuffer:
    """一个统一环形经验池, N 个环境各保存最近 T 步, 总容量为 T*N 条记录.

    构造参数:
        capacity=T, num_envs=N, device 为存储和采样设备.
        store_reward=False 时不保存 reward, 传入 reward 会报错, 避免静默丢弃字段.
        check_values 控制写入时的额外数值检查; metadata 可记录 control_dt 和关节顺序等.

    输入:
        add(data) 接收 N 个环境的一步记录, 六字段 state/next_state 各为 [N,D].
        其余字段与 validation.py 一致, 输入必须与经验池同设备同类型.

    输出:
        sample_windows 返回 (data, mask), 字段形状 [B,K,...], 有效掩码 [B,K].
        sample_starts 返回 (start, history), 前导维分别为 [B] 和 [B,L].
        采样结果拥有独立存储且不保留计算图; 不包含策略观测或未来目标编码.

    本类面向单个采集流程的批量写入, 不提供多线程或多进程同时写入的锁.
    data 是环形物理布局, 上层不要直接改写; 逻辑时间由 write_pos/valid_length 维护.

    用法:
        buffer = ReplayBuffer(capacity=1000, num_envs=64, device="cpu")
        buffer.add(transition)                # 上层先保存重置前的 next_state.
        windows, mask = buffer.sample_windows(batch_size=256, window_length=8)
        start, history = buffer.sample_starts(batch_size=256, history_length=10)
        buffer.save("checkpoints/replay.pt")  # 是否保存及保存频率由脚本决定.
    """

    # ---------------------------------- 初始化与批量写入 ---------------------------------- #

    def __init__(self, capacity: int, num_envs: int, device: str | torch.device = "cpu", *, store_reward: bool = False, check_values: bool = False, metadata: Mapping | None = None):
        """按字段规范分配自有存储, 未写入容量不参与采样或数值检查."""

        self.capacity = _positive_int(capacity, "capacity")   # 每个环境最多保留 T 步
        self.num_envs = _positive_int(num_envs, "num_envs")   # 固定环境轴 N, 总容量为 T*N
        if not isinstance(store_reward, bool):
            raise TypeError("store_reward 必须为 bool")
        self.store_reward, self.check_values = store_reward, check_values
        self.metadata = deepcopy(dict(metadata)) if metadata is not None else {}  # 保存配置副本, 避免上层修改原字典

        # 按 [T,N,D] 分配当前状态和下一状态, 两者均使用经验池自己的存储.
        shape = (self.capacity, self.num_envs)
        self.data = {name: {key: torch.empty(shape + (width,), dtype=torch.float32, device=device) for key, width in STATE_WIDTHS.items()} for name in ("state", "next_state")}

        # 其余字段沿用 validation.py 的规范, reward 由构造参数决定是否分配.
        specs = dict(REPLAY_FIELDS)
        if store_reward:
            specs.update(REPLAY_OPTIONAL_FIELDS)
        for key, (layout, dtype) in specs.items():
            self.data[key] = torch.empty(shape + tuple(layout[1:]), dtype=dtype, device=device)
        self.device = self.data["action"].device
        self.write_pos = self.valid_length = 0  # 下一次写入的物理行号, 已保存的有效时间步数

    def __len__(self):
        """返回有效交互记录数, 等于有效时间步数乘以环境数."""

        return self.valid_length * self.num_envs

    @torch.no_grad()
    def add(self, data: Mapping):
        """校验并复制 N 条记录, 写入完成后推进指针; 不引用上层的可变缓冲."""

        # 先检查完整一步的数据, 通过后才开始写入, 避免错误输入占用有效行.
        validate_replay_data(data, check_values=self.check_values, device=self.device)
        if data["action"].shape != (self.num_envs, ACTION_DIM):
            raise ValueError(f"add 只接受一步数据, action 应为 ({self.num_envs}, {ACTION_DIM})")
        if ("reward" in data) != self.store_reward:
            raise ValueError("reward 是否存在必须与 store_reward 一致")

        # 同一行一次写入 N 个环境, copy_ 与 no_grad 使长期数据独立于原缓冲和计算图.
        for key, target in self.data.items():
            if isinstance(target, Mapping):
                for field, tensor in target.items():
                    tensor[self.write_pos].copy_(data[key][field])
            else:
                target[self.write_pos].copy_(data[key])

        # 写满后循环覆盖最旧一行, 有效时间长度始终不超过 T.
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.valid_length = min(self.valid_length + 1, self.capacity)

    # ---------------------------------- 连续窗口与展开起点采样 ---------------------------------- #

    def _row(self, logical_time):
        """将从最旧记录开始的逻辑时间转换为环形数组的物理行号."""

        return (self.write_pos - self.valid_length + logical_time) % self.capacity

    def _gather(self, times, env_ids, fields=None):
        """沿同一环境取指定逻辑时间, 高级索引产生独立张量而非存储视图."""

        rows = self._row(times)
        fields = self.data.keys() if fields is None else fields
        return {key: _map_tensors(self.data[key], lambda tensor: tensor[rows, env_ids]) if isinstance(self.data[key], Mapping) else self.data[key][rows, env_ids] for key in fields}

    @torch.no_grad()
    # 沿单个环境的时间轴采样连续窗口, 不把不同环境的记录拼成一条轨迹.
    def sample_windows(self, batch_size: int, window_length: int, *, generator: torch.Generator | None = None):
        """随机选 B 个 (时间,环境), 各取 K 步, 返回 (data[B,K,...], mask[B,K]).

        终止那一步自身有效, 从下一步起屏蔽; 也检查 episode_step 跳变, 避免串接回合.
        mask 后的字段仍可能包含其他回合的数据, 世界模型损失必须使用返回的 mask.
        只检查经验池中有足够时间数据, 参考动作剩余长度由上层负责.
        """

        _positive_int(batch_size, "batch_size")
        _positive_int(window_length, "window_length")
        if self.valid_length < window_length:
            raise ValueError("有效时间步不足, 请先收集数据或减小 window_length")

        # 每个样本独立抽取起始时间和环境编号, 后续 K 步保持环境编号不变.
        start = torch.randint(self.valid_length - window_length + 1, (batch_size,), device=self.device, generator=generator)
        envs = torch.randint(self.num_envs, (batch_size,), device=self.device, generator=generator)
        times = start[:, None] + torch.arange(window_length, device=self.device)
        data = self._gather(times, envs[:, None])

        # terminated 或 truncated 都表示回合边界, 步数不连续也视为边界.
        done = data["terminated"] | data["truncated"]
        boundary = done[:, :-1] | (data["episode_step"][:, 1:] != data["episode_step"][:, :-1] + 1)
        # 首条转移有效, 某一步结束后, 后续位置的有效掩码持续为 False.
        mask = torch.cat((torch.ones(batch_size, 1, dtype=torch.bool, device=self.device), (~boundary).long().cumprod(dim=1).bool()), dim=1)
        return data, mask

    @torch.no_grad()
    # 随机抽取展开起点, 同时准备上层生成策略历史观测所需的原始字段.
    def sample_starts(self, batch_size: int, history_length: int = 10, *, generator: torch.Generator | None = None):
        """返回随机起始数据和最近 L 帧原始历史, 起始时刻不要求是回合第一步.

        start 包含 state, previous_action, context, clip_id, frame_id, 前导维为 [B].
        history 包含相同字段, 前导维为 [B,L], 从最早到当前排列.
        历史不越过回合边界或最旧有效数据; 不足部分重复最早可用帧.
        上层用历史生成观测, 并按参考动作剩余长度筛选适合预测 H 步的起点.
        """

        _positive_int(batch_size, "batch_size")
        _positive_int(history_length, "history_length")
        if self.valid_length == 0:
            raise ValueError("经验池为空, 请先收集数据")

        # 起点可以是有效范围内的任意时刻, 不限于回合开始的位置.
        times = torch.randint(self.valid_length, (batch_size,), device=self.device, generator=generator)
        envs = torch.randint(self.num_envs, (batch_size,), device=self.device, generator=generator)
        fields = ("state", "previous_action", "context", "clip_id", "frame_id")
        start = self._gather(times, envs, fields)

        # 从当前向过去检查边界, 比仅依赖回合步数更能防止错误标志造成历史串接.
        back = torch.arange(history_length, device=self.device)
        history_times = (times[:, None] - back).clamp_min(0)
        rows = self._row(history_times)
        steps = self.data["episode_step"][rows, envs[:, None]]
        done = self.data["terminated"][rows, envs[:, None]] | self.data["truncated"][rows, envs[:, None]]
        linked = (steps[:, :-1] == steps[:, 1:] + 1) & ~done[:, 1:] & (history_times[:, :-1] > history_times[:, 1:])
        # 累积连续关系, 得到每个起点最多能向前追溯多少步.
        available = linked.long().cumprod(dim=1).sum(dim=1)
        distance = torch.minimum(back[None], available[:, None])
        # 超出可用历史的位置重复最早帧, 再反转为从最早到当前的顺序.
        history_times = (times[:, None] - distance).flip(1)
        history = self._gather(history_times, envs[:, None], fields)
        return start, history

    # ---------------------------------- 有效数据保存与加载 ---------------------------------- #

    @torch.no_grad()
    def save(self, path: str | Path):
        """将有效数据按时间顺序保存到 CPU .pt 文件, 临时文件写完后原子替换.

        同时保存容量, 环境数, 写入位置, 有效长度及 metadata, 不保存未写入区域.
        metadata 应使用数字, 字符串, 列表, 字典等可安全序列化的配置值.
        保存包含同步和磁盘操作, 调用频率或后台调度由上层决定.
        """

        # 按最旧到最新整理有效行, 只将已经写入的数据保存到 CPU.
        rows = self._row(torch.arange(self.valid_length, device=self.device))
        data = _map_tensors(self.data, lambda tensor: tensor.index_select(0, rows).cpu()) if self.valid_length else None
        snapshot = {"format_version": 1, "capacity": self.capacity, "num_envs": self.num_envs, "write_pos": self.write_pos, "valid_length": self.valid_length, "store_reward": self.store_reward, "metadata": deepcopy(self.metadata), "data": data}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            # 先写同目录临时文件, 写完后替换目标文件, 避免写入中断损坏已有快照.
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(snapshot, stream)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    @torch.no_grad()
    def load(cls, path: str | Path, device: str | torch.device = "cpu", *, check_values: bool = True):
        """恢复容量, 有效数据和环形指针, 可加载到与保存时不同的设备.

        只恢复经验池, 不恢复策略, 仿真现场或随机数状态; 加载结果由调用方持有.
        """

        # 先在 CPU 检查快照的版本, 配置, 指针和有效字段, 再创建目标设备上的经验池.
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        required = {"format_version", "capacity", "num_envs", "write_pos", "valid_length", "store_reward", "metadata", "data"}
        if not isinstance(snapshot, Mapping) or set(snapshot) != required or snapshot["format_version"] != 1:
            raise ValueError("经验池快照字段或格式版本不匹配")
        capacity = _positive_int(snapshot["capacity"], "capacity")
        num_envs = _positive_int(snapshot["num_envs"], "num_envs")
        position, length = snapshot["write_pos"], snapshot["valid_length"]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (position, length)) or not 0 <= position < capacity or not 0 <= length <= capacity:
            raise ValueError("经验池快照指针或有效长度无效")
        if length < capacity and position != length:
            raise ValueError("未满经验池的 write_pos 必须等于 valid_length")
        if not isinstance(snapshot["store_reward"], bool) or not isinstance(snapshot["metadata"], Mapping):
            raise ValueError("经验池快照配置无效")
        data = snapshot["data"]
        if length:
            validate_replay_data(data, check_values=check_values, device="cpu")
            if data["action"].shape != (length, num_envs, ACTION_DIM) or ("reward" in data) != snapshot["store_reward"]:
                raise ValueError("快照数据与容量配置不匹配")
        elif data is not None:
            raise ValueError("空经验池的快照不应包含记录")

        buffer = cls(capacity, num_envs, device, store_reward=snapshot["store_reward"], check_values=check_values, metadata=snapshot["metadata"])
        buffer.write_pos, buffer.valid_length = position, length
        if length:
            # 文件按逻辑时间保存, 恢复时放回对应物理行, 保持后续环形覆盖顺序.
            rows = buffer._row(torch.arange(length, device=buffer.device))
            for key, target in buffer.data.items():
                if isinstance(target, Mapping):
                    for field, tensor in target.items():
                        tensor[rows] = data[key][field].to(buffer.device)
                else:
                    target[rows] = data[key].to(buffer.device)
        return buffer

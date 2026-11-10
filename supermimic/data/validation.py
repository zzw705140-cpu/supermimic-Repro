#######################################################
#                数据检验与数据格式定义                  #
#######################################################

"""
长期经验池：

T: 经验池可保存的时间步数, 容量上限
N: 并行环境数量

replay_buffer
├── state                          # 执行动作前的状态
│   ├── root_pos                    [T, N, 3]
│   ├── root_quat                   [T, N, 4]    # xyzw
│   ├── root_lin_vel                [T, N, 3]
│   ├── root_ang_vel                [T, N, 3]
│   ├── joint_pos                   [T, N, 29]
│   └── joint_vel                   [T, N, 29]
├── next_state                     # 执行动作后的状态, 必须在重置前记录
│   ├── root_pos                    [T, N, 3]
│   ├── root_quat                   [T, N, 4]    # xyzw
│   ├── root_lin_vel                [T, N, 3]
│   ├── root_ang_vel                [T, N, 3]
│   ├── joint_pos                   [T, N, 29]
│   └── joint_vel                   [T, N, 29]
├── action                         [T, N, 29]   # 实际执行动作, 包含执行前加入的噪声
├── previous_action                [T, N, 29]   # 上一时刻实际执行的动作
├── context                        [T, N, 33]   # 原始物理参数, 未归一化
├── clip_id                        [T, N]       # 参考动作编号
├── frame_id                       [T, N]       # state 对应的段内参考帧号
├── episode_step                   [T, N]       # 执行动作前的回合步数
├── terminated                     [T, N]       # 本次转移后是否跟踪失败
├── truncated                      [T, N]       # 本次转移后是否动作结束或回合超时
└── reward                         [T, N]       # 本次转移的奖励, 可选保存用于统计

数据类型:
状态、动作、context、reward      : float32
clip_id、frame_id、episode_step: int64
terminated、truncated          : bool

时间对应:
action[t,n] 将 state[t,n] 推进为 next_state[t,n].
reward[t,n]、terminated[t,n]、truncated[t,n] 对应这次转移.
clip_id[t,n]、frame_id[t,n] 描述动作执行前的参考进度.
环境编号由第 1 维 n 表示, 无需每条记录重复保存 env_id.

存储与采样:
每次写入 N 个环境的一步交互记录, 满容量后覆盖最旧数据.
记录写入经验池自有存储, 不保留计算图, 不长期引用仿真器的共享缓冲.
连续窗口必须按实际时间顺序取样, 不把重置前后当作连续动力学.
done = terminated | truncated, 可在采样时生成, 无需重复保存.
世界模型窗口还需生成有效掩码, 屏蔽回合结束后的部分.

经验池自身维护的元数据, 不属于每条交互记录:
├── write_pos                      # 下一次写入的位置
└── valid_length                   # 当前已保存的有效时间步数, 不超过 T
"""

"""
短期预测轨迹批次：

H: 预测展开步数
B: 本轮预测轨迹数量
L: 初始观测历史长度, 默认 10
K: 未来参考目标帧数, 默认 10

imagined_rollout
├── states                         # 起始状态 + H 个预测状态
│   ├── root_pos                    [H+1, B, 3]
│   ├── root_quat                   [H+1, B, 4]     # xyzw
│   ├── root_lin_vel                [H+1, B, 3]
│   ├── root_ang_vel                [H+1, B, 3]
│   ├── joint_pos                   [H+1, B, 29]
│   └── joint_vel                   [H+1, B, 29]
├── actions                         [H, B, 29]
├── initial_previous_action         [B, 29]
├── initial_observation_history     [B, L, 99]
├── context                         [B, 33]         # 本条轨迹共享的原始物理参数
├── clip_id                         [B]             # 本条轨迹共享的参考动作编号
├── frame_ids                       [H+1, B]        # 各状态对应的参考帧号
├── goals                           [H+1, B, K*73]  # 由 MDP 生成, 可选缓存
├── rewards                         [H, B]
└── soft_terminations               [H, B]

时间对应:
actions[t] 将 states[t] 推进为 states[t+1].
rewards[t] 和 soft_terminations[t] 对应这次转移后的 states[t+1].
第 0 步的上一动作为 initial_previous_action;
第 t>0 步的上一动作为 actions[t-1].

Critic 更新时另外生成:
├── critic_features                 [H+1, B, 184]   # 默认跟踪 14 个身体部位, 未拼 goals
├── values                          [H+1, B]        # 包含末端状态价值
└── return_targets                  [H, B]

策略反向传播前, 预测状态、动作和奖励保留计算图.
Critic 更新使用分离梯度后的特征和固定回报目标.
本轮更新结束后释放临时数据, 不写入长期经验池.
"""

from collections.abc import Mapping
from types import MappingProxyType

import torch


##############################################################################
#                         公共字段与数据类型规范                              #
##############################################################################

STATE_WIDTHS = MappingProxyType({
    "root_pos": 3,                 # 根部位置, 已减去各环境原点
    "root_quat": 4,                # 根部四元数, xyzw 顺序
    "root_lin_vel": 3,             # 根部线速度
    "root_ang_vel": 3,             # 根部角速度
    "joint_pos": 29,               # 关节角, 顺序与动作库一致
    "joint_vel": 29,               # 关节速度
})
ACTION_DIM = 29
CONTEXT_DIM = 33
OBS_DIM = 99
GOAL_FRAME_DIM = 73

# 每项为 (形状, 数据类型); 字符串表示批量或时间维, 整数表示固定维度.
# P 表示长期数据的前导维:  单步写入时为 (N,), 按时间组织时为 (T,N).
REPLAY_FIELDS = MappingProxyType({
    "action": (("P", ACTION_DIM), torch.float32),
    "previous_action": (("P", ACTION_DIM), torch.float32),
    "context": (("P", CONTEXT_DIM), torch.float32),
    "clip_id": (("P",), torch.int64),
    "frame_id": (("P",), torch.int64),
    "episode_step": (("P",), torch.int64),
    "terminated": (("P",), torch.bool),
    "truncated": (("P",), torch.bool),
})
REPLAY_OPTIONAL_FIELDS = MappingProxyType({
    "reward": (("P",), torch.float32),
})

# S = H+1; G = K*73. context 和 clip_id 在整条预测轨迹中共享.
ROLLOUT_FIELDS = MappingProxyType({
    "actions": (("H", "B", ACTION_DIM), torch.float32),
    "initial_previous_action": (("B", ACTION_DIM), torch.float32),
    "initial_observation_history": (("B", "L", OBS_DIM), torch.float32),
    "context": (("B", CONTEXT_DIM), torch.float32),
    "clip_id": (("B",), torch.int64),
    "frame_ids": (("S", "B"), torch.int64),
    "rewards": (("H", "B"), torch.float32),
    "soft_terminations": (("H", "B"), torch.float32),
})
ROLLOUT_OPTIONAL_FIELDS = MappingProxyType({
    "goals": (("S", "B", "G"), torch.float32),
})


##############################################################################
#                         公共结构与数值检查工具                              #
##############################################################################


def _check_fields(data, required, optional, path):
    """检查字典字段, 缺失或拼写错误时给出完整路径; 不增删字段."""

    if not isinstance(data, Mapping):
        raise TypeError(f"{path}: 必须为字段字典, 实际为 {type(data).__name__}")
    missing = set(required) - data.keys()
    unknown = data.keys() - set(required) - set(optional)
    if missing:
        raise ValueError(f"{path}: 缺少字段 {sorted(missing)}")
    if unknown:
        raise ValueError(f"{path}: 未知字段 {sorted(unknown, key=str)}")


def _check_tensor(value, shape, dtype, device, path, check_values):
    """核对张量形状、类型和设备; 可选检查有限值, 不转换或复制输入."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path}: 必须为 torch.Tensor, 实际为 {type(value).__name__}")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"{path}: 期望形状 {tuple(shape)}, 实际为 {tuple(value.shape)}")
    if value.dtype != dtype:
        raise ValueError(f"{path}: 期望类型 {dtype}, 实际为 {value.dtype}")
    if value.device != device:
        raise ValueError(f"{path}: 期望设备 {device}, 实际为 {value.device}")
    if check_values and value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"{path}: 含有 NaN 或 Inf")


def _check_anchor(value, ranks, device, path):
    """由动作张量确定时间、批量维和设备, 并检查调用方指定的设备."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path}: 必须为 torch.Tensor")
    if value.ndim not in ranks or value.shape[-1] != ACTION_DIM:
        raise ValueError(f"{path}: 期望 {ranks} 中的一种维数且末维为 {ACTION_DIM}, 实际形状为 {tuple(value.shape)}")
    if any(size <= 0 for size in value.shape[:-1]):
        raise ValueError(f"{path}: 时间维和批量维必须大于 0")
    if device is not None:
        expected = torch.device(device)
        # device='cuda' 接受任意 CUDA 编号; 指定 'cuda:0' 时要求编号也一致.
        if expected.type != value.device.type or (expected.index is not None and expected.index != value.device.index):
            raise ValueError(f"{path}: 期望设备 {expected}, 实际为 {value.device}")
    return value.device


def _check_state(state, leading_shape, device, path, check_values):
    """共用六字段状态规范, 检查对应批量维以及可选的单位四元数约束."""

    _check_fields(state, STATE_WIDTHS, (), path)
    for key, width in STATE_WIDTHS.items():
        _check_tensor(state[key], leading_shape + (width,), torch.float32, device, f"{path}.{key}", check_values)
    if check_values:
        norm = torch.linalg.vector_norm(state["root_quat"], dim=-1)
        if ((norm - 1).abs() > 1e-3).any():
            raise ValueError(f"{path}.root_quat: 必须为单位四元数, 长度误差不得超过 1e-3")


def _check_specs(data, specs, dimensions, device, path, check_values):
    """将规范中的维度符号替换成实际长度, 逐字段调用同一检查逻辑."""

    for key, (layout, dtype) in specs.items():
        if key not in data:           # 可选字段不存在时跳过; 必需字段已在入口检查
            continue
        shape = []
        for axis in layout:
            size = dimensions[axis] if isinstance(axis, str) else axis
            shape.extend(size if isinstance(size, tuple) else (size,))
        _check_tensor(data[key], shape, dtype, device, f"{path}.{key}", check_values)


def _check_nonnegative(data, keys, path):
    """检查编号和计数非负; 动作库的编号上限由持有动作库的上层检查."""

    for key in keys:
        if (data[key] < 0).any():
            raise ValueError(f"{path}.{key}: 编号或步数不能为负")


##############################################################################
#                         长期经验数据检查                                    #
##############################################################################

"""
两个函数检查的是：
（传入的数据字典是否符合我们约定的格式。）
字段：有没有缺失、拼错或多余的字段。
形状：例如动作末维必须为 29；短期轨迹有 H 个动作，就必须有 H+1 个状态。
类型：状态和动作为 float32，编号为 int64，结束标志为 bool。
设备：同一批数据的张量是否都在同一个 CPU 或 GPU 上。

check_values=True 后，还检查有没有 NaN/Inf、四元数是否为单位长度、编号是否非负、软终止值是否在 [0,1]。
"""

@torch.no_grad()
def validate_replay_data(data: Mapping, *, check_values: bool = False, device=None) -> None:
    """
    检查单步 [N,...] 或按时间组织的 [T,N,...] 真实交互数据.

    state/next_state 为六字段字典, reward 可省略; 其他字段均必需.
    device 可指定期望设备, 未指定时要求所有张量与 action 同设备.
    check_values=True 时额外检查有限值、单位四元数和非负编号.

    通过时返回 None, 类型错误抛出 TypeError, 形状或数值错误抛出 ValueError.
    本函数只检查, 不分配经验池、不采样、不修改输入或切断已有计算图.
    允许传入尚带梯度的待写入数据; 写入自有存储并去除梯度由经验池负责.
    尚未写入的容量区域不应参与数值检查, 应传入已填充的有效部分.
    """

    path = "replay"
    _check_fields(data, ("state", "next_state", *REPLAY_FIELDS), REPLAY_OPTIONAL_FIELDS, path)
    actual_device = _check_anchor(data["action"], (2, 3), device, f"{path}.action")
    leading_shape = tuple(data["action"].shape[:-1])

    for key in ("state", "next_state"):
        _check_state(data[key], leading_shape, actual_device, f"{path}.{key}", check_values)
    dimensions = {"P": leading_shape}
    _check_specs(data, REPLAY_FIELDS, dimensions, actual_device, path, check_values)
    _check_specs(data, REPLAY_OPTIONAL_FIELDS, dimensions, actual_device, path, check_values)
    if check_values:
        _check_nonnegative(data, ("clip_id", "frame_id", "episode_step"), path)


##############################################################################
#                         短期预测轨迹检查                                    #
##############################################################################


@torch.no_grad()
def validate_imagined_rollout(data: Mapping, *, history_length: int = 10, goal_frames: int = 10, check_values: bool = False, device=None) -> None:
    """
    检查 H 步预测批次: states 为 [H+1,B,...], actions 为 [H,B,29].

    history_length/goal_frames 应与算法和 MDP 的配置一致, 默认均为 10.
    goals 可省略; Critic 的特征、价值及回报目标由更新代码另行生成,
    不属于这个基础轨迹字典, 额外传入会按未知字段报错.

    默认只检查结构、类型和设备; check_values=True 额外检查有限值、
    单位四元数、非负编号及 soft_terminations 位于 [0,1].
    数值检查在 GPU 上可能触发同步, 建议调试时开启.

    通过时返回 None, 不改变张量对象、数值或已有计算图.
    no_grad 仅避免检查运算创建新计算图, 不会 detach 传入的预测轨迹.
    不要求每个字段 requires_grad=True; 初始状态和参考数据可以是常量.
    """

    for name, value in (("history_length", history_length), ("goal_frames", goal_frames)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name}: 必须为正整数")
    path = "rollout"
    _check_fields(data, ("states", *ROLLOUT_FIELDS), ROLLOUT_OPTIONAL_FIELDS, path)
    actual_device = _check_anchor(data["actions"], (3,), device, f"{path}.actions")
    horizon, batch = data["actions"].shape[:2]
    dimensions = {"H": horizon, "S": horizon + 1, "B": batch, "L": history_length, "G": goal_frames * GOAL_FRAME_DIM}

    _check_state(data["states"], (horizon + 1, batch), actual_device, f"{path}.states", check_values)
    _check_specs(data, ROLLOUT_FIELDS, dimensions, actual_device, path, check_values)
    _check_specs(data, ROLLOUT_OPTIONAL_FIELDS, dimensions, actual_device, path, check_values)
    if check_values:
        _check_nonnegative(data, ("clip_id", "frame_ids"), path)
        termination = data["soft_terminations"]
        if ((termination < 0) | (termination > 1)).any():
            raise ValueError(f"{path}.soft_terminations: 必须位于 [0,1]")

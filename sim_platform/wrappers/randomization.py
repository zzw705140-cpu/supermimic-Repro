"""生成仿真随机参数与重置扰动, 具体应用时机和参数生效由调用方负责."""

from collections.abc import Mapping

import torch

from sim_platform.wrappers.rotations import quaternion_multiply, rotation_vector_to_quaternion


"""---------------------------- 物理与控制参数 ----------------------------"""


def sample_physical_parameters(num_envs: int, device: str | torch.device, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """均匀采样 N 个环境的摩擦 [N,1] 和躯干质心偏移 [N,3].

    偏移以米表示, 相对机器人原始质心; 仅返回采样值, 不修改仿真器.
    摩擦范围 [0.3,1.2], 质心各轴范围分别为 +/-0.025, +/-0.05, +/-0.05.
    """

    if num_envs < 0:
        raise ValueError("环境数量不能为负数")
    friction = torch.empty((num_envs, 1), device=device, dtype=dtype).uniform_(0.3, 1.2)
    com_range = torch.tensor((0.025, 0.05, 0.05), device=device, dtype=dtype)
    com_offset = torch.empty((num_envs, 3), device=device, dtype=dtype).uniform_(-1, 1) * com_range
    return friction, com_offset


def sample_joint_offsets(num_envs: int, device: str | torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """均匀采样 N 个环境的 29 维关节目标偏移 [N,29], 范围为 +/-0.01 弧度.

    返回偏移, 不修改动作或执行器; 调用方应在重置指定环境后应用对应行.
    """

    if num_envs < 0:
        raise ValueError("环境数量不能为负数")
    return torch.empty((num_envs, 29), device=device, dtype=dtype).uniform_(-0.01, 0.01)


"""---------------------------- 重置状态扰动 ----------------------------"""


def perturb_reset_state(state: Mapping[str, torch.Tensor], joint_lower: torch.Tensor, joint_upper: torch.Tensor) -> dict[str, torch.Tensor]:
    """为 N 个参考状态施加 vibed 的重置扰动, 返回新状态, 不修改输入.

    state 包含 root_pos [N,3], root_quat [N,4] (xyzw), root_lin_vel [N,3],
    root_ang_vel [N,3], joint_pos [N,29], joint_vel [N,29]. joint_lower 和
    joint_upper 为同设备的关节位置硬限位 [29], 输出关节角裁剪到限位内.
    位置、姿态、线速度、角速度和关节角加均匀扰动; 关节速度保持不变.
    """

    required = {"root_pos": 3, "root_quat": 4, "root_lin_vel": 3, "root_ang_vel": 3, "joint_pos": 29, "joint_vel": 29}
    if any(key not in state for key in required):
        raise ValueError("参考状态必须包含全部六个字段")
    root = state["root_pos"]
    if root.ndim != 2 or any(state[key].shape != (root.shape[0], width) for key, width in required.items()):
        raise ValueError("参考状态必须为形状一致的批量 [N,D]")
    if any(state[key].device != root.device or state[key].dtype != root.dtype for key in required):
        raise ValueError("参考状态必须使用相同设备和数据类型")
    if joint_lower.shape != (29,) or joint_upper.shape != (29,) or joint_lower.device != root.device or joint_upper.device != root.device or joint_lower.dtype != root.dtype or joint_upper.dtype != root.dtype or torch.any(joint_lower > joint_upper):
        raise ValueError("关节限位必须是与状态同设备同类型的有效 [29] 张量")

    # 姿态扰动在世界坐标系左乘, 与 vibed 的旋转矩阵操作一致.
    position_range = root.new_tensor((0.05, 0.05, 0.01))
    rotation_range = root.new_tensor((0.1, 0.1, 0.2))
    linear_range = root.new_tensor((0.5, 0.5, 0.2))
    angular_range = root.new_tensor((0.52, 0.52, 0.78))
    rotation = torch.empty_like(root).uniform_(-1, 1) * rotation_range
    result = dict(state)
    result["root_pos"] = root + torch.empty_like(root).uniform_(-1, 1) * position_range
    result["root_quat"] = quaternion_multiply(rotation_vector_to_quaternion(rotation), state["root_quat"])
    result["root_lin_vel"] = state["root_lin_vel"] + torch.empty_like(root).uniform_(-1, 1) * linear_range
    result["root_ang_vel"] = state["root_ang_vel"] + torch.empty_like(root).uniform_(-1, 1) * angular_range
    result["joint_pos"] = (state["joint_pos"] + torch.empty_like(state["joint_pos"]).uniform_(-0.1, 0.1)).clamp(min=joint_lower, max=joint_upper)
    result["joint_vel"] = state["joint_vel"].clone()
    return result

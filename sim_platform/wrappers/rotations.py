"""
公共旋转数学工具和训练回报目标
可微批量旋转工具: 四元数使用 xyzw, 旋转矩阵左乘列向量.
回报目标计算单独分区, 不参与旋转运算.
"""

import math

import torch
from torch.nn import functional as F

##############################################################################
#                         旋转表示与姿态误差                                  #
##############################################################################


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """将非零 xyzw 四元数 [...,4] 归一化并转换为 [...,3,3], 保留梯度."""

    if quaternion.shape[-1:] != (4,):
        raise ValueError("四元数最后一维必须为 4, 使用 xyzw 顺序")
    norm = quaternion.norm(dim=-1, keepdim=True)
    if torch.any(norm < 1e-8):
        raise ValueError("四元数不能为零")
    x, y, z, w = (quaternion / norm).unbind(-1)
    row0 = torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)), dim=-1)
    row1 = torch.stack((2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)), dim=-1)
    row2 = torch.stack((2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def matrix_to_rotation_6d(rotation: torch.Tensor) -> torch.Tensor:
    """取矩阵前两列并按行展平, 输出 [...,6]; 顺序与 vibed 一致."""

    return rotation[..., :, :2].reshape(rotation.shape[:-2] + (6,))


def heading_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """从机体 X 轴的水平投影提取朝向; 投影退化时取零朝向."""

    x, y = rotation[..., 0, 0], rotation[..., 1, 0]
    squared_norm = x*x + y*y
    norm = squared_norm.clamp_min(1e-12).sqrt()
    cosine = torch.where(squared_norm > 1e-12, x / norm, torch.ones_like(x))
    sine = torch.where(squared_norm > 1e-12, y / norm, torch.zeros_like(y))
    zero, one = torch.zeros_like(x), torch.ones_like(x)
    return torch.stack((cosine, -sine, zero, sine, cosine, zero, zero, zero, one), dim=-1).reshape(rotation.shape[:-2] + (3, 3))


def rotation_error_squared(actual: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """计算旋转间最短夹角的平方 [...], 使用 atan2 避免 acos 在零误差处的奇异梯度.

    完全一致时为零; 180 度处距离不可微, autograd 返回有限的选定梯度.
    """

    relative = actual.transpose(-1, -2) @ reference
    skew = torch.stack((relative[..., 2, 1] - relative[..., 1, 2], relative[..., 0, 2] - relative[..., 2, 0], relative[..., 1, 0] - relative[..., 0, 1]), dim=-1)
    sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    cosine = 0.5 * (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1)
    return torch.atan2(sine, cosine).square()


##############################################################################
#                         四元数与旋转向量                                    #
##############################################################################


def quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """计算 xyzw 四元数乘积 [...,4], 表示先执行 right 旋转再执行 left 旋转."""

    vector = left[..., 3:] * right[..., :3] + right[..., 3:] * left[..., :3] + torch.cross(left[..., :3], right[..., :3], dim=-1)
    scalar = left[..., 3:] * right[..., 3:] - (left[..., :3] * right[..., :3]).sum(-1, keepdim=True)
    return torch.cat((vector, scalar), dim=-1)


def quaternion_to_rotation_vector(quaternion: torch.Tensor) -> torch.Tensor:
    """将 xyzw 四元数转为最短旋转向量 [...,3], 在零旋转附近使用稳定展开."""

    quaternion = F.normalize(quaternion, dim=-1)
    quaternion = torch.where(quaternion[..., 3:] < 0, -quaternion, quaternion)
    vector, scalar = quaternion[..., :3], quaternion[..., 3:]
    sine = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sine, scalar)
    scale = torch.where(sine > 1e-6, angle / sine.clamp_min(1e-6), 2 + sine.square() / 3 + 3 * sine.pow(4) / 20)
    return vector * scale


def rotation_vector_to_quaternion(vector: torch.Tensor) -> torch.Tensor:
    """将旋转向量 [...,3] 转为 xyzw 单位四元数, sinc 保证零增量处梯度有限."""

    angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    return torch.cat((0.5 * torch.sinc(angle / (2 * math.pi)) * vector, torch.cos(0.5 * angle)), dim=-1)


##############################################################################
#                         训练回报目标                                        #
##############################################################################


@torch.no_grad()
def lambda_returns(rewards, soft_terminations, values, gamma: float, gae_lambda: float):
    """按论文式 3 计算固定回报目标, 输入奖励和软终止 [H,B], 价值 [H+1,B].

    输出 targets [H,B], 递推起点为末端 values[H], 不保留任何输入的计算图.
    此函数处理强化学习回报目标, 与上面的旋转函数没有数据或功能依赖.
    """

    if rewards.ndim != 2 or rewards.shape[0] == 0 or soft_terminations.shape != rewards.shape or values.shape != (rewards.shape[0] + 1, rewards.shape[1]):
        raise ValueError("回报目标要求 rewards/d 为 [H,B], values 为 [H+1,B], H 必须大于零")
    if not 0 <= gamma < 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma 或 gae_lambda 超出合法范围")
    if any(tensor.device != rewards.device or tensor.dtype != rewards.dtype for tensor in (soft_terminations, values)):
        raise ValueError("奖励, 软终止和价值必须同设备同精度")
    target, targets = values[-1], []
    for time in range(rewards.shape[0] - 1, -1, -1):
        target = rewards[time] + gamma * (1 - soft_terminations[time]) * ((1 - gae_lambda) * values[time + 1] + gae_lambda * target)
        targets.append(target)
    return torch.stack(targets[::-1])

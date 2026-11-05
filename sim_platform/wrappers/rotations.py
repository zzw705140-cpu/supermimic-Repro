"""可微批量旋转工具: 四元数使用 xyzw, 旋转矩阵左乘列向量."""

import torch

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

"""
通用 PyTorch MLP 与运行统计归一化, 供动力学, 策略和 Critic 网络复用.

RunningNorm 负责整理数据尺度，MLP 负责学习输入到输出的映射。 两者是独立组件，由具体网络组合使用：
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import nn

##############################################################################
#                         参数检查与激活函数                                  #
##############################################################################


def _positive_int(value: int, name: str) -> int:
    """检查输入, 输出及隐藏层宽度, 不接受 bool 或非正整数."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须为正整数")
    return value


def _activation_class(activation: str | type[nn.Module]) -> type[nn.Module]:
    """将激活名称或无参模块类统一为模块类, 每个隐藏层创建独立实例."""

    if isinstance(activation, str):
        supported = {"elu": nn.ELU, "silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
        if activation.lower() not in supported:
            raise ValueError(f"未知激活函数 {activation!r}, 支持 {tuple(supported)}")
        return supported[activation.lower()]
    
    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation
    raise TypeError("activation 必须为支持的字符串或可无参构造的 nn.Module 类")


##############################################################################
#                         通用多层感知机                                      #
##############################################################################


# 只定义全连接映射, 层宽和层数由调用方传入, 不处理任务输入拼接或状态积分.
class MLP(nn.Module):
    """以统一组件构建不同深度, 宽度和激活函数的全连接网络.

    构造参数:
        input_dim 为输入维度 F, output_dim 为输出维度 A.
        hidden_dims 只包含隐藏层宽度, 如 (2048,2048,2048) 表示三层隐藏层.
        空元组表示只有一个线性输出层; 输出层不包含在 hidden_dims 中.
        activation 默认 'elu', 也支持 'silu', 'relu', 'gelu', 'tanh' 或无参模块类.
        layer_norm=True 时在每个隐藏层的线性变换后, 激活之前使用 LayerNorm.

    输入与输出:
        单样本 [F] -> [A], 批量 [B,F] -> [B,A], 轨迹 [H,B,F] -> [H,B,A].
        仅改变最后的特征维, 不展平历史帧或拼接其他输入.
        最后一层固定为 Linear, 编码器输出的 tanh 等操作由具体网络负责.

    初始化:
        使用 PyTorch Linear 默认初始化, 与 vibed 的基础 MLP 一致.
        net[-1] 为输出层, 具体网络可按需将其权重置零或缩小.

    配置:
        input_dim=128
        hidden_dims=(256, 256)          项数决定隐藏层数量，各项数值决定对应层宽度
        output_dim=1
        activation="gelu"               激活函数可以替换为 ELU、SiLU 
        layer_norm=True                 layer_norm=False 时跳过 LayerNorm。

        输入 features
        [B, 128]
            ↓ Linear → LayerNorm → GELU
        [B, 256]
            ↓ Linear → LayerNorm → GELU
        [B, 256]
            ↓ Linear
        [B, 1]
        输出
    """

    # ---------------------------------- 网络组装 ---------------------------------- #
    
    # 根据传入配置搭建网络，创建各层及其参数。
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int, *, activation: str | type[nn.Module] = "elu", layer_norm: bool = False):
        """检查维度并按配置组装隐藏层与线性输出层, 不固定任何任务的网络规模."""

        super().__init__()

        # 检查并保存网络尺寸
        self.input_dim = _positive_int(input_dim, "input_dim")
        self.output_dim = _positive_int(output_dim, "output_dim")
        self.hidden_dims = tuple(_positive_int(width, "hidden_dims 中的宽度") for width in hidden_dims)
        # 检查归一化设置，确定激活函数
        if not isinstance(layer_norm, bool):
            raise TypeError("layer_norm 必须为 bool")
        activation_class = _activation_class(activation)

        # 每个隐藏层执行 Linear -> 可选 LayerNorm -> 激活, 不使用 Dropout.
        layers, width = [], self.input_dim
        for hidden_width in self.hidden_dims:
            # 将当前特征映射到本层规定的宽度
            layers.append(nn.Linear(width, hidden_width))
            # 如果开启归一化，就追加 LayerNorm
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_width))
            # 创建并追加激活函数
            layers.append(activation_class())
            # 更新 width，供下一层确定输入尺寸
            width = hidden_width

        # 添加最后的线性输出层，将最后一个隐藏层映射到输出维度。这一层不添加归一化和激活函数。
        layers.append(nn.Linear(width, self.output_dim))
        # 所有层按顺序组成一个完整网络
        self.net = nn.Sequential(*layers)

    # ---------------------------------- 前向计算 ---------------------------------- #

    # 接收输入数据，检查形状，然后通过已经搭好的网络计算输出。
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features [...,F] -> 隐藏层 Linear/LN/激活 -> 输出 Linear -> [...,A].

        LN 仅在 layer_norm=True 时存在, 前导的批量维和时间维保持不变.
        调用前由上层统一网络与输入的设备和精度, 前向过程保留梯度.
        """

        if not isinstance(features, torch.Tensor):
            raise TypeError("features 必须为 torch.Tensor")
        if features.ndim < 1 or features.shape[-1] != self.input_dim:
            raise ValueError(f"features 最后一维必须为 {self.input_dim}, 实际形状为 {tuple(features.shape)}")
        return self.net(features)


##############################################################################
#                         运行均值与方差归一化                                #
##############################################################################


# 统计数据分布, 与隐藏层中的 LayerNorm 分工不同, 不作为可训练神经网络.
class RunningNorm(nn.Module):
    """参考 vibed 维护逐特征运行统计, 用于观测, 目标和动力学输入输出的归一化.

    构造参数:
        dim 为特征维度 D, eps 为标准差的平滑下限, 默认 1e-2.
        max_count 默认 2e8, 已统计样本数达到该值后停止更新.
        更新按整批计数, 最后一批可使 count 略超过 max_count, 沿用 vibed 规则.

    输入与接口:
        update(x) 接收浮点 [...,D], 合并前导维统计样本, 不保存输入或计算图.
        forward(x) 对 [D], [B,D] 或 [H,B,D] 归一化, 返回相同形状并保留输入梯度.
        std() 返回 [D] 标准差, 可用来将模型预测恢复为原尺度.

    生命周期:
        mean/var/count 注册为 buffer, 随 .to(device) 移动并包含在 state_dict 中.
        初始 mean=0, var=1, count=0; 没有可训练参数.
        forward 不更新统计, 训练时显式调用 update, 测评时不调用 update.
    """

    # ---------------------------------- 初始化与统计更新 ---------------------------------- #

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        """创建均值, 总体方差和样本计数, 不分配历史数据缓存."""

        super().__init__()
        self.dim = _positive_int(dim, "dim")
        if isinstance(max_count, bool) or not math.isfinite(max_count) or max_count <= 0:
            raise ValueError("max_count 必须为正的有限数值")
        if isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps 必须为正的有限数值")
        self.max_count, self.eps = max_count, eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros((), dtype=torch.int64))  # 整数计数避免大样本量时丢失计数精度

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """将本批均值和总体方差合并到已有统计, 原地更新 buffer, 空批次不处理."""

        self._check_input(x)
        if x.numel() == 0 or self.count >= self.max_count:
            return

        # 与 vibed 一样使用 float32 统计, 将批量维和时间维视作独立样本.
        values = x.reshape(-1, self.dim).float()
        batch_count = values.shape[0]
        batch_mean, batch_var = values.mean(dim=0), values.var(dim=0, unbiased=False)
        total = self.count + batch_count
        delta = batch_mean - self.mean

        # 方差合并还需计入两批均值不同造成的偏差, 不能只对各批方差取平均.
        combined_var = (self.var * self.count + batch_var * batch_count + delta.square() * self.count * batch_count / total) / total
        self.mean.add_(delta * (batch_count / total))
        self.var.copy_(combined_var.clamp_min(0))
        self.count.add_(batch_count)

    # ---------------------------------- 归一化与标准差读取 ---------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [...,D] -> 减运行均值 -> 除平滑标准差 -> 同形状输出, 不更新统计."""

        self._check_input(x)
        return (x - self.mean) / self.std()

    def std(self) -> torch.Tensor:
        """返回 sqrt(var + eps**2) [D], 恒定特征也有正标准差, 避免除零."""

        return torch.sqrt(self.var + self.eps**2)

    def _check_input(self, x: torch.Tensor) -> None:
        """检查特征维和设备, 仅接受浮点张量, 不复制或修改输入."""

        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError("归一化输入必须为浮点 torch.Tensor")
        if x.ndim < 1 or x.shape[-1] != self.dim:
            raise ValueError(f"归一化输入最后一维必须为 {self.dim}, 实际形状为 {tuple(x.shape)}")
        if x.device != self.mean.device:
            raise ValueError("归一化组件与输入必须在同一设备")


__all__ = ["MLP", "RunningNorm"]

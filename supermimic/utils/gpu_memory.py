"""读取 GPU 当前显存和 PyTorch 显存峰值, 调用频率与停止策略由训练流程决定."""

from __future__ import annotations

import torch

##############################################################################
#                         当前显存与峰值显存                                  #
##############################################################################


# 当前占用统计整张显卡; 峰值仅统计当前进程由 PyTorch 分配的张量显存.
def read_gpu_memory(device: int | str | torch.device | None = None) -> dict[str, int | float]:
    """返回显存字节数和整卡占用比例; device=None 表示当前 CUDA 设备.

    peak_allocated_bytes 是自上次重置峰值统计以来的最大张量分配量,
    不包含仿真器、其他进程及 PyTorch 缓存中未使用的显存.
    本函数不会重置峰值, 不控制训练停止.
    """

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    used_bytes = total_bytes - free_bytes
    return {
        "total_bytes": total_bytes,
        "free_bytes": free_bytes,
        "used_bytes": used_bytes,
        "used_fraction": used_bytes / total_bytes,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }


# 训练流程在一个统计区间开始前调用; 不释放显存, 只重置 PyTorch 的峰值计数.
def reset_gpu_memory_peak(device: int | str | torch.device | None = None) -> None:
    """从当前已分配量重新开始统计指定设备的 PyTorch 显存峰值."""

    torch.cuda.reset_peak_memory_stats(device)

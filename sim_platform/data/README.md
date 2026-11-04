# 参考动作数据

- `lafan1_g1/`: 已下载 40 段重定向到 G1 的 LAFAN1 动作 CSV.
- `motion.py`: 加载 CSV, 重采样, 计算根部与关节速度, 提供单帧和未来帧查询.

数据来源：<https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main/g1>。

每个 CSV 对应一段动作，原始频率为 30 FPS；每行包含根部位置 XYZ、
根部四元数 XYZW，以及按约定顺序排列的 29 个关节角，共 36 列。

这些轨迹用于提供模仿目标、计算跟踪奖励和设置参考初始状态。
世界模型的监督数据则来自训练期间的仿真交互，由经验池另行管理。

```python
from sim_platform.data.motion import MotionLib

motion = MotionLib(control_dt=0.02, device="cpu")
state = motion.get_reference(clip_id=0, frame_id=0)
future = motion.get_future_reference(clip_id=[0, 1], frame_id=[0, 10], num_frames=10)
```

状态字段为 `root_pos`, `root_quat`, `root_lin_vel`, `root_ang_vel`,
`joint_pos`, `joint_vel`. 根部量使用参考世界坐标系, 四元数为 xyzw.
原始频率默认为 30 FPS, `control_dt` 由调用方设置并与环境保持一致.
输出为指定设备上的 float32 张量, 原始 CSV 不变.
环境负责动作和初始帧采样, 身体正运动学与策略目标特征编码尚未在此实现.
# 参考动作数据

- `lafan1_g1/`: 已下载 40 段重定向到 G1 的 LAFAN1 动作 CSV.
- `motion.py`: 加载 CSV, 重采样, 计算根部与关节速度, 可分批预计算并缓存参考身体状态.
- `kinematics.py`: 从 MJCF 读取结构, 用可微 PyTorch 运算计算指定连杆的世界系位姿和速度.

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
环境负责动作和初始帧采样. 策略目标特征编码尚未实现.

创建时传入 `G1Kinematics`, 将参考身体状态与根部和关节状态一起缓存:

```python
from sim_platform.data.kinematics import G1Kinematics

# 三项配置由调用方提供: MJCF 路径, CSV 关节顺序, 需要跟踪的连杆名称列表.
kin = G1Kinematics(mjcf_path, joint_names, tracked_bodies, device="cpu")
motion = MotionLib(control_dt=0.02, device="cpu", kinematics=kin, fk_batch_size=8192)
state = motion.get_reference(clip_id=0, frame_id=0)
reference_body_positions = state["body_pos"]

# 单独使用可微正运动学时, 只传入根部和关节字段.
base = motion.get_reference(clip_id=0, frame_id=0, include_body=False)
bodies = kin.forward(**base)
```

提供 `kinematics` 后, 查询默认额外返回 `body_pos`, `body_rot`, `body_lin_vel`,
`body_ang_vel`, 输出顺序与 `motion.tracked_bodies` 一致. 未来查询超出动作末尾时,
保持末帧身体位姿并将身体速度置零. 不提供 `kinematics` 时保持原来的六字段查询.

初始化期间使用 no_grad 分批计算并预分配缓存, 不保留参考数据的计算图,
也不影响 `kin.forward` 对世界模型预测状态的梯度反传. 计算和缓存均在 `device` 上,
`kin` 必须使用相同设备和 float32. 批大小限制临时计算量, 不改变完整缓存的大小.
身体线速度对应连杆原点而非质心. 缓存为原始参考世界系状态, 奖励所需的坐标对齐另行处理.

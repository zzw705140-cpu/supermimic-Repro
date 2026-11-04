"""G1 参考动作库, 负责 CSV 加载, 时间重采样和参考速度计算.

原始 CSV 不会被修改. 动作采样策略, 身体正运动学和交互经验池由其他模块负责.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch


class MotionLib:
    """加载整个动作目录, 在内存中建立可按动作和帧索引的参考状态库.

    Args: motion_dir 为 CSV 目录, 默认使用同级 lafan1_g1/. source_fps 为原始帧率,
    control_dt 为控制周期秒数, 必须与环境一致. device 为输出张量所在设备.
    输入 CSV: [T, 36], 每行为根部 xyz, 根部 xyzw 四元数和 29 个关节角.
    输出状态字典: root_pos/root_lin_vel/root_ang_vel [..., 3], root_quat [..., 4],
    joint_pos/joint_vel [..., 29]. 位置为米, 角度为弧度, 速度单位为每秒.
    根部位置, 姿态及线/角速度均在参考轨迹世界坐标系中, 四元数统一为 xyzw.
    单帧查询无前导维, 批量查询前导维为 B, 未来查询额外增加 K 维.
    关节顺序保留 CSV 顺序, 对应 robot.py 的 G1_JOINT_NAMES.
    """

    """---------------- ① 配置与初始化 ----------------"""

    def __init__(self, motion_dir: str | Path | None = None, source_fps: float = 30.0, control_dt: float = 0.02, device: str | torch.device = "cpu"):
        """逐段加载, 重采样和计算速度, 最后将参考状态转换为指定设备的 float32 张量."""
        self.motion_dir = Path(motion_dir).resolve() if motion_dir is not None else Path(__file__).resolve().parent / "lafan1_g1"
        self.source_fps, self.control_dt = float(source_fps), float(control_dt)
        self.device = torch.device(device)
        if not np.isfinite(self.source_fps) or self.source_fps <= 0:
            raise ValueError("source_fps 必须为正的有限数值")
        if not np.isfinite(self.control_dt) or self.control_dt <= 0:
            raise ValueError("control_dt 必须为正的有限数值")
        if not self.motion_dir.is_dir():
            raise FileNotFoundError(f"动作目录不存在: {self.motion_dir}")

        self.files = sorted(self.motion_dir.glob("*.csv"))
        if not self.files:
            raise FileNotFoundError(f"动作目录中没有 CSV: {self.motion_dir}")
        self.names = [path.stem for path in self.files]
        self.num_clips = len(self.files)
        clips, lengths = [], []

        # 每段动作独立处理, 不在两个 CSV 的边界插值或求导.
        for path in self.files:
            root_pos, root_quat, joint_pos = self._load_csv(path)
            root_pos, root_quat, joint_pos = self._resample(root_pos, root_quat, joint_pos)
            root_lin_vel, root_ang_vel, joint_vel = self._compute_velocities(root_pos, root_quat, joint_pos)
            clips.append({"root_pos": root_pos, "root_quat": root_quat, "joint_pos": joint_pos, "root_lin_vel": root_lin_vel, "root_ang_vel": root_ang_vel, "joint_vel": joint_vel})
            lengths.append(len(root_pos))

        # 状态沿时间维连续存储, 用每段的起止索引保持动作边界.
        self.clip_lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
        self.clip_starts = torch.cumsum(self.clip_lengths, dim=0) - self.clip_lengths
        self.clip_ends = self.clip_starts + self.clip_lengths - 1  # 包含最后一帧.
        self.num_frames = sum(lengths)
        self.states = {key: torch.tensor(np.concatenate([clip[key] for clip in clips]), dtype=torch.float32, device=self.device) for key in clips[0]}

    """---------------- ② 读取原始动作 ----------------"""

    def _load_csv(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """读取 [T, 36] 数值表并检查格式, 返回位置 [T, 3], 单位四元数 [T, 4], 关节角 [T, 29]."""
        raw = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
        if raw.shape[1] != 36 or raw.shape[0] < 2:
            raise ValueError(f"{path.name} 必须至少有 2 帧且每帧为 36 列, 实际为 {raw.shape}")
        if not np.isfinite(raw).all():
            raise ValueError(f"{path.name} 包含 NaN 或 Inf")

        root_pos, root_quat, joint_pos = raw[:, :3], raw[:, 3:7].copy(), raw[:, 7:36]
        norm = np.linalg.norm(root_quat, axis=1, keepdims=True)
        if np.any(norm < 1e-8):
            raise ValueError(f"{path.name} 包含零四元数")
        root_quat /= norm

        # q 与 -q 表示同一旋转, 统一相邻帧符号以避免表示上的跳变.
        signs = np.where(np.sum(root_quat[1:] * root_quat[:-1], axis=1) < 0, -1.0, 1.0)
        root_quat[1:] *= np.cumprod(signs)[:, None]
        return root_pos, root_quat, joint_pos

    """---------------- ③ 时间重采样 ----------------"""

    def _resample(self, root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """将 T 帧重采样为 M 帧, 保持原时间尺度, 返回 [M, 3], [M, 4], [M, 29]."""
        source_times = np.arange(len(root_pos), dtype=np.float64) / self.source_fps
        num_intervals = int(np.floor(source_times[-1] / self.control_dt + 1e-9))
        if num_intervals < 1:
            raise ValueError("动作时长小于 control_dt, 无法生成至少两帧参考数据")
        # 只生成原时间范围内的等间隔采样点, 不强行追加非等间隔的末帧.
        target_times = np.minimum(np.arange(num_intervals + 1) * self.control_dt, source_times[-1])

        # 平移和关节角线性插值, 四元数用最短路径球面插值.
        source_indices = np.clip(np.floor(target_times * self.source_fps).astype(np.int64), 0, len(root_pos) - 2)
        weight = np.clip((target_times - source_times[source_indices]) * self.source_fps, 0.0, 1.0)[:, None]
        position = root_pos[source_indices] * (1 - weight) + root_pos[source_indices + 1] * weight
        joints = joint_pos[source_indices] * (1 - weight) + joint_pos[source_indices + 1] * weight
        quaternion = Slerp(source_times, Rotation.from_quat(root_quat))(target_times).as_quat()
        return position, quaternion, joints

    """---------------- ④ 计算参考速度 ----------------"""

    def _compute_velocities(self, root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """对重采样轨迹求导, 返回线速度 [M, 3], 世界系角速度 [M, 3], 关节速度 [M, 29]."""
        # 内部帧中心差分, 首尾帧使用单边差分, 不连接其他动作.
        root_lin_vel = np.gradient(root_pos, self.control_dt, axis=0)
        joint_vel = np.gradient(joint_pos, self.control_dt, axis=0)

        # 用相对旋转的旋转向量计算角速度, 不能直接对四元数分量求差分.
        rotations = Rotation.from_quat(root_quat)
        root_ang_vel = np.empty_like(root_pos)
        root_ang_vel[0] = (rotations[1] * rotations[0].inv()).as_rotvec() / self.control_dt
        root_ang_vel[-1] = (rotations[-1] * rotations[-2].inv()).as_rotvec() / self.control_dt
        if len(root_pos) > 2:
            root_ang_vel[1:-1] = (rotations[2:] * rotations[:-2].inv()).as_rotvec() / (2 * self.control_dt)
        return root_lin_vel, root_ang_vel, joint_vel

    """---------------- ⑤ 对外提供参考数据 ----------------"""

    def _global_indices(self, clip_id, frame_id) -> torch.Tensor:
        """检查动作编号和段内帧号, 支持整数标量或 [B] 批量输入, 返回全局帧索引."""
        clip = torch.as_tensor(clip_id, device=self.device)
        frame = torch.as_tensor(frame_id, device=self.device)
        for name, index in (("clip_id", clip), ("frame_id", frame)):
            if index.ndim > 1 or index.dtype == torch.bool or index.is_floating_point() or index.is_complex():
                raise ValueError(f"{name} 必须为整数标量或一维整数数组")
        clip, frame = torch.broadcast_tensors(clip.long(), frame.long())
        if torch.any((clip < 0) | (clip >= self.num_clips)):
            raise IndexError("clip_id 超出动作编号范围")
        if torch.any((frame < 0) | (frame >= self.clip_lengths[clip])):
            raise IndexError("frame_id 超出该动作的帧数范围")
        return self.clip_starts[clip] + frame

    def get_reference(self, clip_id, frame_id) -> dict[str, torch.Tensor]:
        """返回指定帧的状态副本, 标量索引输出 [D], 批量索引输出 [B, D], 字段见类说明."""
        indices = self._global_indices(clip_id, frame_id)
        return {key: value[indices].clone() for key, value in self.states.items()}

    def get_future_reference(self, clip_id, frame_id, num_frames: int) -> dict[str, torch.Tensor]:
        """返回下一帧起 K=num_frames 帧, 单样本 [K, D], 批量 [B, K, D].

        超过动作末尾时保持末帧姿态并将速度置零, 不进入下一段动作.
        最后一帧原有的参考速度保留, 仅超出时长的合成帧速度为零.
        """
        if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0:
            raise ValueError("num_frames 必须为正整数")
        current = self._global_indices(clip_id, frame_id)
        clip = torch.as_tensor(clip_id, dtype=torch.long, device=self.device)
        end = self.clip_ends[clip][..., None]
        requested = current[..., None] + torch.arange(1, num_frames + 1, device=self.device)
        indices = torch.minimum(requested, end)
        reference = {key: value[indices].clone() for key, value in self.states.items()}
        for key in ("root_lin_vel", "root_ang_vel", "joint_vel"):
            reference[key] = torch.where((requested <= end)[..., None], reference[key], 0.0)
        return reference

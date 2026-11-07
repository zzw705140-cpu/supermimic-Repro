#########################################################################
#      加载 CSV, 重采样, 计算根部与关节速度, 可分批预计算并缓存参考身体状态.      #
#########################################################################

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch

if TYPE_CHECKING:
    from .kinematics import G1Kinematics


BASE_STATE_FIELDS = ("root_pos", "root_quat", "root_lin_vel", "root_ang_vel", "joint_pos", "joint_vel")


class MotionLib:
    """
    加载并重采样参考动作,计算速度,按动作编号和帧号提供参考状态.

    输入:
        CSV 动作数据 [T, 36]:根部位置,xyzw 四元数和 29 个关节角.
        查询索引:单样本 clip_id [],frame_id [];批量样本 clip_id [B],frame_id [B].

    输出:
        根部位置,线速度和角速度:    [3] 或 [B, 3].
        根部 xyzw 四元数:         [4] 或 [B, 4].
        关节角和关节速度:          [29] 或 [B, 29].
        可选身体位置,线速度和角速度: [N, 3] 或 [B, N, 3].
        可选身体旋转矩阵:          [N, 3, 3] 或 [B, N, 3, 3].
        未来 K 帧查询:增加时间维,如 [K, 3] 或 [B, K, 3].

    T 为原始帧数,B 为批量大小,N 为跟踪部位数;根部和身体状态均使用世界系.
    """

    """---------------- ① 配置与初始化 ----------------"""

    # 保存配置，加载动作并建立状态缓存。
    def __init__(self, motion_dir: str | Path | None = None, source_fps: float = 30.0, control_dt: float = 0.02, device: str | torch.device = "cpu", *, kinematics: G1Kinematics | None = None, fk_batch_size: int = 8192):

        # 保存动作目录、原始帧率和控制周期。
        self.motion_dir = Path(motion_dir).resolve() if motion_dir is not None else Path(__file__).resolve().parent / "lafan1_g1"
        self.source_fps, self.control_dt = float(source_fps), float(control_dt)
        self.device = torch.device(device)

        # 检查时间参数、批大小和目录。
        if not np.isfinite(self.source_fps) or self.source_fps <= 0:
            raise ValueError("source_fps 必须为正的有限数值")
        if not np.isfinite(self.control_dt) or self.control_dt <= 0:
            raise ValueError("control_dt 必须为正的有限数值")
        if isinstance(fk_batch_size, bool) or not isinstance(fk_batch_size, int) or fk_batch_size <= 0:
            raise ValueError("fk_batch_size 必须为正整数")
        if not self.motion_dir.is_dir():
            raise FileNotFoundError(f"动作目录不存在: {self.motion_dir}")

        # 按文件名排序，确定动作编号。
        self.files = sorted(self.motion_dir.glob("*.csv"))
        if not self.files:
            raise FileNotFoundError(f"动作目录中没有 CSV: {self.motion_dir}")
        self.names = [path.stem for path in self.files]
        self.num_clips = len(self.files)
        clips, lengths = [], []

        # 逐段读取、重采样并计算速度，保持动作边界独立。
        for path in self.files:
            root_pos, root_quat, joint_pos = self._load_csv(path)
            root_pos, root_quat, joint_pos = self._resample(root_pos, root_quat, joint_pos)
            root_lin_vel, root_ang_vel, joint_vel = self._compute_velocities(root_pos, root_quat, joint_pos)
            clips.append({"root_pos": root_pos, "root_quat": root_quat, "joint_pos": joint_pos, "root_lin_vel": root_lin_vel, "root_ang_vel": root_ang_vel, "joint_vel": joint_vel})
            lengths.append(len(root_pos))

        # 记录每段动作的帧数及起止索引。
        self.clip_lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
        self.clip_starts = torch.cumsum(self.clip_lengths, dim=0) - self.clip_lengths
        self.clip_ends = self.clip_starts + self.clip_lengths - 1  # 包含最后一帧.

        # 合并所有动作，创建 float32 张量缓存。
        self.num_frames = sum(lengths)
        self.states = {key: torch.tensor(np.concatenate([clip[key] for clip in clips]), dtype=torch.float32, device=self.device) for key in clips[0]}

        # 提供运动学对象时，额外缓存身体状态。
        self.tracked_bodies = []
        if kinematics is not None:
            self._cache_body_states(kinematics, fk_batch_size)

    """---------------- ② 读取原始动作 ----------------"""

    # 读取 CSV，拆分状态并归一化四元数。
    def _load_csv(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        raw = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
        if raw.shape[1] != 36 or raw.shape[0] < 2:
            raise ValueError(f"{path.name} 必须至少有 2 帧且每帧为 36 列, 实际为 {raw.shape}")
        if not np.isfinite(raw).all():
            raise ValueError(f"{path.name} 包含 NaN 或 Inf")

        # 拆分根部位置、姿态和关节角。
        root_pos, root_quat, joint_pos = raw[:, :3], raw[:, 3:7].copy(), raw[:, 7:36]

        # 归一化四元数，排除零四元数。
        norm = np.linalg.norm(root_quat, axis=1, keepdims=True)
        if np.any(norm < 1e-8):
            raise ValueError(f"{path.name} 包含零四元数")
        root_quat /= norm

        # q 与 -q 表示同一旋转，统一相邻帧符号。
        signs = np.where(np.sum(root_quat[1:] * root_quat[:-1], axis=1) < 0, -1.0, 1.0)
        root_quat[1:] *= np.cumprod(signs)[:, None]
        return root_pos, root_quat, joint_pos

    """---------------- ③ 时间重采样 ----------------"""

    # 按环境控制周期重采样动作。
    def _resample(self, root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        # 计算原始帧时刻和目标采样间隔数。
        source_times = np.arange(len(root_pos), dtype=np.float64) / self.source_fps
        num_intervals = int(np.floor(source_times[-1] / self.control_dt + 1e-9))
        if num_intervals < 1:
            raise ValueError("动作时长小于 control_dt, 无法生成至少两帧参考数据")
        # 按控制周期生成采样时刻，不超出原动作时长。
        target_times = np.minimum(np.arange(num_intervals + 1) * self.control_dt, source_times[-1])

        # 计算插值区间和权重。
        source_indices = np.clip(np.floor(target_times * self.source_fps).astype(np.int64), 0, len(root_pos) - 2)
        weight = np.clip((target_times - source_times[source_indices]) * self.source_fps, 0.0, 1.0)[:, None]

        # 位置和关节角用线性插值，姿态用球面插值。
        position = root_pos[source_indices] * (1 - weight) + root_pos[source_indices + 1] * weight
        joints = joint_pos[source_indices] * (1 - weight) + joint_pos[source_indices + 1] * weight
        quaternion = Slerp(source_times, Rotation.from_quat(root_quat))(target_times).as_quat()
        return position, quaternion, joints

    """---------------- ④ 计算参考速度 ----------------"""

    # 根据重采样轨迹计算根部和关节速度。
    def _compute_velocities(self, root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        # 中间帧用中心差分，首尾帧用单边差分。
        root_lin_vel = np.gradient(root_pos, self.control_dt, axis=0)
        joint_vel = np.gradient(joint_pos, self.control_dt, axis=0)

        # 根据相对旋转计算世界系角速度。
        rotations = Rotation.from_quat(root_quat)
        root_ang_vel = np.empty_like(root_pos)
        root_ang_vel[0] = (rotations[1] * rotations[0].inv()).as_rotvec() / self.control_dt
        root_ang_vel[-1] = (rotations[-1] * rotations[-2].inv()).as_rotvec() / self.control_dt
        if len(root_pos) > 2:
            root_ang_vel[1:-1] = (rotations[2:] * rotations[:-2].inv()).as_rotvec() / (2 * self.control_dt)
        return root_lin_vel, root_ang_vel, joint_vel

    """---------------- ⑤ 分批计算并缓存参考身体状态 ----------------"""

    # 分批计算并缓存身体状态，不保存梯度。
    @torch.no_grad()
    def _cache_body_states(self, kinematics: G1Kinematics, batch_size: int) -> None:

        # 检查运动学对象的设备、精度和关节数量。
        sample = self.states["root_pos"]
        if kinematics.offset_pos.device != sample.device or kinematics.dtype != sample.dtype:
            raise ValueError("kinematics 必须与 MotionLib 使用相同设备和 float32 dtype")
        if kinematics.num_joints != self.states["joint_pos"].shape[-1]:
            raise ValueError("kinematics 的关节数量必须与 CSV 的 29 个关节一致")
        cached = {}

        # 分批取出基础状态，计算身体位姿与速度。
        for start in range(0, self.num_frames, batch_size):
            stop = min(start + batch_size, self.num_frames)
            inputs = {key: self.states[key][start:stop] for key in BASE_STATE_FIELDS}
            bodies = kinematics.forward(**inputs)
            for key, value in bodies.items():
                # 首批分配完整缓存，再逐批写入。
                if start == 0:
                    cached[key] = torch.empty((self.num_frames,) + value.shape[1:], dtype=value.dtype, device=value.device)
                cached[key][start:stop] = value

        # 合并身体状态，保存跟踪部位顺序。
        self.states.update(cached)
        self.tracked_bodies = list(kinematics.tracked_bodies)

    """---------------- ⑥ 对外提供参考数据 ----------------"""

    # 检查动作编号和帧号，转换为全局索引。
    def _global_indices(self, clip_id, frame_id) -> torch.Tensor:

        clip = torch.as_tensor(clip_id, device=self.device)
        frame = torch.as_tensor(frame_id, device=self.device)
        for name, index in (("clip_id", clip), ("frame_id", frame)):
            if index.ndim > 1 or index.dtype == torch.bool or index.is_floating_point() or index.is_complex():
                raise ValueError(f"{name} 必须为整数标量或一维整数数组")

        # 对齐标量或批量索引，并检查取值范围。
        clip, frame = torch.broadcast_tensors(clip.long(), frame.long())
        if torch.any((clip < 0) | (clip >= self.num_clips)):
            raise IndexError("clip_id 超出动作编号范围")
        if torch.any((frame < 0) | (frame >= self.clip_lengths[clip])):
            raise IndexError("frame_id 超出该动作的帧数范围")
        return self.clip_starts[clip] + frame

    # 返回指定帧的状态副本，include_body 控制是否包含身体状态。
    def get_reference(self, clip_id, frame_id, *, include_body: bool = True) -> dict[str, torch.Tensor]:
        indices = self._global_indices(clip_id, frame_id)
        fields = self.states if include_body else BASE_STATE_FIELDS
        return {key: self.states[key][indices].clone() for key in fields}

    # 返回下一帧起的 num_frames 帧参考状态。
    def get_future_reference(self, clip_id, frame_id, num_frames: int, *, include_body: bool = True) -> dict[str, torch.Tensor]:
        # 超出动作末尾时保持末帧位姿，仅将超出部分的速度置零。
        if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0:
            raise ValueError("num_frames 必须为正整数")

        # 生成未来帧索引，并限制在当前动作内。
        current = self._global_indices(clip_id, frame_id)
        clip = torch.as_tensor(clip_id, dtype=torch.long, device=self.device)
        end = self.clip_ends[clip][..., None]
        requested = current[..., None] + torch.arange(1, num_frames + 1, device=self.device)
        indices = torch.minimum(requested, end)

        # 按需提取基础状态和已缓存的身体状态。
        fields = self.states if include_body else BASE_STATE_FIELDS
        reference = {key: self.states[key][indices].clone() for key in fields}

        for key in ("root_lin_vel", "root_ang_vel", "joint_vel", "body_lin_vel", "body_ang_vel"):
            if key in reference:
                # 扩展掩码维度，将超出末帧的速度置零。
                valid = (requested <= end).reshape(requested.shape + (1,) * (reference[key].ndim - requested.ndim))
                reference[key] = torch.where(valid, reference[key], 0.0)
        return reference

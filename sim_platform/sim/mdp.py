"""G1 动作跟踪任务的解析计算, 供物理仿真和可微世界模型共同调用."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch

from sim_platform.assets.train_data.kinematics import G1Kinematics
from sim_platform.assets.train_data.motion import BASE_STATE_FIELDS, MotionLib
from sim_platform.wrappers.rotations import heading_matrix, matrix_to_rotation_6d, quaternion_to_matrix, rotation_error_squared

##############################################################################
#                         任务配置与公共状态工具                              #
##############################################################################

NUM_JOINTS = 29                      # 机器人关节数
STATE_DIM = 71                       # 完整状态维度
OBS_DIM = 99                         # 策略观测维度
GOAL_FRAME_DIM = 73                  # 单帧参考目标维度
INVARIANT_DIM = 71                   # 去除绝对水平位置和朝向后的特征维度
CONTEXT_DIM = 33                     # 摩擦系数、关节目标偏移和质心偏移的总维度
STATE_WIDTHS = (3, 4, 3, 3, 29, 29)  # 根部位置、四元数、线速度、角速度、关节角、关节速度
TRACKING_TERMS = ("body_pos", "body_ori", "body_lin_vel", "body_ang_vel", "root_pos", "root_ori")
# 六项跟踪误差, 依次为身体位置、姿态, 身体线速度、角速度, 根部位置、姿态; 乘权重后加总为奖励

@dataclass(frozen=True)
# 保存任务配置，包括跟踪部位、奖励权重、终止阈值、观测噪声等，并检查参数是否合法
class TrackingMDPConfig:
    """任务参数, 默认数值和跟踪部位沿用 vibed; 噪声为均匀分布的半宽.

    tracking_stds/weights 顺序: 身体位置, 姿态, 线速度, 角速度, 根部位置, 姿态.
    observation_noise     顺序: 关节角, 关节速度, 根部角速度, 重力, 上一动作, 相对姿态.
    """

    tracked_bodies: tuple[str, ...] = (
        "pelvis",                      # 骨盆
        "left_hip_roll_link",          # 左髋侧摆连杆
        "left_knee_link",              # 左膝连杆
        "left_ankle_roll_link",        # 左踝侧摆连杆
        "right_hip_roll_link",         # 右髋侧摆连杆
        "right_knee_link",             # 右膝连杆
        "right_ankle_roll_link",       # 右踝侧摆连杆
        "torso_link",                  # 躯干
        "left_shoulder_roll_link",     # 左肩侧摆连杆
        "left_elbow_link",             # 左肘连杆
        "left_wrist_yaw_link",         # 左腕偏航连杆
        "right_shoulder_roll_link",    # 右肩侧摆连杆
        "right_elbow_link",            # 右肘连杆
        "right_wrist_yaw_link",        # 右腕偏航连杆
    )
    tracking_stds: tuple[float, ...] = (0.3, 0.4, 1.0, 3.14, 0.3, 0.4)        # 六项跟踪误差的容忍尺度, 顺序对应 TRACKING_TERMS
    tracking_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 0.5, 0.5)      # 六项跟踪奖励的权重, 顺序对应 TRACKING_TERMS
    action_rate_weight: float = -0.1                                          # 相邻动作变化的惩罚权重
    joint_limit_weight: float = -2.0                                          # 关节超出软限位的惩罚权重
    soft_limit_factor: float = 0.9                                            # 软限位范围占原关节范围的比例
    term_height: float = 0.25                                                 # 根部高度误差的终止阈值, 单位为米
    term_tilt: float = 0.8                                                    # 根部旋转矩阵 R[2, 2] 误差的终止阈值
    height_softness: float = 0.02                                             # 高度软终止的平滑尺度
    tilt_softness: float = 0.04                                               # 倾斜软终止的平滑尺度
    observation_noise: tuple[float, ...] = (0.01, 0.5, 0.2, 0.05, 0.0, 0.05)  # 均匀噪声半宽: 关节角、关节速度、根部角速度、重力、上一动作、相对姿态
    control_dt: float = 0.02                                                  # 控制周期, 单位为秒
    goal_frames: int = 10                                                     # 未来参考目标的帧数


    def __post_init__(self):
        """检查尺度, 权重和布局, 在进入计算前暴露无效配置."""

        for name in ("tracking_stds", "tracking_weights", "observation_noise"):
            values = getattr(self, name)
            if len(values) != 6 or any(not math.isfinite(value) for value in values):
                raise ValueError(f"{name} 必须包含 6 个有限数值")
        if any(value <= 0 for value in self.tracking_stds):
            raise ValueError("tracking_stds 必须为正数")
        if any(value < 0 for value in self.tracking_weights + self.observation_noise):
            raise ValueError("跟踪权重和噪声幅度不能为负")
        for name in ("control_dt", "term_height", "term_tilt", "height_softness", "tilt_softness"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为正的有限数值")
        if not 0 < self.soft_limit_factor <= 1:
            raise ValueError("soft_limit_factor 必须位于 (0,1]")
        if any(not math.isfinite(value) or value > 0 for value in (self.action_rate_weight, self.joint_limit_weight)):
            raise ValueError("惩罚权重必须为非正的有限数值")
        if not self.tracked_bodies or len(set(self.tracked_bodies)) != len(self.tracked_bodies):
            raise ValueError("tracked_bodies 必须非空且不重复")
        if isinstance(self.goal_frames, bool) or not isinstance(self.goal_frames, int) or self.goal_frames <= 0:
            raise ValueError("goal_frames 必须为正整数")


def _state_values(state: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """检查六字段状态的前导维, 设备和精度, 不复制或切断计算图."""

    values = tuple(state[key] for key in BASE_STATE_FIELDS)
    first = values[0]

    if not isinstance(first, torch.Tensor) or first.ndim < 1 or not first.is_floating_point():
        raise ValueError("状态字段必须为浮点张量")
    for key, width, value in zip(BASE_STATE_FIELDS, STATE_WIDTHS, values):
        if not isinstance(value, torch.Tensor) or value.shape != first.shape[:-1] + (width,):
            raise ValueError(f"{key} 的形状必须为 {first.shape[:-1] + (width,)}")
        if value.device != first.device or value.dtype != first.dtype:
            raise ValueError("所有状态字段必须使用相同设备和精度")
    return values


def flatten_state(state: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """按六字段顺序打包为 [...,71], 根部四元数保留 xyzw 顺序."""

    return torch.cat(_state_values(state), dim=-1)


def unflatten_state(value: torch.Tensor) -> dict[str, torch.Tensor]:
    """将 [...,71] 解包为六字段状态视图, 保留梯度; 修改视图会影响输入."""

    if value.shape[-1:] != (STATE_DIM,):
        raise ValueError("状态向量最后一维必须为 71")
    return dict(zip(BASE_STATE_FIELDS, value.split(STATE_WIDTHS, dim=-1)))


def invariant_features(state: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """去除绝对水平位置和朝向, 输出高度, 倾斜, 机体系速度和关节状态 [...,71]."""

    position, quaternion, linear, angular, joints, joint_velocity = _state_values(state)
    rotation = quaternion_to_matrix(quaternion)
    inverse = rotation.transpose(-1, -2)
    tilt = heading_matrix(rotation).transpose(-1, -2) @ rotation
    return torch.cat((position[..., 2:3], matrix_to_rotation_6d(tilt), (inverse @ linear[..., None]).squeeze(-1), (inverse @ angular[..., None]).squeeze(-1), joints, joint_velocity), dim=-1)


def normalize_context(context: torch.Tensor) -> torch.Tensor:
    """归一化原始 [...,33] 参数: 摩擦(1), 关节目标偏移(29), 躯干质心偏移(3).

    只转换数值, 不采样参数或修改仿真; 比例与 vibed 一致.
    """

    if context.shape[-1:] != (CONTEXT_DIM,):
        raise ValueError("随机化参数最后一维必须为 33")
    return torch.cat(((context[..., :1] - 0.75) / 0.45, context[..., 1:30] / 0.01, context[..., 30:] / 0.05), dim=-1)


##############################################################################
#                         G1 动作跟踪任务                                    #
##############################################################################


# 使用这些配置，计算策略观测、未来目标、奖励、终止条件和 Critic 特征，不负责推进仿真。
class TrackingMDP:
    """计算观测, 参考目标, 奖励和终止, 不创建场景或推进物理.

    状态:
        state 包含根部位置, 姿态, 线速度, 角速度, 以及关节角和关节速度.
        根部四元数按 xyzw 排列, 位置和速度用世界坐标系表示.
        并行环境中的位置要先减去各自的环境原点偏移.
        身体线速度指连杆坐标原点的速度, 不是质心速度.

    输入:
        支持单个状态或批量状态, clip_id/frame_id 可使用标量或匹配批量维度.
        action [...,29] 为无量纲控制量, context [...,33] 为原始物理参数.
        输入张量的设备和精度要与动作库一致.

    梯度与缓存:
        奖励和软终止支持对 state/action 反向传播.
        参考身体状态直接从缓存读取.

    用法:
        config = TrackingMDPConfig()
        mdp    = TrackingMDP(kinematics, motion, default_joint_pos, config)
        obs    = mdp.observation(state, previous_action, clip_id, frame_id, context)
        goal   = mdp.goal_features(clip_id, frame_id)
        reward = mdp.reward(next_state, clip_id, next_frame_id, action, previous_action)
    """

    def __init__(self, kinematics: G1Kinematics, motion: MotionLib, default_joint_pos: torch.Tensor, config: TrackingMDPConfig | None = None):
        """检查资源一致性, 准备软限位和噪声; 本类不重新加载模型或预计算参考缓存."""

        self.config = config or TrackingMDPConfig()
        # 保存运动学对象和动作库，供后续计算使用
        self.kinematics, self.motion = kinematics, motion
        # 读取动作库的设备和数据精度，作为本类的统一标准
        self.device, self.dtype = motion.states["root_pos"].device, motion.states["root_pos"].dtype

        if kinematics.num_joints != NUM_JOINTS or kinematics.q_lo.device != self.device or kinematics.dtype != self.dtype:
            raise ValueError("正运动学必须有 29 个关节, 且与动作库同设备同精度")
        bodies = tuple(self.config.tracked_bodies)
        if tuple(kinematics.tracked_bodies) != bodies or tuple(motion.tracked_bodies) != bodies:
            raise ValueError("配置, 正运动学与动作缓存的跟踪部位及顺序必须一致")
        if not all(key in motion.states for key in ("body_pos", "body_rot", "body_lin_vel", "body_ang_vel")):
            raise ValueError("请先向 MotionLib 传入 kinematics, 生成参考身体状态缓存")
        if not math.isclose(motion.control_dt, self.config.control_dt, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("动作库与任务的 control_dt 必须一致")
        self.default_joint_pos = torch.as_tensor(default_joint_pos, device=self.device, dtype=self.dtype).detach().clone()
        if self.default_joint_pos.shape != (NUM_JOINTS,) or not torch.isfinite(self.default_joint_pos).all():
            raise ValueError("default_joint_pos 必须包含 29 个有限关节角, 顺序与 CSV 一致")
        if not torch.isfinite(kinematics.q_lo).all() or not torch.isfinite(kinematics.q_hi).all():
            raise ValueError("G1 的关节限位必须为有限数值")

        ############################################
        #     提前准备后面计算要用的范围、噪声和维度     #
        ############################################

        # 围绕关节范围的中点，缩小允许范围，得到软限位。比如原范围为 [-1, 1]，比例为 0.9，软限位就是 [-0.9, 0.9]；超出后产生惩罚
        middle = (kinematics.q_lo + kinematics.q_hi) / 2
        half_range = (kinematics.q_hi - kinematics.q_lo) * self.config.soft_limit_factor / 2
        self.soft_low, self.soft_high = middle - half_range, middle + half_range

        # 将六类噪声幅度展开，为观测的每一维分配噪声幅度。
        self.obs_noise = torch.tensor(self.config.observation_noise, device=self.device, dtype=self.dtype).repeat_interleave(torch.tensor((29, 29, 3, 3, 29, 6), device=self.device))
        # 计算单步奖励的理论上限：跟踪完全准确，且没有惩罚。
        self.max_step_reward = sum(self.config.tracking_weights) * self.config.control_dt
        # 计算未来目标总维度：每帧 73 维，默认 10 帧，共 730 维。
        self.goal_dim = GOAL_FRAME_DIM * self.config.goal_frames
        # 计算 Critic 特征总维度：状态特征、关节角误差、身体位置误差、根部位置误差、相对姿态和物理参数，默认共 184 维
        self.critic_dim = INVARIANT_DIM + NUM_JOINTS + 3 * len(bodies) + 3 + 6 + CONTEXT_DIM

    # ---------------------------------- 策略观测与目标 ---------------------------------- #

    def observation(self, state, previous_action, clip_id, frame_id, context=None) -> torch.Tensor:
        """
        构造无噪声 [...,99] 观测; 上一动作仅作为输入信息, 此函数不执行动作.
        
        策略观测组成:
        关节角偏移    (29),
        关节速度      (29),
        根部角速度    (3),
        重力方向      (3),
        上一动作      (29),
        相对参考姿态  (6)   
        """

        self._check_state(state)
        previous_action = self._input(previous_action, state, NUM_JOINTS, "previous_action")
        context = self._context(context, state)
        reference = self._reference(clip_id, frame_id, state["root_pos"].shape[:-1], include_body=False)
        rotation = quaternion_to_matrix(state["root_quat"])
        inverse = rotation.transpose(-1, -2)
        relative = inverse @ quaternion_to_matrix(reference["root_quat"])
        joint_offset = state["joint_pos"] - self.default_joint_pos - context[..., 1:30]
        angular = (inverse @ state["root_ang_vel"][..., None]).squeeze(-1)
        gravity = -rotation[..., 2, :]
        return torch.cat((joint_offset, state["joint_vel"], angular, gravity, previous_action, matrix_to_rotation_6d(relative)), dim=-1)

    # 按配置的幅度，给观测各维加入均匀随机噪声
    def add_observation_noise(self, observation: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """在原始观测尺度添加均匀噪声, 不原地修改输入; 评估时由调用方跳过."""

        if observation.shape[-1:] != (OBS_DIM,) or observation.device != self.device or observation.dtype != self.dtype:
            raise ValueError("观测必须为同设备同精度的 [...,99] 张量")
        noise = torch.rand(observation.shape, device=self.device, dtype=self.dtype, generator=generator)
        return observation + (2 * noise - 1) * self.obs_noise

    # 将未来多帧参考动作编码并拼接，默认得到 730 维目标，告诉策略“接下来应该怎样运动”
    def goal_features(self, clip_id, frame_id, num_frames: int | None = None) -> torch.Tensor:
        """
        从下一参考帧起编码 K 帧, 返回 [...,K*73], 超尾速度沿用动作库的置零规则.
        这里的位移相对于当前参考帧的根部位置，不是实际机器人位置
        位移和姿态用当前参考的水平朝向系表达，速度用未来参考帧自身的机体系表达。
        
        单帧目标组成：
        根部相对当前位置的参考位移   (3),
        根部参考姿态，6D 表示       (6),
        根部参考线速度             (3),
        根部参考角速度             (3),
        参考关节角                (29),
        参考关节速度 × 0.1        (29)
        """

        count = self.config.goal_frames if num_frames is None else num_frames
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("num_frames 必须为正整数")
        clip, frame = self._indices(clip_id, frame_id)
        leading = clip.shape
        reference = self._reference(clip, frame, leading, include_body=False)
        future = self.motion.get_future_reference(clip.reshape(-1), frame.reshape(-1), count, include_body=False)
        future = {key: value.reshape(leading + value.shape[1:]) for key, value in future.items()}
        heading_inverse = heading_matrix(quaternion_to_matrix(reference["root_quat"])).transpose(-1, -2).unsqueeze(-3)
        rotation = quaternion_to_matrix(future["root_quat"])
        inverse = rotation.transpose(-1, -2)
        displacement = (heading_inverse @ (future["root_pos"] - reference["root_pos"].unsqueeze(-2))[..., None]).squeeze(-1)
        linear = (inverse @ future["root_lin_vel"][..., None]).squeeze(-1)
        angular = (inverse @ future["root_ang_vel"][..., None]).squeeze(-1)
        features = torch.cat((displacement, matrix_to_rotation_6d(heading_inverse @ rotation), linear, angular, future["joint_pos"], 0.1 * future["joint_vel"]), dim=-1)
        return features.reshape(leading + (count * GOAL_FRAME_DIM,))

    # ---------------------------------- 跟踪奖励 ---------------------------------- #

    def _align_reference(self, state, reference):
        """按实际根部 XY 与相对水平朝向对齐参考身体位姿, 保留参考根部高度."""

        rotation = quaternion_to_matrix(state["root_quat"])
        reference_rotation = quaternion_to_matrix(reference["root_quat"])
        heading_delta = heading_matrix(rotation @ reference_rotation.transpose(-1, -2))
        anchor = torch.cat((state["root_pos"][..., :2], reference["root_pos"][..., 2:3]), dim=-1)
        relative_position = reference["body_pos"] - reference["root_pos"].unsqueeze(-2)
        position = anchor.unsqueeze(-2) + (heading_delta.unsqueeze(-3) @ relative_position[..., None]).squeeze(-1)
        orientation = heading_delta.unsqueeze(-3) @ reference["body_rot"]
        return position, orientation

    # 根据跟踪误差计算奖励，再扣除动作变化和关节超限惩罚
    def reward(self, state, clip_id, frame_id, action, previous_action, *, return_terms: bool = False):
        """计算 [...]-形状的单步奖励, 已乘 control_dt; 输入为动作执行后的状态和参考帧.

        身体位姿先对齐, 身体速度直接在世界系比较, 根部仍使用原参考位姿.
        return_terms=True 时返回 (总奖励, 分项字典), 奖励项已加权乘 dt, error_* 为原始误差.
        """

        self._check_state(state)
        action = self._input(action, state, NUM_JOINTS, "action")
        previous_action = self._input(previous_action, state, NUM_JOINTS, "previous_action")
        reference = self._reference(clip_id, frame_id, state["root_pos"].shape[:-1])
        bodies = self.kinematics.forward(**{key: state[key] for key in BASE_STATE_FIELDS})
        aligned_position, aligned_rotation = self._align_reference(state, reference)
        errors = (
            (aligned_position - bodies["body_pos"]).square().sum(-1).mean(-1),
            rotation_error_squared(bodies["body_rot"], aligned_rotation).mean(-1),
            (reference["body_lin_vel"] - bodies["body_lin_vel"]).square().sum(-1).mean(-1),
            (reference["body_ang_vel"] - bodies["body_ang_vel"]).square().sum(-1).mean(-1),
            (reference["root_pos"] - state["root_pos"]).square().sum(-1),
            rotation_error_squared(quaternion_to_matrix(state["root_quat"]), quaternion_to_matrix(reference["root_quat"])),
        )
        config = self.config
        terms = {name: weight * torch.exp(-error / scale**2) * config.control_dt for name, error, scale, weight in zip(TRACKING_TERMS, errors, config.tracking_stds, config.tracking_weights)}
        action_rate = (action - previous_action).square().sum(-1)
        joint_limit = (torch.relu(state["joint_pos"] - self.soft_high) + torch.relu(self.soft_low - state["joint_pos"])).sum(-1)
        terms["action_rate"] = config.action_rate_weight * action_rate * config.control_dt
        terms["joint_limit"] = config.joint_limit_weight * joint_limit * config.control_dt
        total = sum(terms.values())
        if return_terms:
            terms.update({f"error_{name}": error for name, error in zip(TRACKING_TERMS, errors)})
            return total, terms
        return total

    # ---------------------------------- 硬终止与软终止 ---------------------------------- #

    def _termination_errors(self, state, clip_id, frame_id):
        """计算高度绝对差及 R[2,2] 绝对差; 后者不是姿态夹角."""

        self._check_state(state)
        reference = self._reference(clip_id, frame_id, state["root_pos"].shape[:-1], include_body=False)
        rotation = quaternion_to_matrix(state["root_quat"])
        reference_rotation = quaternion_to_matrix(reference["root_quat"])
        height = (state["root_pos"][..., 2] - reference["root_pos"][..., 2]).abs()
        tilt = (rotation[..., 2, 2] - reference_rotation[..., 2, 2]).abs()
        return height, tilt

    # 上述任一误差超过阈值，就判定跟踪失败
    def termination(self, state, clip_id, frame_id) -> torch.Tensor:
        """返回跟踪失败的布尔标志 [...]; 不处理动作结束或回合超时."""

        height, tilt = self._termination_errors(state, clip_id, frame_id)
        return (height > self.config.term_height) | (tilt > self.config.term_tilt)

    # 用平滑函数将上述误差转换为 0～1 的终止强度
    def soft_termination(self, state, clip_id, frame_id) -> torch.Tensor:
        """返回 [0,1] 范围的可微终止强度 [...], 用于模型内预测, 并非学习出的概率."""

        height, tilt = self._termination_errors(state, clip_id, frame_id)
        height_alive = torch.sigmoid((self.config.term_height - height) / self.config.height_softness)
        tilt_alive = torch.sigmoid((self.config.term_tilt - tilt) / self.config.tilt_softness)
        return 1 - height_alive * tilt_alive

    # ---------------------------------- Critic 特征 ---------------------------------- #

    # 拼接状态特征、跟踪误差和物理参数，作为 Critic 的输入信息
    def critic_features(self, state, clip_id, frame_id, context=None) -> torch.Tensor:
        """拼接不变状态, 关节/身体/根部跟踪误差, 相对姿态和归一化 context.

        输出 [...,critic_dim], 默认 14 个跟踪部位时为 184 维; 未来目标由调用方另外拼接.
        context 接收原始物理参数, 本函数内部归一化, 不应提前调用 normalize_context.

        Critic 特征组成：      共184 维：
        根部高度                  (1),
        去除水平朝向后的根部姿态     (6),
        机体系根部线速度            (3),
        机体系根部角速度            (3),
        关节角                    (29),
        关节速度                   (29),

        参考关节角与实际关节角之差   (29),
        对齐后的身体位置误差        (14 × 3 = 42),
        根部位置误差               (3),
        相对参考根部姿态           (6),

        归一化物理参数             (33)
        """

        self._check_state(state)
        reference = self._reference(clip_id, frame_id, state["root_pos"].shape[:-1])
        rotation = quaternion_to_matrix(state["root_quat"])
        heading_inverse = heading_matrix(rotation).transpose(-1, -2)
        bodies = self.kinematics.forward(**{key: state[key] for key in BASE_STATE_FIELDS})
        aligned_position, _ = self._align_reference(state, reference)
        body_error = (heading_inverse.unsqueeze(-3) @ (aligned_position - bodies["body_pos"])[..., None]).squeeze(-1).flatten(-2)
        root_error = (heading_inverse @ (reference["root_pos"] - state["root_pos"])[..., None]).squeeze(-1)
        relative = rotation.transpose(-1, -2) @ quaternion_to_matrix(reference["root_quat"])
        return torch.cat((invariant_features(state), reference["joint_pos"] - state["joint_pos"], body_error, root_error, matrix_to_rotation_6d(relative), normalize_context(self._context(context, state))), dim=-1)

    # ---------------------------------- 输入布局与参考查询 ---------------------------------- #

    # 检查状态的形状、设备和精度是否正确
    def _check_state(self, state):
        """校验布局及与任务资源一致的设备/精度; 不复制状态或分离梯度."""

        values = _state_values(state)
        if values[0].device != self.device or values[0].dtype != self.dtype:
            raise ValueError("状态必须与任务资源使用相同设备和精度")

    # 检查动作或物理参数是否与状态的批量维、设备和精度一致。
    def _input(self, value, state, width, name):
        """检查动作或参数与状态具有相同前导维, 防止静默广播."""

        expected = state["root_pos"].shape[:-1] + (width,)
        if not isinstance(value, torch.Tensor) or value.shape != expected or value.device != self.device or value.dtype != self.dtype:
            raise ValueError(f"{name} 必须为同设备同精度的 {expected} 张量")
        return value

    # 使用传入的物理参数; 未传入时, 摩擦系数为 1, 其余偏移为 0。
    def _context(self, context, state):
        """未提供时使用摩擦 1.0, 关节目标与质心偏移为零的固定参数."""

        if context is not None:
            return self._input(context, state, CONTEXT_DIM, "context")
        friction = torch.ones_like(state["root_pos"][..., :1])
        return torch.cat((friction, torch.zeros_like(state["joint_pos"]), torch.zeros_like(state["root_pos"])), dim=-1)

    # 将动作编号和帧号转换为整数张量, 并对齐批量维度。
    def _indices(self, clip_id, frame_id, leading_shape=None):
        """校验整数索引, 支持标量广播及世界模型展开的多维批量."""

        clip, frame = torch.as_tensor(clip_id, device=self.device), torch.as_tensor(frame_id, device=self.device)
        if any(value.dtype == torch.bool or value.is_floating_point() or value.is_complex() for value in (clip, frame)):
            raise ValueError("动作编号和帧号必须为整数")
        clip, frame = torch.broadcast_tensors(clip, frame)
        if leading_shape is not None:
            clip, frame = torch.broadcast_to(clip, leading_shape), torch.broadcast_to(frame, leading_shape)
        return clip.long(), frame.long()

    # 将索引展平后查询动作库, 再恢复原来的批量形状。
    def _reference(self, clip_id, frame_id, leading_shape, *, include_body=True):
        """将多维索引展平交给 MotionLib, 再恢复前导维; 不重算参考正运动学."""

        clip, frame = self._indices(clip_id, frame_id, leading_shape)
        reference = self.motion.get_reference(clip.reshape(-1), frame.reshape(-1), include_body=include_body)
        return {key: value.reshape(clip.shape + value.shape[1:]) for key, value in reference.items()}

"""SuperMimic 的网络与学习规则, 由 workflows 组织数据采集, 采样和预测展开.

本页定义配置, 世界模型, 策略, Critic 以及损失和参数更新, 不创建仿真或经验池.
状态沿用六字段字典, 四元数为 xyzw; context 始终接收未归一化的原始物理参数.
网络规模和优化目标参考论文表 A.4/A.5 及 vibed, 奖励与终止复用 TrackingMDP.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from sim_platform.sim.mdp import INVARIANT_DIM, TrackingMDP, invariant_features, normalize_context
from sim_platform.wrappers.rotations import heading_matrix, lambda_returns, quaternion_multiply, quaternion_to_matrix, quaternion_to_rotation_vector, rotation_vector_to_quaternion
from supermimic.data.validation import ACTION_DIM, CONTEXT_DIM, GOAL_FRAME_DIM, OBS_DIM, validate_imagined_rollout, validate_replay_data
from supermimic.networks.mlp import MLP, RunningNorm

##############################################################################
#                         网络与训练配置                                      #
##############################################################################

DYNAMICS_DIM = 70  # 位置, 旋转向量, 线速度, 角速度各 3 维, 关节角和速度各 29 维


@dataclass(frozen=True)
class SuperMimicConfig:
    """集中定义网络和学习参数, 启动脚本可覆盖, 不包含环境数量或日志路径.

    wm_window 为真实交互的监督窗口长度, imagination_horizon 为预测展开长度.
    policy_type/chunk_length/execute_steps 为预留的策略接口, 本版训练仅支持 mlp 和单步.
    critic_minibatches 是每轮 Critic 拆分的批次数, 不是每个批次的样本数量.
    """

    # 网络规模, 历史和参考目标布局.
    wm_hidden_dims: tuple[int, ...] = (2048, 2048, 2048)         # 世界模型的隐藏层宽度, 每个数字对应一层
    policy_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)  # 策略中参考动作编码器与动作解码器的隐藏层宽度
    critic_hidden_dims: tuple[int, ...] = (256, 128)              # Critic 价值网络的隐藏层宽度
    history_length: int = 10                                      # 策略使用的历史观测帧数, 包含当前帧
    goal_frames: int = 10                                         # 参考目标中使用的未来动作帧数
    latent_dim: int = 32                                          # 参考动作编码后的潜变量维度
    policy_type: str = "mlp"                                      # 策略网络类型, 当前仅实现 mlp
    chunk_length: int = 1                                         # 一次预测的动作步数, 当前仅实现单步
    execute_steps: int = 1                                        # 每次预测后连续执行的步数, 当前仅实现单步
    norm_max_count: float = 2e8                                   # 运行均值和方差达到该样本数后停止更新

    # 损失, 梯度和优化参数; 采样和循环次数由 workflows 读取后调度.
    control_dt: float = 0.02                                      # 一个控制周期的时长, 单位为秒, 需与 MDP 和环境一致
    wm_window: int = 8                                            # 训练世界模型时采样的连续交互步数
    imagination_horizon: int = 32                                 # 使用世界模型展开预测轨迹的步数
    learning_rate: float = 2e-4                                   # 世界模型, 策略和 Critic 优化器的初始学习率
    policy_betas: tuple[float, float] = (0.9, 0.95)              # 策略 Adam 优化器的一阶和二阶动量系数
    gamma: float = 0.99                                           # 每个控制步的奖励折扣因子
    gae_lambda: float = 0.95                                      # Critic 回报目标递推中混合价值与后续回报的系数
    critic_epochs: int = 5                                        # 同一批固定 Critic 目标的训练轮数
    critic_minibatches: int = 4                                   # 每轮 Critic 更新划分的小批次数量
    grad_clip_norm: float = 1.0                                   # 每个网络更新时的梯度整体范数上限
    policy_grad_clip_value: float = 0.1                           # 策略梯度整体裁剪后, 每个梯度元素的绝对值上限
    lr_decay_iterations: int = 250_000                            # 余弦学习率从初始值衰减到最低值所用的迭代数
    min_lr_ratio: float = 0.03                                    # 最低学习率占初始学习率的比例, 衰减结束后保持不变

    def __post_init__(self):
        """在创建网络前检查维度, 时间和优化参数, 不提前启用预留的策略后端."""

        for name in ("history_length", "goal_frames", "latent_dim", "chunk_length", "execute_steps", "wm_window", "imagination_horizon", "critic_epochs", "critic_minibatches", "lr_decay_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须为正整数")
        for name in ("wm_hidden_dims", "policy_hidden_dims", "critic_hidden_dims"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(isinstance(width, bool) or not isinstance(width, int) or width <= 0 for width in values):
                raise ValueError(f"{name} 必须为正整数元组, 空元组表示仅使用线性输出层")
        for name in ("norm_max_count", "control_dt", "learning_rate", "grad_clip_norm", "policy_grad_clip_value"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为正的有限数值")
        if not isinstance(self.policy_type, str) or not self.policy_type:
            raise ValueError("policy_type 必须为非空字符串")
        if self.execute_steps > self.chunk_length:
            raise ValueError("execute_steps 不能大于 chunk_length")
        if not 0 <= self.gamma < 1 or not 0 <= self.gae_lambda <= 1 or not 0 < self.min_lr_ratio <= 1:
            raise ValueError("gamma 必须位于 [0,1), gae_lambda 位于 [0,1], min_lr_ratio 位于 (0,1]")
        if len(self.policy_betas) != 2 or any(not 0 <= beta < 1 for beta in self.policy_betas):
            raise ValueError("policy_betas 必须包含两个 [0,1) 范围内的数值")


##############################################################################
#                         世界模型                                            #
##############################################################################


class WorldModel(nn.Module):
    """确定性动力学网络, 使用完整状态预测下一状态, 不读取经验池或参考动作.

    输入: 六字段 state [...,D], action [...,29], 原始 context [...,33].
    输出: 同前导维的六字段下一状态, 保持 xyzw 四元数和原始物理量单位.
    结构: 不变状态特征 71 + 动作 29 + 归一化 context 33 -> RunningNorm ->
          MLP(2048,2048,2048, SiLU, LayerNorm) -> 70 维局部增量 -> 状态更新.
    update_statistics 显式更新归一化统计, forward/step 本身不修改统计.
    """

    def __init__(self, config: SuperMimicConfig | None = None):
        """构建动力学 MLP 与输入输出统计, 输出层零初始化以从平均状态增量开始."""

        super().__init__()
        self.config = config or SuperMimicConfig()
        input_dim = INVARIANT_DIM + ACTION_DIM + CONTEXT_DIM
        self.net = MLP(input_dim, self.config.wm_hidden_dims, DYNAMICS_DIM, activation="silu", layer_norm=True)
        nn.init.zeros_(self.net.net[-1].weight)
        nn.init.zeros_(self.net.net[-1].bias)
        self.in_norm = RunningNorm(input_dim, self.config.norm_max_count)
        self.out_norm = RunningNorm(DYNAMICS_DIM, self.config.norm_max_count, eps=1e-4)
        # 与 vibed 一致, 防止增量方差很小时某类误差获得过大的损失权重.
        floor = torch.tensor((3e-3, 5e-3, 2e-2, 5e-2, 3e-3, 5e-2))
        self.register_buffer("error_floor", floor.repeat_interleave(torch.tensor((3, 3, 3, 3, 29, 29))))

    def inputs(self, state, action, context):
        """状态特征 [...,71] + 动作 [...,29] + 内部归一化物理参数 [...,33] -> [...,133]."""

        features = invariant_features(state)
        leading = features.shape[:-1]
        for name, tensor, width in (("action", action, ACTION_DIM), ("context", context, CONTEXT_DIM)):
            if tensor.shape != leading + (width,) or tensor.device != features.device or tensor.dtype != features.dtype:
                raise ValueError(f"{name} 必须与状态同设备同精度, 形状为 {leading + (width,)}")
        return torch.cat((features, action, normalize_context(context)), dim=-1)

    @staticmethod
    def targets(state, next_state):
        """计算两状态间的局部增量 [...,70], 用于统计和状态误差, 不改变原始状态.

        位移在原状态的水平朝向系表达, 旋转为原根部系中的旋转向量.
        线速度和角速度增量在原根部系表达, 关节角和速度直接做差.
        """

        rotation = quaternion_to_matrix(state["root_quat"])
        heading_inverse = heading_matrix(rotation).transpose(-1, -2)
        inverse = rotation.transpose(-1, -2)
        quaternion = F.normalize(state["root_quat"], dim=-1)
        conjugate = torch.cat((-quaternion[..., :3], quaternion[..., 3:]), dim=-1)
        return torch.cat(((heading_inverse @ (next_state["root_pos"] - state["root_pos"])[..., None]).squeeze(-1), quaternion_to_rotation_vector(quaternion_multiply(conjugate, next_state["root_quat"])), (inverse @ (next_state["root_lin_vel"] - state["root_lin_vel"])[..., None]).squeeze(-1), (inverse @ (next_state["root_ang_vel"] - state["root_ang_vel"])[..., None]).squeeze(-1), next_state["joint_pos"] - state["joint_pos"], next_state["joint_vel"] - state["joint_vel"]), dim=-1)

    @torch.no_grad()
    def update_statistics(self, data: Mapping):
        """接收单步 [N,...] 或批量真实转移, 显式更新输入和局部增量的统计."""

        validate_replay_data(data, device=self.in_norm.mean.device)
        self.in_norm.update(self.inputs(data["state"], data["action"], data["context"]))
        self.out_norm.update(self.targets(data["state"], data["next_state"]))

    def step(self, state, action, context):
        """state/action/context -> 拼接与标准化 -> MLP -> 反标准化增量 -> 六字段下一状态.

        保留状态和动作的梯度; 参数冻结时仍可用于策略的多步反向传播.
        """

        prediction = self.net(self.in_norm(self.inputs(state, action, context)))
        increment = prediction * self.out_norm.std() + self.out_norm.mean
        position, orientation, linear, angular, joints, velocities = increment.split((3, 3, 3, 3, 29, 29), dim=-1)
        rotation = quaternion_to_matrix(state["root_quat"])
        quaternion = F.normalize(quaternion_multiply(state["root_quat"], rotation_vector_to_quaternion(orientation)), dim=-1)
        return {"root_pos": state["root_pos"] + (heading_matrix(rotation) @ position[..., None]).squeeze(-1), "root_quat": quaternion, "root_lin_vel": state["root_lin_vel"] + (rotation @ linear[..., None]).squeeze(-1), "root_ang_vel": state["root_ang_vel"] + (rotation @ angular[..., None]).squeeze(-1), "joint_pos": state["joint_pos"] + joints, "joint_vel": state["joint_vel"] + velocities}

    def forward(self, state, action, context):
        """支持标准 nn.Module 调用, 与 step 一样返回可微的六字段下一状态."""

        return self.step(state, action, context)


##############################################################################
#                         策略网络                                            #
##############################################################################


class Policy(nn.Module):
    """参考目标编码器和动作解码器, 输出统一动作序列接口.

    输入: obs_history [...,L,99], goal [...,K*73], L/K 由配置决定.
    输出: actions [...,C,29], 当前 C=1, workflows 取 actions[...,0,:] 执行一步.
    结构: 目标 -> RunningNorm -> 编码器 -> tanh -> z [...,32];
          历史 -> RunningNorm -> 展平 -> 与 z 拼接 -> 解码器 -> [...,C,29].
    编码器和解码器隐藏层均为 (2048,1024,512,256), 使用 ELU, 不使用 LayerNorm.
    policy_type 保留生成式后端的选择接口, 本版不假装实现 Diffusion 或 Flow Matching.
    """

    def __init__(self, config: SuperMimicConfig | None = None):
        """创建目标编码器和动作解码器, 对未实现的后端或 chunk 训练明确报错."""

        super().__init__()
        self.config = config or SuperMimicConfig()
        if self.config.policy_type != "mlp":
            raise NotImplementedError("当前仅实现 policy_type='mlp', 生成式策略需接入对应网络和训练规则")
        if self.config.chunk_length != 1 or self.config.execute_steps != 1:
            raise NotImplementedError("当前按论文实现单步策略, chunk 接口已保留, 多步执行和学习规则尚未接入")
        self.goal_dim = self.config.goal_frames * GOAL_FRAME_DIM
        self.encoder = MLP(self.goal_dim, self.config.policy_hidden_dims, self.config.latent_dim)
        self.decoder = MLP(OBS_DIM * self.config.history_length + self.config.latent_dim, self.config.policy_hidden_dims, ACTION_DIM * self.config.chunk_length)
        with torch.no_grad():
            self.decoder.net[-1].weight.mul_(0.01)
            self.decoder.net[-1].bias.zero_()
        self.obs_norm = RunningNorm(OBS_DIM, self.config.norm_max_count)
        self.goal_norm = RunningNorm(self.goal_dim, self.config.norm_max_count)

    @torch.no_grad()
    def update_statistics(self, observation, goal):
        """用真实采集的单帧观测 [...,99] 和参考目标 [...,K*73] 更新统计, 不重复计数历史."""

        self.obs_norm.update(observation)
        self.goal_norm.update(goal)

    def predict_actions(self, obs_history, goal):
        """目标 -> 编码器/tanh; 历史 -> 展平; 两者拼接 -> 解码器 -> 动作序列 [...,C,29].

        前导批量维必须相同, 本函数不采样探索噪声, 不裁剪动作, 不保存动作队列.
        """

        if obs_history.ndim < 2 or obs_history.shape[-2:] != (self.config.history_length, OBS_DIM):
            raise ValueError(f"obs_history 末两维必须为 ({self.config.history_length}, {OBS_DIM})")
        if goal.shape != obs_history.shape[:-2] + (self.goal_dim,):
            raise ValueError("goal 必须与观测历史具有相同前导批量维和正确目标维度")
        latent = torch.tanh(self.encoder(self.goal_norm(goal)))
        history = self.obs_norm(obs_history).flatten(-2)
        actions = self.decoder(torch.cat((latent, history), dim=-1))
        return actions.reshape(obs_history.shape[:-2] + (self.config.chunk_length, ACTION_DIM))

    def forward(self, obs_history, goal):
        """标准 nn.Module 接口, 与 predict_actions 一样返回 [...,C,29], 保留策略梯度."""

        return self.predict_actions(obs_history, goal)


##############################################################################
#                         Critic 价值网络                                     #
##############################################################################


class Critic(nn.Module):
    """状态价值网络, 输入 MDP 特权特征和未来目标, 与策略不共享参数.

    feature_dim 由 workflows 从 mdp.critic_dim 传入, 默认跟踪 14 个部位时为 184.
    结构: features [...,F] + goal [...,K*73] -> RunningNorm -> MLP(256,128) -> [...].
    输出为无界状态价值 V, 不使用概率激活, 也不创建目标 Critic.
    """

    def __init__(self, feature_dim: int, config: SuperMimicConfig | None = None):
        """接收任务提供的特征维度, 创建独立 ELU MLP 和输入统计."""

        super().__init__()
        self.config = config or SuperMimicConfig()
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("feature_dim 必须为正整数")
        self.feature_dim = feature_dim
        self.goal_dim = self.config.goal_frames * GOAL_FRAME_DIM
        self.norm = RunningNorm(feature_dim + self.goal_dim, self.config.norm_max_count)
        self.net = MLP(feature_dim + self.goal_dim, self.config.critic_hidden_dims, 1)

    def inputs(self, features, goal):
        """拼接任务特征 [...,F] 和未来目标 [...,G] 为 [...,F+G], 不计算参考跟踪误差."""

        if features.shape[-1:] != (self.feature_dim,) or goal.shape != features.shape[:-1] + (self.goal_dim,):
            raise ValueError("Critic 特征和目标维度错误, 或前导批量维不一致")
        return torch.cat((features, goal), dim=-1)

    def forward(self, features, goal):
        """特征与目标拼接 -> 标准化 -> ELU MLP -> 线性标量 -> 去除末维, 输出 [...]."""

        return self.net(self.norm(self.inputs(features, goal))).squeeze(-1)


##############################################################################
#                         算法状态与梯度管理                                  #
##############################################################################


class SuperMimicAgent:
    """持有 workflows 已组装的网络与任务, 管理学习规则和优化器状态.

    构造输入: world_model, policy, critic, mdp, 可选的同一份 SuperMimicConfig.
    update_world_model 接收经验池采样的 windows [B,k,...] 和 bool mask [B,k].
    update_policy 接收 ImaginedRollout.as_dict() 的可微轨迹, 不在内部进行预测展开.
    prepare_critic_batch 接收同布局轨迹, 计算固定目标; update_critic 接收该批次.
    各 update 返回已经分离梯度的指标字典, 本类不写日志或磁盘文件.

    策略阶段用法:
        with agent.policy_phase():
            # workflows 循环调用 policy 和 world_model, 用 MDP 计算奖励并收集轨迹.
            info = agent.update_policy(rollout.as_dict())
        critic_batch = agent.prepare_critic_batch(rollout.as_dict())
        critic_info = agent.update_critic(critic_batch)
        # workflows 释放本轮轨迹及临时批次, 再进入下一轮采集或模型更新.
    """

    def __init__(self, world_model: WorldModel, policy: Policy, critic: Critic, mdp: TrackingMDP, config: SuperMimicConfig | None = None):
        """校验网络和任务配置的一致性, 创建三个独立 Adam 优化器, 不重新组装网络."""

        self.config = config or world_model.config
        self.world_model, self.policy, self.critic, self.mdp = world_model, policy, critic, mdp
        if any(model.config != self.config for model in (world_model, policy, critic)):
            raise ValueError("各网络和 Agent 必须使用一致的 SuperMimicConfig")
        if mdp.config.goal_frames != self.config.goal_frames or not math.isclose(mdp.config.control_dt, self.config.control_dt, rel_tol=0, abs_tol=1e-10):
            raise ValueError("算法与 MDP 的目标帧数和控制周期必须一致")
        if critic.feature_dim != mdp.critic_dim:
            raise ValueError("Critic 特征维度必须等于 mdp.critic_dim")
        self.device = world_model.in_norm.mean.device
        if mdp.device != self.device or mdp.dtype != torch.float32:
            raise ValueError("MDP 必须与网络同设备, 并使用 float32")
        for model in (world_model, policy, critic):
            if any(tensor.device != self.device or (tensor.is_floating_point() and tensor.dtype != torch.float32) for tensor in (*model.parameters(), *model.buffers())):
                raise ValueError("所有网络和统计量必须在同一设备并使用 float32, 请先由 workflows 调用 .to(device)")
        self.wm_optimizer = torch.optim.Adam(world_model.parameters(), lr=self.config.learning_rate)
        self.policy_optimizer = torch.optim.Adam(policy.parameters(), lr=self.config.learning_rate, betas=self.config.policy_betas)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self.config.learning_rate)
        self.iteration = 0
        self.updates = {"world_model": 0, "policy": 0, "critic": 0}
        self._policy_phase_active = False

    @contextmanager
    def policy_phase(self):
        """在预测展开和策略反向传播期间固定模型与 Critic 参数, 退出时恢复原标志.

        这里不使用 no_grad, 以保留网络输出对状态和动作输入的导数.
        上下文内不能更新归一化统计或另外两个优化器, 异常退出也会恢复参数标志.
        """

        if self._policy_phase_active:
            raise RuntimeError("不能嵌套进入 policy_phase")
        parameters = list(self.world_model.parameters()) + list(self.critic.parameters())
        original_flags = [parameter.requires_grad for parameter in parameters]
        self._policy_phase_active = True
        try:
            for parameter in parameters:
                parameter.grad = None
                parameter.requires_grad_(False)
            yield
        finally:
            for parameter, flag in zip(parameters, original_flags):
                parameter.requires_grad_(flag)
            self._policy_phase_active = False

    def set_iteration(self, iteration: int):
        """按 workflows 传入的迭代编号更新余弦学习率, 衰减结束后保持最低比例."""

        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise ValueError("iteration 必须为非负整数")
        progress = min(iteration / self.config.lr_decay_iterations, 1.0)
        ratio = self.config.min_lr_ratio + (1 - self.config.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        for optimizer in (self.wm_optimizer, self.policy_optimizer, self.critic_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = self.config.learning_rate * ratio
        self.iteration = iteration

    @torch.no_grad()
    def update_statistics(self, transition, observation, goal):
        """采集后更新世界模型与策略的真实数据统计, 不从预测轨迹反复更新这些统计."""

        if self._policy_phase_active:
            raise RuntimeError("策略预测展开期间不能更新归一化统计")
        self.world_model.update_statistics(transition)
        self.policy.update_statistics(observation, goal)

    def _apply_update(self, optimizer, parameters, loss, *, clip_value=None):
        """共享反向传播和梯度裁剪规则, 发现非有限损失或梯度时停止且不更新参数."""

        parameters = list(parameters)
        optimizer.zero_grad(set_to_none=True)
        if loss.ndim != 0 or not loss.requires_grad:
            raise ValueError("更新需要保留计算图的标量损失, 请检查是否错误使用了 no_grad/detach")
        if not torch.isfinite(loss):
            raise FloatingPointError("损失包含 NaN 或 Inf, 未执行参数更新")
        try:
            loss.backward()
            if not any(parameter.grad is not None for parameter in parameters):
                raise ValueError("损失未连接到待更新网络, 请检查预测展开中是否错误分离了梯度")
            norm = nn.utils.clip_grad_norm_(parameters, self.config.grad_clip_norm, error_if_nonfinite=True)
            if clip_value is not None:
                nn.utils.clip_grad_value_(parameters, clip_value)
            optimizer.step()
        except Exception:
            optimizer.zero_grad(set_to_none=True)
            raise
        return {"loss": loss.detach(), "grad_norm": norm.detach()}

    # ---------------------------------- 世界模型损失与更新 ---------------------------------- #

    def world_model_loss(self, windows: Mapping, mask: torch.Tensor):
        """从真实起始状态按已记录动作自回归 k 步, 返回标量损失和末步误差.

        windows 使用 [B,k,...], mask 由经验池提供; 结束转移有效, 不跨回合反传.
        对每步 70 维误差取均值, 对有效时间求和, 最后除以 B, 与 vibed 一致.
        """

        validate_replay_data(windows, device=self.device)
        batch, steps = windows["action"].shape[:2]
        if windows["action"].ndim != 3 or steps != self.config.wm_window:
            raise ValueError("windows 必须为 [B,wm_window,...] 的连续窗口")
        if mask.shape != (batch, steps) or mask.dtype != torch.bool or mask.device != self.device:
            raise ValueError("mask 必须为同设备的 bool [B,wm_window]")
        boundary = windows["terminated"][:, :-1] | windows["truncated"][:, :-1] | (windows["episode_step"][:, 1:] != windows["episode_step"][:, :-1] + 1)
        if (mask[:, 1:] & (~mask[:, :-1] | boundary)).any() or not mask.any():
            raise ValueError("mask 不得在结束后重新有效, 不得跨回合, 且至少保留一步")

        # 逐步缩小有效批次, 避免先对跨回合或无效预测计算损失再乘零造成 NaN 污染.
        indices = torch.arange(batch, device=self.device)
        state = {key: value[:, 0].detach() for key, value in windows["state"].items()}
        scale = torch.maximum(self.world_model.out_norm.std(), self.world_model.error_floor)
        total, last_error = torch.zeros((), device=self.device), torch.zeros((), device=self.device)
        for time in range(steps):
            keep = mask[indices, time]
            indices = indices[keep]
            state = {key: value[keep] for key, value in state.items()}
            if indices.numel() == 0:
                break
            state = self.world_model.step(state, windows["action"][indices, time].detach(), windows["context"][indices, time].detach())
            target = {key: value[indices, time].detach() for key, value in windows["next_state"].items()}
            error = (self.world_model.targets(target, state) / scale).square().mean(-1)
            total = total + error.sum()
            if time == steps - 1:
                last_error = error.mean()
        return total / batch, last_error

    def update_world_model(self, windows: Mapping, mask: torch.Tensor):
        """对 workflows 采样的一批真实窗口执行一次模型更新, 不在这里访问经验池."""

        if self._policy_phase_active:
            raise RuntimeError("策略阶段不能更新世界模型")
        loss, last_error = self.world_model_loss(windows, mask)
        info = self._apply_update(self.wm_optimizer, self.world_model.parameters(), loss)
        self.updates["world_model"] += 1
        return {**info, "mean_step_loss": info["loss"] / self.config.wm_window, "last_step_error": last_error.detach()}

    # ---------------------------------- 策略目标与更新 ---------------------------------- #

    def policy_loss(self, rollout: Mapping):
        """按论文式 2 计算负预测回报, 包含软存活权重和末端 Critic 自举价值."""

        validate_imagined_rollout(rollout, history_length=self.config.history_length, goal_frames=self.config.goal_frames, device=self.device)
        rewards, terminations = rollout["rewards"], rollout["soft_terminations"]
        horizon, batch = rewards.shape
        if horizon != self.config.imagination_horizon:
            raise ValueError("预测轨迹长度必须与 imagination_horizon 一致")
        alive = (1 - terminations).cumprod(dim=0)
        weight = torch.cat((torch.ones(1, batch, device=self.device), alive[:-1]), dim=0)
        discount = self.config.gamma ** torch.arange(horizon, device=self.device, dtype=torch.float32)
        returns = (discount[:, None] * weight * rewards).sum(dim=0)

        # Critic 参数已固定, 但末端价值对预测状态的梯度继续传回策略.
        terminal_state = {key: value[-1] for key, value in rollout["states"].items()}
        goal = rollout["goals"][-1] if "goals" in rollout else self.mdp.goal_features(rollout["clip_id"], rollout["frame_ids"][-1])
        features = self.mdp.critic_features(terminal_state, rollout["clip_id"], rollout["frame_ids"][-1], rollout["context"])
        terminal_value = self.critic(features, goal)
        returns = returns + self.config.gamma**horizon * alive[-1] * terminal_value
        return -returns.mean(), {"return": returns.detach().mean(), "survival": alive[-1].detach().mean(), "reward": rewards.detach().mean()}

    def update_policy(self, rollout: Mapping):
        """在 policy_phase 内更新策略, 使用保留计算图的预测轨迹, 不重新生成轨迹."""

        if not self._policy_phase_active:
            raise RuntimeError("请在 with agent.policy_phase() 中完成预测展开和策略更新")
        loss, metrics = self.policy_loss(rollout)
        info = self._apply_update(self.policy_optimizer, self.policy.parameters(), loss, clip_value=self.config.policy_grad_clip_value)
        self.updates["policy"] += 1
        return {**info, **metrics}

    # ---------------------------------- Critic 固定目标与更新 ---------------------------------- #

    @torch.no_grad()
    def prepare_critic_batch(self, rollout: Mapping):
        """策略更新结束后计算一次固定目标, 返回 inputs [H+1,B,F+G] 与 targets [H,B].

        归一化统计在本批目标计算前更新一次; 后续多轮拟合不重新计算目标或统计.
        所有输出均分离计算图, rewards 和状态也不会将 Critic 梯度传回策略或模型.
        """

        if self._policy_phase_active:
            raise RuntimeError("请退出 policy_phase 后再准备 Critic 批次, 避免改变策略图引用的统计量")
        validate_imagined_rollout(rollout, history_length=self.config.history_length, goal_frames=self.config.goal_frames, device=self.device)
        horizon, batch = rollout["rewards"].shape
        states = {key: value.detach() for key, value in rollout["states"].items()}
        clips = rollout["clip_id"].unsqueeze(0).expand(horizon + 1, -1)
        context = rollout["context"].unsqueeze(0).expand(horizon + 1, -1, -1)
        goals = rollout["goals"] if "goals" in rollout else self.mdp.goal_features(clips, rollout["frame_ids"])
        features = self.mdp.critic_features(states, clips, rollout["frame_ids"], context)
        inputs = self.critic.inputs(features, goals).detach()
        self.critic.norm.update(inputs)
        values = self.critic.net(self.critic.norm(inputs)).squeeze(-1)
        targets = lambda_returns(rollout["rewards"], rollout["soft_terminations"], values, self.config.gamma, self.config.gae_lambda)
        return {"inputs": inputs, "targets": targets}

    def update_critic(self, batch: Mapping, *, generator: torch.Generator | None = None):
        """对固定回报目标执行多轮随机小批次回归, 覆盖全部样本, 不丢弃尾部数据."""

        if self._policy_phase_active:
            raise RuntimeError("策略阶段不能更新 Critic")
        inputs, targets = batch["inputs"], batch["targets"]
        if targets.ndim != 2 or inputs.shape != (targets.shape[0] + 1, targets.shape[1], self.critic.norm.dim):
            raise ValueError("Critic 批次必须为 inputs [H+1,B,F+G], targets [H,B]")
        if any(tensor.device != self.device or tensor.dtype != torch.float32 or tensor.requires_grad for tensor in (inputs, targets)):
            raise ValueError("Critic 批次必须为同设备 float32, 且已分离梯度")
        inputs = inputs[:-1].flatten(0, 1)
        targets = targets.flatten()
        if targets.numel() == 0:
            raise ValueError("Critic 批次不能为空")
        losses, norms = [], []
        for _ in range(self.config.critic_epochs):
            permutation = torch.randperm(targets.numel(), device=self.device, generator=generator)
            for indices in permutation.tensor_split(min(self.config.critic_minibatches, targets.numel())):
                prediction = self.critic.net(self.critic.norm(inputs[indices])).squeeze(-1)
                loss = (prediction - targets[indices]).square().mean()
                info = self._apply_update(self.critic_optimizer, self.critic.parameters(), loss)
                losses.append(info["loss"])
                norms.append(info["grad_norm"])
                self.updates["critic"] += 1
        return {"loss": torch.stack(losses).mean(), "grad_norm": torch.stack(norms).mean()}

    # ---------------------------------- 训练状态导出与恢复 ---------------------------------- #

    def state_dict(self):
        """返回模型, 统计量, 优化器和计数, 供 workflows 保存; 不包含经验池或仿真现场.

        与 PyTorch state_dict 一样含张量引用, 异步保存需要由调用方复制快照.
        """

        if self._policy_phase_active:
            raise RuntimeError("请在本轮参数更新结束后导出训练状态")
        return {"format_version": 1, "config": asdict(self.config), "critic_feature_dim": self.critic.feature_dim, "world_model": self.world_model.state_dict(), "policy": self.policy.state_dict(), "critic": self.critic.state_dict(), "wm_optimizer": self.wm_optimizer.state_dict(), "policy_optimizer": self.policy_optimizer.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(), "iteration": self.iteration, "updates": dict(self.updates)}

    def load_state_dict(self, state: Mapping):
        """加载 workflows 提供的完整训练状态, 要求算法配置一致, 不读取磁盘文件."""

        if self._policy_phase_active:
            raise RuntimeError("策略阶段不能恢复训练状态")
        required = {"format_version", "config", "critic_feature_dim", "world_model", "policy", "critic", "wm_optimizer", "policy_optimizer", "critic_optimizer", "iteration", "updates"}
        if set(state) != required or state["format_version"] != 1 or state["config"] != asdict(self.config) or state["critic_feature_dim"] != self.critic.feature_dim:
            raise ValueError("训练状态格式, 算法配置或 Critic 特征维度不匹配")
        counters = state["updates"]
        if set(counters) != set(self.updates) or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (*counters.values(), state["iteration"])):
            raise ValueError("训练状态的更新计数或迭代编号无效")
        self.world_model.load_state_dict(state["world_model"])
        self.policy.load_state_dict(state["policy"])
        self.critic.load_state_dict(state["critic"])
        self.wm_optimizer.load_state_dict(state["wm_optimizer"])
        self.policy_optimizer.load_state_dict(state["policy_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.updates = dict(counters)
        self.set_iteration(state["iteration"])


__all__ = ["SuperMimicConfig", "WorldModel", "Policy", "Critic", "SuperMimicAgent", "lambda_returns"]

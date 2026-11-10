"""本轮预测轨迹容器, 组织上层生成的张量, 不调用世界模型或执行训练更新."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from supermimic.data.validation import ACTION_DIM, CONTEXT_DIM, GOAL_FRAME_DIM, OBS_DIM, STATE_WIDTHS, _check_nonnegative, _check_state, _check_tensor, validate_imagined_rollout


##############################################################################
#                         公共参数与数据视图工具                              #
##############################################################################


def _positive_int(value, name):
    """检查预测长度, 历史长度及目标帧数, 不接受 bool."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须为正整数")
    return value


def _detach_data(data):
    """递归分离计算图, 返回共享原存储的只读使用视图, 不额外复制张量."""

    return {key: _detach_data(value) if isinstance(value, Mapping) else value.detach() for key, value in data.items()}


##############################################################################
#                         本轮短期预测轨迹                                    #
##############################################################################


# 收集上层生成的预测结果, 同一批数据供策略和 Critic 使用, 本轮更新结束后释放.
class ImaginedRollout:
    """组织 B 条 H 步预测轨迹, 本轮策略和 Critic 更新完成后由上层 clear.

    构造输入:
        initial_state 为六字段 [B,D], initial_previous_action [B,29].
        initial_observation_history [B,L,99], context [B,33], clip_id/frame_id [B].
        horizon=H, history_length=L, goal_frames=K; initial_goal 可选, 形状 [B,K*73].
        所有浮点字段为同设备 float32, 编号为 int64, 默认 L=10 和 K=10.

    逐步输入:
        append 接收 next_state[B,D], action[B,29], reward/soft_termination[B], frame_id[B].
        frame_id 对应 next_state; 初始缓存 goal 时每步也必须提供 next goal.
        参考帧推进, 观测历史更新以及模型调用均由上层执行.

    输出:
        as_dict 返回 validation.py 定义的完整轨迹, 状态 [H+1,B,D], 动作 [H,B,29].
        detached 返回相同布局的分离梯度视图, 供上层构造 Critic 特征和回报目标.
        不计算 Critic 特征, 价值或回报目标, 不加入长期经验池.

    张量由引用持有, 收集后上层不得原地修改; 状态字典本身会浅复制.
    clear 只释放容器持有的引用, 外部仍持有输出时内存不会立即回收.

    用法:
        rollout = ImaginedRollout(state, previous_action, history, context, clip_id, frame_id, horizon=16)
        # 上层反复调用策略和世界模型, 每次将输出交给 rollout.append(...).
        batch = rollout.as_dict()            # 策略损失使用保留计算图的预测数据.
        critic_batch = rollout.detached()    # 上层另行计算 Critic 特征和固定目标.
        # 本轮更新完成后删除外部 batch 引用, 再调用 rollout.clear().
    """

    # ---------------------------------- 初始化与逐步收集 ---------------------------------- #

    def __init__(self, initial_state: Mapping, initial_previous_action: torch.Tensor, initial_observation_history: torch.Tensor, context: torch.Tensor, clip_id: torch.Tensor, frame_id: torch.Tensor, *, horizon: int, history_length: int = 10, goal_frames: int = 10, initial_goal: torch.Tensor | None = None, check_values: bool = False):
        """校验起始数据并保存引用, 不预分配会覆盖计算图的环形存储."""

        self.horizon = _positive_int(horizon, "horizon")                # 预测展开步数 H
        self.history_length = _positive_int(history_length, "history_length")  # 初始观测历史长度 L
        self.goal_frames = _positive_int(goal_frames, "goal_frames")   # 每个目标包含的未来参考帧数 K
        self.check_values = check_values
        if not isinstance(initial_previous_action, torch.Tensor) or initial_previous_action.ndim != 2 or initial_previous_action.shape[0] <= 0:
            raise ValueError("initial_previous_action 必须为非空的 [B,29] 张量")
        # B 是本轮预测轨迹数量, 独立于物理仿真的并行环境数 N.
        self.batch_size, self.device = initial_previous_action.shape[0], initial_previous_action.device

        # 检查六字段状态, 历史观测, 原始物理参数和参考编号, 不修改或分离输入.
        self._check_state(initial_state)
        self._check_tensor(initial_previous_action, (self.batch_size, ACTION_DIM), "initial_previous_action")
        self._check_tensor(initial_observation_history, (self.batch_size, history_length, OBS_DIM), "initial_observation_history")
        self._check_tensor(context, (self.batch_size, CONTEXT_DIM), "context")
        self._check_tensor(clip_id, (self.batch_size,), "clip_id", torch.int64)
        self._check_tensor(frame_id, (self.batch_size,), "frame_id", torch.int64)
        if check_values:
            _check_nonnegative({"clip_id": clip_id, "frame_id": frame_id}, ("clip_id", "frame_id"), "rollout")
        if initial_goal is not None:
            self._check_tensor(initial_goal, (self.batch_size, self.goal_frames * GOAL_FRAME_DIM), "initial_goal")

        # 本轮共享的数据只保存一次, 每步变化的状态和参考帧则分别收集.
        self._initial = {"initial_previous_action": initial_previous_action, "initial_observation_history": initial_observation_history, "context": context, "clip_id": clip_id}
        self._states, self._frame_ids = [dict(initial_state)], [frame_id]
        self._goals = None if initial_goal is None else [initial_goal]  # 不缓存目标时由上层按需生成

        # 使用列表持有每步张量, 最后 stack, 保留跨预测步骤的计算图.
        self._actions, self._rewards, self._terminations = [], [], []
        self._batch = None      # 完整轨迹堆叠后的缓存, 避免重复 stack
        self._released = False  # clear 后禁止继续复用本轮容器

    def __len__(self):
        """返回已经收集的转移步数, 不计初始状态."""

        return len(self._actions)

    # 每次接收上层完成的一步预测, 本函数只收集结果, 不调用策略或世界模型.
    def append(self, next_state: Mapping, action: torch.Tensor, reward: torch.Tensor, soft_termination: torch.Tensor, frame_id: torch.Tensor, *, goal: torch.Tensor | None = None):
        """校验本步数据后加入列表, 保留原张量计算图; 不生成动作或推进参考帧."""

        self._check_active()
        if len(self) >= self.horizon:
            raise RuntimeError("已收集 horizon 步, 请完成本轮更新后释放并创建新容器")

        # 先验证整步数据, 避免某个字段错误时只收集了一部分结果.
        self._check_state(next_state)
        self._check_tensor(action, (self.batch_size, ACTION_DIM), "action")
        self._check_tensor(reward, (self.batch_size,), "reward")
        self._check_tensor(soft_termination, (self.batch_size,), "soft_termination")
        self._check_tensor(frame_id, (self.batch_size,), "frame_id", torch.int64)
        if self.check_values:
            _check_nonnegative({"frame_id": frame_id}, ("frame_id",), "rollout")
            if ((soft_termination < 0) | (soft_termination > 1)).any():
                raise ValueError("soft_termination 必须位于 [0,1]")
        if (goal is None) != (self._goals is None):
            raise ValueError("goal 缓存方式必须与 initial_goal 一致")
        if goal is not None:
            self._check_tensor(goal, (self.batch_size, self.goal_frames * GOAL_FRAME_DIM), "goal")

        # 状态和参考帧对应 t+1, 动作, 奖励和软终止对应本次 t 到 t+1 的转移.
        self._states.append(dict(next_state))
        self._frame_ids.append(frame_id)
        self._actions.append(action)
        self._rewards.append(reward)
        self._terminations.append(soft_termination)
        if self._goals is not None:
            self._goals.append(goal)

    # ---------------------------------- 批次组织与 Critic 数据视图 ---------------------------------- #

    # 把逐步收集的数据整理成完整轨迹, 保留策略反向传播所需的梯度关系.
    def as_dict(self) -> dict:
        """完整收集 H 步后堆叠并缓存批次, 保留计算图; 返回值不得原地修改."""

        self._check_active()
        if len(self) != self.horizon:
            raise RuntimeError(f"轨迹尚未完成, 已收集 {len(self)}/{self.horizon} 步")
        if self._batch is None:
            # 状态和参考帧有 H+1 项, 动作, 奖励和软终止有 H 项, 新增首维为时间轴.
            batch = {"states": {key: torch.stack([state[key] for state in self._states]) for key in STATE_WIDTHS}, "actions": torch.stack(self._actions), **self._initial, "frame_ids": torch.stack(self._frame_ids), "rewards": torch.stack(self._rewards), "soft_terminations": torch.stack(self._terminations)}
            if self._goals is not None:
                batch["goals"] = torch.stack(self._goals)
            # 复用统一字段检查, 通过后缓存批次; 检查不会切断原有计算图.
            validate_imagined_rollout(batch, history_length=self.history_length, goal_frames=self.goal_frames, check_values=self.check_values, device=self.device)
            self._batch = batch
        return self._batch

    # Critic 更新使用分离梯度后的预测数据, 特征和回报目标仍由上层计算.
    def detached(self) -> dict:
        """返回无梯度的同布局视图供 Critic 更新; 共享存储, 上层不得原地改写."""

        return _detach_data(self.as_dict())

    # 本轮更新结束后释放容器引用, 不主动清空 PyTorch 的显存缓存.
    def clear(self):
        """释放本轮数据和计算图引用, 可重复调用; 本容器之后不可继续 append."""

        # 先清空逐步列表, 再移除共享起始数据和已堆叠批次的引用.
        self._states.clear()
        self._frame_ids.clear()
        self._actions.clear()
        self._rewards.clear()
        self._terminations.clear()
        if self._goals is not None:
            self._goals.clear()
        self._initial = self._goals = self._batch = None
        self._released = True

    # ---------------------------------- 共用字段检查 ---------------------------------- #

    def _check_active(self):
        """防止已经释放的容器再次参与训练或继续积累引用."""

        if self._released:
            raise RuntimeError("本轮轨迹已释放, 请创建新的 ImaginedRollout")

    def _check_state(self, state):
        """复用 validation 的六字段结构, 类型和可选数值检查."""

        with torch.no_grad():
            _check_state(state, (self.batch_size,), self.device, "rollout.state", self.check_values)

    def _check_tensor(self, value, shape, name, dtype=torch.float32):
        """检查单步字段但不改变输入或 detach, 检查运算不加入训练计算图."""

        with torch.no_grad():
            _check_tensor(value, shape, dtype, self.device, f"rollout.{name}", self.check_values)

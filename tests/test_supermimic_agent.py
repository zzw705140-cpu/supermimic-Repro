"""检查 SuperMimic 的动力学坐标, 梯度边界, 固定价值目标与训练状态恢复.

运行: python -m unittest discover -s tests -p test_supermimic_agent.py -v
仅使用 CPU, 短合成动作和真实 G1 运动学, 不启动 Isaac Lab.
"""

import ast
from copy import deepcopy
from dataclasses import replace
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from sim_platform.assets.train_data.kinematics import G1Kinematics
from sim_platform.assets.train_data.motion import MotionLib
from sim_platform.sim.mdp import TrackingMDP, TrackingMDPConfig
from sim_platform.wrappers.rotations import quaternion_multiply as _quaternion_product, quaternion_to_matrix, quaternion_to_rotation_vector as _rotation_vector, rotation_vector_to_quaternion as _rotation_quaternion
from supermimic.agent.algorithm.supermimic_repro import Critic, Policy, SuperMimicAgent, SuperMimicConfig, WorldModel, lambda_returns
from supermimic.data.imagined_rollout.imagined_rollout import ImaginedRollout


class SuperMimicAgentTest(unittest.TestCase):
    """以独立数值结果和小规模学习过程验证算法, 不用真实仿真训练代替单元检查."""

    @classmethod
    def setUpClass(cls):
        """创建短动作库和真实机器人正运动学, 供策略奖励及 Critic 特征计算."""

        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        assets = PROJECT / "sim_platform/assets/robots/g1"
        tree = ast.parse((assets / "robot.py").read_text())
        names = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "G1_JOINT_NAMES" for target in node.targets))
        task_config = TrackingMDPConfig(goal_frames=2)
        kinematics = G1Kinematics(assets / "g1_29dof.xml", names, task_config.tracked_bodies)
        default_joints = (kinematics.q_lo + kinematics.q_hi) / 2
        time = np.arange(31) / 30
        rotation = Rotation.from_euler("xyz", np.stack((0.1 * time, 0.05 * time, 0.3 + 0.2 * time), axis=-1)).as_quat()
        for clip in range(2):
            position = np.stack((time + clip * 2, 0.2 * time, 1 + 0.03 * time), axis=-1)
            joints = default_joints.numpy()[None] + 0.02 * np.sin(time[:, None] * np.arange(1, 30)[None])
            np.savetxt(Path(cls.directory.name) / f"clip{clip}.csv", np.concatenate((position, rotation, joints), axis=-1), delimiter=",")
        motion = MotionLib(cls.directory.name, kinematics=kinematics)
        cls.mdp = TrackingMDP(kinematics, motion, default_joints, task_config)
        cls.config = SuperMimicConfig(wm_hidden_dims=(32, 32), policy_hidden_dims=(32, 16), critic_hidden_dims=(16, 8), history_length=3, goal_frames=2, latent_dim=4, wm_window=3, imagination_horizon=3, critic_epochs=2, critic_minibatches=4, learning_rate=1e-3)

    def agent(self, seed=5):
        """由测试流程显式组装网络, 对应未来 workflows 的创建职责."""

        torch.manual_seed(seed)
        return SuperMimicAgent(WorldModel(self.config), Policy(self.config), Critic(self.mdp.critic_dim, self.config), self.mdp)

    def reference(self, clips, frames):
        """从动作库读取任意批量布局的六字段参考状态, 不修改缓存."""

        clips, frames = torch.broadcast_tensors(clips, frames)
        state = self.mdp.motion.get_reference(clips.flatten(), frames.flatten(), include_body=False)
        return {key: value.reshape(clips.shape + value.shape[1:]) for key, value in state.items()}

    def windows(self, batch=5):
        """生成同一回合内连续的真实状态转移布局, 原始 context 的摩擦为 1."""

        steps = self.config.wm_window
        clips = (torch.arange(batch) % 2)[:, None].expand(-1, steps)
        frames = torch.arange(steps)[None] + 3 + torch.arange(batch)[:, None]
        context = torch.zeros(batch, steps, 33)
        context[..., 0] = 1
        return {"state": self.reference(clips, frames), "next_state": self.reference(clips, frames + 1), "action": torch.randn(batch, steps, 29) * 0.1, "previous_action": torch.zeros(batch, steps, 29), "context": context, "clip_id": clips, "frame_id": frames, "episode_step": torch.arange(steps)[None].expand(batch, -1), "terminated": torch.zeros(batch, steps, dtype=torch.bool), "truncated": torch.zeros(batch, steps, dtype=torch.bool)}

    def imagine(self, agent, batch=5):
        """在测试流程逐步调用网络和 MDP, 算法组件自身不展开轨迹或读取经验池."""

        clips, frames = torch.arange(batch) % 2, torch.arange(batch) + 3
        state = self.reference(clips, frames)
        previous = torch.zeros(batch, 29)
        context = torch.zeros(batch, 33)
        context[:, 0] = 1
        observation = self.mdp.observation(state, previous, clips, frames, context)
        history = observation[:, None].expand(-1, self.config.history_length, -1).clone()
        goal = self.mdp.goal_features(clips, frames)
        rollout = ImaginedRollout(state, previous, history, context, clips, frames, horizon=self.config.imagination_horizon, history_length=self.config.history_length, goal_frames=self.config.goal_frames, initial_goal=goal)
        actions = []
        for _ in range(self.config.imagination_horizon):
            action = agent.policy.predict_actions(history, goal)[:, 0]
            state = agent.world_model.step(state, action, context)
            frames = frames + 1
            goal = self.mdp.goal_features(clips, frames)
            reward = self.mdp.reward(state, clips, frames, action, previous)
            termination = self.mdp.soft_termination(state, clips, frames)
            rollout.append(state, action, reward, termination, frames, goal=goal)
            observation = self.mdp.observation(state, action, clips, frames, context)
            history = torch.cat((history[:, 1:], observation[:, None]), dim=1)
            previous = action
            actions.append(action)
        return rollout, actions, state

    def test_rotation_math_against_scipy_and_zero_gradients(self):
        """旋转向量与四元数结果匹配 SciPy, 零增量和接近 pi 时梯度有限."""

        vectors = torch.tensor([[0, 0, 0], [0.3, -0.2, 0.5], [np.pi - 1e-7, 0, 0], [0, np.pi, 0]], dtype=torch.float64, requires_grad=True)
        quaternion = _rotation_quaternion(vectors)
        np.testing.assert_allclose(quaternion.detach().numpy(), Rotation.from_rotvec(vectors.detach().numpy()).as_quat(), atol=1e-12)
        recovered = _rotation_vector(quaternion)
        np.testing.assert_allclose(Rotation.from_rotvec(recovered.detach().numpy()).as_matrix(), Rotation.from_rotvec(vectors.detach().numpy()).as_matrix(), atol=1e-12)
        torch.testing.assert_close(_rotation_vector(-quaternion), recovered)
        (quaternion.sum() + recovered.sum()).backward()
        self.assertTrue(torch.isfinite(vectors.grad).all())
        self.assertTrue(torch.autograd.gradcheck(_rotation_quaternion, (vectors.detach()[:2].requires_grad_(),), atol=1e-5))
        left, right = quaternion.detach()[1], quaternion.detach()[2]
        expected = Rotation.from_quat(left.numpy()) * Rotation.from_quat(right.numpy())
        np.testing.assert_allclose(quaternion_to_matrix(_quaternion_product(left, right)).numpy(), expected.as_matrix(), atol=1e-12)

    def test_dynamics_increment_roundtrip_and_heading_equivariance(self):
        """预测与监督增量互相对应, 整体水平旋转和平移不改变局部动力学."""

        model = self.agent().world_model
        state = self.reference(torch.tensor([0, 1]), torch.tensor([4, 7]))
        delta = torch.linspace(-0.01, 0.01, 70)
        with torch.no_grad():
            model.net.net[-1].bias.copy_(delta / model.out_norm.std())
        action = torch.randn(2, 29)
        context = torch.zeros(2, 33)
        context[:, 0] = 1
        predicted = model.step(state, action, context)
        torch.testing.assert_close(model.targets(state, predicted), delta.expand(2, -1), atol=3e-7, rtol=2e-4)
        torch.testing.assert_close(predicted["root_quat"].norm(dim=-1), torch.ones(2))
        yaw = torch.tensor(Rotation.from_euler("z", 0.8).as_quat(), dtype=torch.float32).expand(2, -1)
        matrix = quaternion_to_matrix(yaw)
        shift = torch.tensor([2., -3., 0.])
        transformed = {**state, "root_pos": (matrix @ state["root_pos"][..., None]).squeeze(-1) + shift, "root_quat": _quaternion_product(yaw, state["root_quat"]), "root_lin_vel": (matrix @ state["root_lin_vel"][..., None]).squeeze(-1), "root_ang_vel": (matrix @ state["root_ang_vel"][..., None]).squeeze(-1)}
        torch.testing.assert_close(model.inputs(transformed, action, context), model.inputs(state, action, context), atol=3e-7, rtol=2e-5)
        output = model.step(transformed, action, context)
        torch.testing.assert_close(output["root_pos"], (matrix @ predicted["root_pos"][..., None]).squeeze(-1) + shift)
        torch.testing.assert_close(quaternion_to_matrix(output["root_quat"]), matrix @ quaternion_to_matrix(predicted["root_quat"]))

    def test_policy_contract_and_unsupported_extensions(self):
        """统一输出 C=1 的动作序列, 批量和轨迹前导维都保留, 未实现接口明确报错."""

        policy = self.agent().policy
        self.assertEqual(policy.predict_actions(torch.zeros(5, 3, 99), torch.zeros(5, 146)).shape, (5, 1, 29))
        self.assertEqual(policy(torch.zeros(2, 5, 3, 99), torch.zeros(2, 5, 146)).shape, (2, 5, 1, 29))
        self.assertEqual(policy(torch.zeros(3, 99), torch.zeros(146)).shape, (1, 29))
        for config in (replace(self.config, policy_type="flow_matching"), replace(self.config, chunk_length=4, execute_steps=2)):
            with self.assertRaises(NotImplementedError):
                Policy(config)
        with self.assertRaises(ValueError):
            policy(torch.zeros(5, 2, 99), torch.zeros(5, 146))

    def test_world_model_update_and_masked_invalid_data(self):
        """模型能学习小批连续转移, 无效时间位置的 NaN 不污染损失或梯度."""

        agent = self.agent()
        windows = self.windows()
        agent.world_model.update_statistics(windows)
        mask = torch.ones(5, 3, dtype=torch.bool)
        before = agent.world_model_loss(windows, mask)[0].item()
        for _ in range(12):
            info = agent.update_world_model(windows, mask)
            self.assertTrue(torch.isfinite(info["loss"]))
        after = agent.world_model_loss(windows, mask)[0].item()
        self.assertLess(after, before)
        windows["terminated"][0, 0] = True
        mask[0, 1:] = False
        for value in windows["next_state"].values():
            value[0, 1:] = float("nan")
        windows["action"][0, 1:] = float("nan")
        info = agent.update_world_model(windows, mask)
        self.assertTrue(torch.isfinite(info["loss"]))
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in agent.world_model.parameters()))
        with self.assertRaises(ValueError):
            agent.world_model_loss(windows, torch.ones_like(mask))

    def test_lambda_returns_and_policy_survival_objective(self):
        """用手算值验证软终止递推和奖励的时序权重, 目标不保留原计算图."""

        rewards = torch.tensor([[1.], [2.], [3.]], requires_grad=True)
        terminations = torch.tensor([[0.], [0.5], [1.]])
        values = torch.tensor([[10.], [20.], [30.], [40.]], requires_grad=True)
        targets = lambda_returns(rewards, terminations, values, 0.9, 0.8)
        torch.testing.assert_close(targets, torch.tensor([[8.7616], [5.78], [3.]]))
        self.assertFalse(targets.requires_grad)
        agent = self.agent()
        agent.config = replace(agent.config, gamma=0.9)
        with agent.policy_phase():
            container, _, _ = self.imagine(agent, batch=1)
            rollout = container.as_dict()
            rollout["rewards"] = rewards
            rollout["soft_terminations"] = torch.tensor([[0.], [0.5], [0.25]])
            with torch.no_grad():
                agent.critic.net.net[-1].weight.zero_()
                agent.critic.net.net[-1].bias.fill_(40)
            loss, _ = agent.policy_loss(rollout)
            self.assertAlmostEqual(loss.item(), -14.95, places=4)

    def test_policy_gradient_freeze_and_fixed_critic_targets(self):
        """固定模型参数时仍有动作导数, 策略和 Critic 更新不会改动彼此或动力学."""

        agent = self.agent()
        with torch.no_grad():
            agent.world_model.net.net[-1].weight.normal_(std=0.02)
            agent.world_model.out_norm.var.fill_(0.01)
        world_before = [parameter.detach().clone() for parameter in agent.world_model.parameters()]
        policy_before = [parameter.detach().clone() for parameter in agent.policy.parameters()]
        critic_before = [parameter.detach().clone() for parameter in agent.critic.parameters()]
        with agent.policy_phase():
            self.assertTrue(all(not parameter.requires_grad for parameter in agent.world_model.parameters()))
            container, actions, terminal = self.imagine(agent)
            derivative = torch.autograd.grad(terminal["joint_pos"].sum(), actions[0], retain_graph=True)[0]
            self.assertTrue(torch.isfinite(derivative).all())
            self.assertGreater(derivative.abs().sum().item(), 0)
            rollout = container.as_dict()
            goal = self.mdp.goal_features(rollout["clip_id"], rollout["frame_ids"][-1])
            features = self.mdp.critic_features(terminal, rollout["clip_id"], rollout["frame_ids"][-1], rollout["context"])
            derivative = torch.autograd.grad(agent.critic(features, goal).sum(), terminal["joint_pos"], retain_graph=True)[0]
            self.assertTrue(torch.isfinite(derivative).all())
            self.assertGreater(derivative.abs().sum().item(), 0)
            info = agent.update_policy(rollout)
            self.assertTrue(torch.isfinite(info["loss"]))
            self.assertTrue(all(parameter.grad is None for parameter in agent.world_model.parameters()))
            self.assertTrue(all(parameter.grad is None for parameter in agent.critic.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in agent.world_model.parameters()))
        self.assertTrue(any(not torch.equal(before, after) for before, after in zip(policy_before, agent.policy.parameters())))
        for before, after in zip(world_before, agent.world_model.parameters()):
            torch.testing.assert_close(before, after)
        for before, after in zip(critic_before, agent.critic.parameters()):
            torch.testing.assert_close(before, after)
        batch = agent.prepare_critic_batch(rollout)
        targets = batch["targets"].clone()
        norm_count = agent.critic.norm.count.item()
        policy_after = [parameter.detach().clone() for parameter in agent.policy.parameters()]
        info = agent.update_critic(batch, generator=torch.Generator().manual_seed(8))
        self.assertTrue(torch.isfinite(info["loss"]))
        torch.testing.assert_close(batch["targets"], targets)
        self.assertEqual(agent.critic.norm.count.item(), norm_count)
        self.assertEqual(agent.updates, {"world_model": 0, "policy": 1, "critic": 8})
        self.assertTrue(any(not torch.equal(before, after) for before, after in zip(critic_before, agent.critic.parameters())))
        for before, after in zip(policy_after, agent.policy.parameters()):
            torch.testing.assert_close(before, after)
        container.clear()

    def test_phase_exception_restore_and_statistics_guard(self):
        """异常时恢复原冻结标志, 拒绝在策略图仍使用统计量时修改它们."""

        agent = self.agent()
        next(agent.critic.parameters()).requires_grad_(False)
        before = [parameter.requires_grad for parameter in agent.critic.parameters()]
        with self.assertRaisesRegex(RuntimeError, "故意中断"):
            with agent.policy_phase():
                with self.assertRaises(RuntimeError):
                    agent.update_statistics(None, None, None)
                with self.assertRaises(RuntimeError):
                    agent.prepare_critic_batch({})
                raise RuntimeError("故意中断")
        self.assertEqual([parameter.requires_grad for parameter in agent.critic.parameters()], before)
        self.assertFalse(agent._policy_phase_active)
        with self.assertRaises(RuntimeError):
            agent.update_policy({})

    def test_disconnected_or_nonfinite_loss_does_not_update(self):
        """即使输入叶子保留梯度, 未连到待更新网络的损失也不能被当作成功训练."""

        agent = self.agent()
        before = [parameter.detach().clone() for parameter in agent.policy.parameters()]
        for loss, error in ((torch.tensor(1., requires_grad=True), ValueError), (torch.tensor(float("nan"), requires_grad=True), FloatingPointError)):
            with self.assertRaises(error):
                agent._apply_update(agent.policy_optimizer, agent.policy.parameters(), loss)
        for actual, expected in zip(agent.policy.parameters(), before):
            torch.testing.assert_close(actual, expected)
            self.assertIsNone(actual.grad)

    def test_checkpoint_restores_optimizer_statistics_and_next_update(self):
        """恢复后下一次模型更新与原实例一致, 统计量和优化器动量都得到恢复."""

        agent = self.agent()
        windows = self.windows()
        mask = torch.ones(5, 3, dtype=torch.bool)
        agent.world_model.update_statistics(windows)
        agent.policy.update_statistics(torch.randn(5, 99), torch.randn(5, 146))
        agent.update_world_model(windows, mask)
        with agent.policy_phase():
            container, _, _ = self.imagine(agent)
            agent.update_policy(container.as_dict())
        agent.update_critic(agent.prepare_critic_batch(container.as_dict()))
        agent.set_iteration(self.config.lr_decay_iterations)
        self.assertAlmostEqual(agent.policy_optimizer.param_groups[0]["lr"], self.config.learning_rate * self.config.min_lr_ratio)
        stream = io.BytesIO()
        torch.save(agent.state_dict(), stream)
        stream.seek(0)
        restored = self.agent(seed=99)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        self.assertEqual(restored.iteration, agent.iteration)
        self.assertEqual(restored.updates, agent.updates)
        torch.testing.assert_close(restored.policy.goal_norm.mean, agent.policy.goal_norm.mean)
        torch.testing.assert_close(restored.critic.norm.var, agent.critic.norm.var)
        original_info = agent.update_world_model(windows, mask)
        restored_info = restored.update_world_model(windows, mask)
        torch.testing.assert_close(original_info["loss"], restored_info["loss"], rtol=0, atol=0)
        for actual, expected in zip(restored.world_model.parameters(), agent.world_model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        incompatible = deepcopy(restored.state_dict())
        incompatible["config"]["history_length"] = 100
        with self.assertRaises(ValueError):
            restored.load_state_dict(incompatible)


if __name__ == "__main__":
    unittest.main()

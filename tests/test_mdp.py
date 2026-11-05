"""验证任务计算, 坐标约定与可微性; 不启动 Isaac Lab 或仿真窗口.

运行: python -m unittest discover -s tests -p test_mdp.py -v
"""

import ast
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from sim_platform.data.kinematics import G1Kinematics
from sim_platform.data.motion import BASE_STATE_FIELDS, MotionLib
from sim_platform.sim.mdp import TrackingMDP, TrackingMDPConfig, flatten_state, invariant_features, normalize_context, unflatten_state
from sim_platform.wrappers.rotations import heading_matrix, matrix_to_rotation_6d, quaternion_to_matrix, rotation_error_squared


class RotationTest(unittest.TestCase):
    """以 SciPy 的独立实现检查 xyzw 约定, 零误差与接近 180 度的数值稳定性."""

    def test_rotation_values_and_gradients(self):
        rotations = Rotation.from_rotvec([[0, 0, 0], [0.3, -0.5, 0.7], [np.pi, 0, 0], [0, np.pi - 1e-7, 0]])
        quaternion = torch.tensor(rotations.as_quat(), dtype=torch.float64, requires_grad=True)
        actual = quaternion_to_matrix(quaternion)
        np.testing.assert_allclose(actual.detach().numpy(), rotations.as_matrix(), atol=1e-12)
        squared = rotation_error_squared(torch.eye(3, dtype=torch.float64), actual)
        np.testing.assert_allclose(squared.detach().numpy(), rotations.magnitude() ** 2, atol=1e-12)
        torch.testing.assert_close(rotation_error_squared(actual, actual), torch.zeros(4, dtype=torch.float64), atol=1e-12, rtol=0)
        squared.sum().backward()
        self.assertTrue(torch.isfinite(quaternion.grad).all())
        self.assertTrue(torch.autograd.gradcheck(lambda q: rotation_error_squared(torch.eye(3, dtype=q.dtype), quaternion_to_matrix(q)), (quaternion.detach()[:2].requires_grad_(),), atol=1e-5))

    def test_heading_and_6d_layout(self):
        matrix = torch.tensor(Rotation.from_euler("xyz", [0.2, -0.3, 0.7]).as_matrix(), dtype=torch.float64)
        np.testing.assert_allclose(heading_matrix(matrix).numpy(), Rotation.from_euler("z", 0.7).as_matrix(), atol=1e-12)
        torch.testing.assert_close(matrix_to_rotation_6d(matrix), matrix[:, :2].reshape(6))
        vertical = torch.tensor(Rotation.from_euler("y", np.pi / 2).as_matrix(), dtype=torch.float64, requires_grad=True)
        heading = heading_matrix(vertical)
        torch.testing.assert_close(heading, torch.eye(3, dtype=torch.float64))
        heading.sum().backward()
        self.assertTrue(torch.isfinite(vertical.grad).all())
        with self.assertRaises(ValueError):
            quaternion_to_matrix(torch.zeros(4))


class TrackingMDPTest(unittest.TestCase):
    """用真实 G1 结构和短合成动作检查奖励, 目标编码, 终止与梯度."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        path = Path(cls.directory.name)
        assets = PROJECT / "sim_platform/assets/robots/g1"
        tree = ast.parse((assets / "robot.py").read_text())
        names = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "G1_JOINT_NAMES" for target in node.targets))
        cls.config = TrackingMDPConfig(goal_frames=3)
        cls.kin = G1Kinematics(assets / "g1_29dof.xml", names, cls.config.tracked_bodies)
        cls.middle = (cls.kin.q_lo + cls.kin.q_hi) / 2
        time = np.arange(31) / 30
        quaternion = (Rotation.from_rotvec(np.stack((np.zeros_like(time), np.zeros_like(time), 0.3 + 0.4 * time), axis=-1)) * Rotation.from_euler("xy", [0.2, -0.1])).as_quat()
        for index in range(2):
            position = np.stack((time + 3 * index, 0.2 * time, 1 + 0.1 * time), axis=-1)
            joints = cls.middle.numpy() + time[:, None] * 0.01
            np.savetxt(path / f"clip{index}.csv", np.concatenate((position, quaternion, joints), axis=-1), delimiter=",")
        cls.motion = MotionLib(path, kinematics=cls.kin, fk_batch_size=11)
        cls.mdp = TrackingMDP(cls.kin, cls.motion, cls.middle, cls.config)

    def reference(self, clip_id=0, frame_id=10):
        return self.motion.get_reference(clip_id, frame_id, include_body=False)

    def test_state_roundtrip_and_batch_layout(self):
        state = self.reference([0, 1], [10, 20])
        state = {key: value.unsqueeze(0).expand(3, -1, -1).clone().requires_grad_() for key, value in state.items()}
        packed = flatten_state(state)
        self.assertEqual(packed.shape, (3, 2, 71))
        restored = unflatten_state(packed)
        for key in BASE_STATE_FIELDS:
            torch.testing.assert_close(restored[key], state[key])
        zero = torch.zeros(3, 2, 29)
        self.assertEqual(self.mdp.observation(state, zero, [0, 1], [10, 20]).shape, (3, 2, 99))
        self.assertEqual(self.mdp.reward(state, [0, 1], [10, 20], zero, zero).shape, (3, 2))
        self.assertEqual(self.mdp.critic_features(state, [0, 1], [10, 20]).shape, (3, 2, 184))
        packed.sum().backward()
        for key in BASE_STATE_FIELDS:
            torch.testing.assert_close(state[key].grad, torch.ones_like(state[key]))

    def test_observation_and_noise(self):
        state = self.reference()
        previous = torch.linspace(-0.2, 0.2, 29)
        context = torch.zeros(33)
        context[0], context[1:30] = 1.0, 0.01
        observation = self.mdp.observation(state, previous, 0, 10, context)
        rotation = Rotation.from_quat(state["root_quat"].numpy()).as_matrix()
        expected = np.concatenate((state["joint_pos"].numpy() - self.middle.numpy() - 0.01, state["joint_vel"].numpy(), rotation.T @ state["root_ang_vel"].numpy(), -rotation[2], previous.numpy(), np.eye(3)[:, :2].reshape(6)))
        np.testing.assert_allclose(observation.numpy(), expected, atol=2e-6)
        source = observation.expand(100, -1).clone()
        noisy = self.mdp.add_observation_noise(source, generator=torch.Generator().manual_seed(10))
        self.assertTrue(((noisy - source).abs() <= self.mdp.obs_noise + 1e-6).all())
        torch.testing.assert_close(noisy[:, 64:93], source[:, 64:93])
        torch.testing.assert_close(source, observation.expand_as(source))

    def test_reward_identity_penalties_and_dt(self):
        state = self.reference()
        zero = torch.zeros(29)
        reward, terms = self.mdp.reward(state, 0, 10, zero, zero, return_terms=True)
        self.assertAlmostEqual(reward.item(), 0.1, places=6)
        for name in ("body_pos", "body_ori", "body_lin_vel", "body_ang_vel", "root_pos", "root_ori"):
            self.assertLess(terms[f"error_{name}"].abs().item(), 1e-10)
        changed = self.mdp.reward(state, 0, 10, torch.ones(29), zero)
        self.assertAlmostEqual((changed - reward).item(), -0.1 * 29 * 0.02, places=6)
        outside = {key: value.clone() for key, value in state.items()}
        outside["joint_pos"][0] = self.mdp.soft_high[0] + 0.2
        _, penalized = self.mdp.reward(outside, 0, 10, zero, zero, return_terms=True)
        self.assertAlmostEqual(penalized["joint_limit"].item(), -2 * 0.2 * 0.02, places=6)

    def test_alignment_preserves_local_pose_but_root_tracks_original(self):
        state = self.reference()
        delta = Rotation.from_euler("z", 0.8)
        changed = {key: value.clone() for key, value in state.items()}
        changed["root_pos"][:2] += torch.tensor([2.0, -1.0])
        changed["root_quat"] = torch.tensor((delta * Rotation.from_quat(state["root_quat"].numpy())).as_quat(), dtype=torch.float32)
        zero = torch.zeros(29)
        _, terms = self.mdp.reward(changed, 0, 10, zero, zero, return_terms=True)
        self.assertLess(terms["error_body_pos"].item(), 1e-11)
        self.assertLess(terms["error_body_ori"].item(), 1e-11)
        self.assertAlmostEqual(terms["error_root_pos"].item(), 5.0, places=5)
        self.assertAlmostEqual(terms["error_root_ori"].item(), 0.8**2, places=5)
        actual = self.kin.forward(**changed)
        reference = self.motion.get_reference(0, 10)
        expected_velocity_error = (reference["body_lin_vel"] - actual["body_lin_vel"]).square().sum(-1).mean()
        torch.testing.assert_close(terms["error_body_lin_vel"], expected_velocity_error)

    def test_goal_features_and_clip_end(self):
        goal = self.mdp.goal_features(0, 49).reshape(3, 73)
        self.assertEqual(self.mdp.goal_features([0, 1], [10, 20]).shape, (2, 219))
        self.assertEqual(self.mdp.goal_features(torch.zeros(2, 3, dtype=torch.long), 10).shape, (2, 3, 219))
        current = self.reference(0, 49)
        final = self.reference(0, 50)
        rotation = Rotation.from_quat(current["root_quat"].numpy())
        forward = rotation.as_matrix()[:2, 0]
        heading = Rotation.from_euler("z", np.arctan2(forward[1], forward[0]))
        final_rotation = Rotation.from_quat(final["root_quat"].numpy())
        expected = np.concatenate((heading.inv().apply((final["root_pos"] - current["root_pos"]).numpy()), (heading.inv() * final_rotation).as_matrix()[:, :2].reshape(6), final_rotation.inv().apply(final["root_lin_vel"].numpy()), final_rotation.inv().apply(final["root_ang_vel"].numpy()), final["joint_pos"].numpy(), 0.1 * final["joint_vel"].numpy()))
        np.testing.assert_allclose(goal[0].numpy(), expected, atol=2e-6)
        torch.testing.assert_close(goal[1:, 9:15], torch.zeros(2, 6))
        torch.testing.assert_close(goal[1:, 44:], torch.zeros(2, 29))
        torch.testing.assert_close(goal[1:, 15:44], final["joint_pos"].expand(2, -1))

    def test_termination_thresholds_and_soft_gradient(self):
        state = self.reference()
        changed = {key: value.clone() for key, value in state.items()}
        for offset, expected in ((0.249, False), (0.251, True)):
            changed["root_pos"][2] = state["root_pos"][2] + offset
            self.assertEqual(self.mdp.termination(changed, 0, 10).item(), expected)
        changed["root_pos"] = state["root_pos"].clone()
        reference_tilt = Rotation.from_quat(state["root_quat"].numpy()).as_matrix()[2, 2]
        for difference, expected in ((0.799, False), (0.801, True)):
            changed["root_quat"] = torch.tensor(Rotation.from_euler("x", np.arccos(reference_tilt - difference)).as_quat(), dtype=torch.float32)
            self.assertEqual(self.mdp.termination(changed, 0, 10).item(), expected)
        changed = {key: value.clone() for key, value in state.items()}
        changed["root_pos"][2] += 0.24
        changed["root_pos"].requires_grad_()
        strength = self.mdp.soft_termination(changed, 0, 10)
        self.assertTrue(0 < strength.item() < 1)
        strength.backward()
        self.assertGreater(changed["root_pos"].grad[2].item(), 0)

    def test_invariant_and_critic_features(self):
        state = self.reference()
        original = invariant_features(state)
        delta = Rotation.from_euler("z", 1.2)
        transformed = {key: value.clone() for key, value in state.items()}
        transformed["root_quat"] = torch.tensor((delta * Rotation.from_quat(state["root_quat"].numpy())).as_quat(), dtype=torch.float32)
        for key in ("root_lin_vel", "root_ang_vel"):
            transformed[key] = torch.tensor(delta.apply(state[key].numpy()), dtype=torch.float32)
        transformed["root_pos"][:2] += 100
        torch.testing.assert_close(invariant_features(transformed), original, atol=2e-6, rtol=2e-6)
        context = torch.cat((torch.tensor([0.75]), torch.full((29,), 0.01), torch.full((3,), 0.05)))
        features = self.mdp.critic_features(state, 0, 10, context)
        self.assertEqual(features.shape, (184,))
        torch.testing.assert_close(features[:71], original)
        torch.testing.assert_close(features[71:145], torch.zeros(74), atol=1e-6, rtol=0)
        torch.testing.assert_close(features[-33:], normalize_context(context))

    def test_reward_and_critic_keep_state_gradients(self):
        state = self.reference()
        state["root_pos"] = state["root_pos"] + torch.tensor([0.02, 0.01, 0.04])
        state["joint_pos"] = state["joint_pos"] + 0.02
        state["joint_vel"] = state["joint_vel"] + 0.1
        state = {key: value.detach().requires_grad_() for key, value in state.items()}
        action = torch.full((29,), 0.1, requires_grad=True)
        loss = self.mdp.reward(state, 0, 10, action, torch.zeros(29)) + 1e-4 * self.mdp.critic_features(state, 0, 10).square().sum()
        loss.backward()
        for key in BASE_STATE_FIELDS:
            self.assertIsNotNone(state[key].grad, key)
            self.assertTrue(torch.isfinite(state[key].grad).all(), key)
        self.assertGreater(action.grad.abs().sum().item(), 0)
        # 用中心有限差分独立检查经过 FK 的关节奖励梯度.
        base = {key: value.detach() for key, value in state.items()}
        joint = base["joint_pos"].clone().requires_grad_()
        target = dict(base, joint_pos=joint)
        derivative = torch.autograd.grad(self.mdp.reward(target, 0, 10, action.detach(), torch.zeros(29)), joint)[0][3]
        epsilon = 1e-3
        plus, minus = joint.detach().clone(), joint.detach().clone()
        plus[3] += epsilon
        minus[3] -= epsilon
        numeric = (self.mdp.reward(dict(base, joint_pos=plus), 0, 10, action.detach(), torch.zeros(29)) - self.mdp.reward(dict(base, joint_pos=minus), 0, 10, action.detach(), torch.zeros(29))) / (2 * epsilon)
        torch.testing.assert_close(derivative, numeric, atol=2e-5, rtol=0.01)

    def test_real_motion_integration(self):
        """检查已下载动作经 MotionLib/FK 缓存后能直接用于任务计算."""
        source = next((PROJECT / "sim_platform/data/lafan1_g1").glob("*.csv"))
        with tempfile.TemporaryDirectory() as directory:
            # 只用真实动作开头 64 帧, 避免测试分配整个数据集的缓存.
            raw = np.loadtxt(source, delimiter=",", max_rows=64)
            np.savetxt(Path(directory) / source.name, raw, delimiter=",")
            motion = MotionLib(directory, kinematics=self.kin, fk_batch_size=17)
        mdp = TrackingMDP(self.kin, motion, self.middle, self.config)
        state = motion.get_reference([0, 0], [20, 50], include_body=False)
        zero = torch.zeros(2, 29)
        reward, terms = mdp.reward(state, 0, [20, 50], zero, zero, return_terms=True)
        for name in ("body_pos", "body_ori", "body_lin_vel", "body_ang_vel", "root_pos", "root_ori"):
            self.assertLess(terms[f"error_{name}"].abs().max().item(), 1e-8)
        self.assertTrue(torch.isfinite(reward).all())
        self.assertFalse(mdp.termination(state, 0, [20, 50]).any())
        self.assertEqual(mdp.observation(state, zero, 0, [20, 50]).shape, (2, 99))
        self.assertEqual(mdp.goal_features([0, 0], [20, 50]).shape, (2, 219))

    def test_configuration_and_input_errors(self):
        with self.assertRaises(ValueError):
            replace(self.config, tracking_stds=(0.0,) * 6)
        with self.assertRaises(ValueError):
            TrackingMDP(self.kin, self.motion, self.middle, replace(self.config, tracked_bodies=tuple(reversed(self.config.tracked_bodies))))
        with self.assertRaises(ValueError):
            TrackingMDP(self.kin, self.motion, self.middle, replace(self.config, control_dt=0.01))
        state = self.reference()
        with self.assertRaises(ValueError):
            self.mdp.observation(state, torch.zeros(28), 0, 10)
        with self.assertRaises(ValueError):
            self.mdp.goal_features(0.1, 10)
        with self.assertRaises(IndexError):
            self.mdp.reward(state, 0, 51, torch.zeros(29), torch.zeros(29))


if __name__ == "__main__":
    unittest.main()

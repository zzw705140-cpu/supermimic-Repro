"""验证长期与短期数据契约、错误定位及检查后的梯度完整性.

运行: python -m unittest discover -s tests -p test_validation.py -v
"""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.data.validation import validate_imagined_rollout, validate_replay_data


def make_state(*leading):
    state = {
        "root_pos": torch.zeros(*leading, 3),
        "root_quat": torch.zeros(*leading, 4),
        "root_lin_vel": torch.zeros(*leading, 3),
        "root_ang_vel": torch.zeros(*leading, 3),
        "joint_pos": torch.zeros(*leading, 29),
        "joint_vel": torch.zeros(*leading, 29),
    }
    state["root_quat"][..., 3] = 1
    return state


def make_replay(*leading):
    return {
        "state": make_state(*leading),
        "next_state": make_state(*leading),
        "action": torch.zeros(*leading, 29),
        "previous_action": torch.zeros(*leading, 29),
        "context": torch.zeros(*leading, 33),
        "clip_id": torch.zeros(leading, dtype=torch.int64),
        "frame_id": torch.zeros(leading, dtype=torch.int64),
        "episode_step": torch.zeros(leading, dtype=torch.int64),
        "terminated": torch.zeros(leading, dtype=torch.bool),
        "truncated": torch.zeros(leading, dtype=torch.bool),
    }


def make_rollout(horizon=3, batch=2, history=10):
    return {
        "states": make_state(horizon + 1, batch),
        "actions": torch.zeros(horizon, batch, 29),
        "initial_previous_action": torch.zeros(batch, 29),
        "initial_observation_history": torch.zeros(batch, history, 99),
        "context": torch.zeros(batch, 33),
        "clip_id": torch.zeros(batch, dtype=torch.int64),
        "frame_ids": torch.arange(horizon + 1)[:, None].expand(-1, batch),
        "rewards": torch.zeros(horizon, batch),
        "soft_terminations": torch.zeros(horizon, batch),
    }


class ValidationTest(unittest.TestCase):
    def test_replay_single_step_and_time_batch(self):
        for leading in ((2,), (3, 2)):
            with self.subTest(leading=leading):
                data = make_replay(*leading)
                self.assertIsNone(validate_replay_data(data, check_values=True, device="cpu"))
                data["reward"] = torch.zeros(leading)
                validate_replay_data(data, check_values=True)

    def test_missing_unknown_and_non_tensor_fields(self):
        data = make_replay(2)
        del data["state"]["joint_pos"]
        with self.assertRaisesRegex(ValueError, "replay.state.*joint_pos"):
            validate_replay_data(data)
        data = make_replay(2)
        data["env_id"] = torch.arange(2)
        with self.assertRaisesRegex(ValueError, "未知字段.*env_id"):
            validate_replay_data(data)
        data = make_replay(2)
        data["action"] = [[0.0] * 29] * 2
        with self.assertRaisesRegex(TypeError, "replay.action"):
            validate_replay_data(data)

    def test_replay_shape_dtype_and_device_errors(self):
        changes = (
            ("previous_action", torch.zeros(3, 29)),
            ("clip_id", torch.zeros(2)),
            ("terminated", torch.zeros(2, dtype=torch.int64)),
            ("context", torch.zeros(2, 33, device="meta")),
        )
        for key, value in changes:
            with self.subTest(key=key):
                data = make_replay(2)
                data[key] = value
                with self.assertRaisesRegex(ValueError, f"replay.{key}"):
                    validate_replay_data(data)
        data = make_replay(2)
        data["next_state"]["joint_vel"] = torch.zeros(2, 28)
        with self.assertRaisesRegex(ValueError, "replay.next_state.joint_vel"):
            validate_replay_data(data)
        with self.assertRaisesRegex(ValueError, "期望设备"):
            validate_replay_data(make_replay(2), device="meta")

    def test_rollout_different_time_axes_and_custom_lengths(self):
        validate_imagined_rollout(make_rollout(), check_values=True)
        data = make_rollout(horizon=1, batch=1, history=4)
        data["goals"] = torch.zeros(2, 1, 3 * 73)
        validate_imagined_rollout(data, history_length=4, goal_frames=3, check_values=True)
        with self.assertRaisesRegex(ValueError, "initial_observation_history"):
            validate_imagined_rollout(data)
        with self.assertRaisesRegex(ValueError, "rollout.goals"):
            validate_imagined_rollout(data, history_length=4)
        for key, value in (("rewards", torch.zeros(4, 2)), ("context", torch.zeros(3, 2, 33))):
            with self.subTest(key=key):
                data = make_rollout()
                data[key] = value
                with self.assertRaisesRegex(ValueError, f"rollout.{key}"):
                    validate_imagined_rollout(data)

    def test_optional_value_checks(self):
        for key, value in (("context", float("nan")), ("previous_action", float("inf")), ("frame_id", -1)):
            with self.subTest(key=key):
                data = make_replay(2)
                data[key].fill_(value)
                validate_replay_data(data)      # 结构检查不扫描数值
                with self.assertRaisesRegex(ValueError, f"replay.{key}"):
                    validate_replay_data(data, check_values=True)
        data = make_replay(2)
        data["state"]["root_quat"].zero_()
        with self.assertRaisesRegex(ValueError, "replay.state.root_quat"):
            validate_replay_data(data, check_values=True)
        data = make_rollout()
        data["soft_terminations"][0, 0] = 1.1
        with self.assertRaisesRegex(ValueError, "soft_terminations"):
            validate_imagined_rollout(data, check_values=True)

    def test_empty_axes_and_invalid_config(self):
        with self.assertRaisesRegex(ValueError, "大于 0"):
            validate_replay_data(make_replay(0))
        with self.assertRaisesRegex(ValueError, "大于 0"):
            validate_imagined_rollout(make_rollout(horizon=0))
        for name in ("history_length", "goal_frames"):
            for value in (True, 0, -1, 1.5):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        validate_imagined_rollout(make_rollout(), **{name: value})

    def test_checks_preserve_policy_gradient_and_tensor_identity(self):
        data = make_rollout()
        policy_parameter = torch.tensor(0.2, requires_grad=True)
        actions = policy_parameter.expand(3, 2, 29)
        predicted_joints = torch.cat((torch.zeros(1, 2, 29), actions.cumsum(0)))
        rewards = -predicted_joints[1:].square().sum(-1)
        data["actions"] = actions
        data["states"]["joint_pos"] = predicted_joints
        data["rewards"] = rewards
        graph = rewards.grad_fn
        validate_imagined_rollout(data, check_values=True)
        self.assertIs(data["actions"], actions)
        self.assertIs(data["states"]["joint_pos"], predicted_joints)
        self.assertIs(data["rewards"].grad_fn, graph)
        (-data["rewards"].sum()).backward()
        # 两条轨迹、29 个关节, 三步关节位置依次为 p、2p、3p.
        self.assertAlmostEqual(policy_parameter.grad.item(), 2 * 2 * 29 * 14 * 0.2, places=4)
        replay = make_replay(2)
        replay["action"] = policy_parameter.expand(2, 29)
        validate_replay_data(replay, check_values=True)
        self.assertIsNotNone(replay["action"].grad_fn)


if __name__ == "__main__":
    unittest.main()

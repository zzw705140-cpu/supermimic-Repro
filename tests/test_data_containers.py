"""检查经验池的数据所有权, 回合边界和持久化, 以及预测轨迹的梯度与释放.

运行: python -m unittest discover -s tests -p test_data_containers.py -v
"""

import gc
from pathlib import Path
import sys
import tempfile
import unittest
import weakref

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.data.imagined_rollout.imagined_rollout import ImaginedRollout
from supermimic.data.replay_buffer.replay_buffer import ReplayBuffer
from supermimic.data.validation import STATE_WIDTHS, validate_imagined_rollout


def make_state(batch):
    """构造 CPU 六字段状态, xyzw 四元数初始化为单位旋转."""

    state = {key: torch.zeros(batch, width) for key, width in STATE_WIDTHS.items()}
    state["root_quat"][:, 3] = 1
    return state


def transition(time, num_envs=3, *, steps=None, terminated=(), truncated=(), reward=False):
    """用状态数值编码时间和环境编号, 便于独立检查采样是否串行或串环境."""

    state, next_state = make_state(num_envs), make_state(num_envs)
    marker = torch.arange(num_envs) * 100 + time
    state["joint_pos"][:] = marker[:, None]
    next_state["joint_pos"][:] = marker[:, None] + 1
    data = {"state": state, "next_state": next_state, "action": torch.full((num_envs, 29), float(time)), "previous_action": torch.full((num_envs, 29), float(time - 1)), "context": torch.zeros(num_envs, 33), "clip_id": torch.arange(num_envs), "frame_id": torch.full((num_envs,), time + 10), "episode_step": torch.tensor(steps if steps is not None else [time] * num_envs), "terminated": torch.tensor([env in terminated for env in range(num_envs)]), "truncated": torch.tensor([env in truncated for env in range(num_envs)])}
    if reward:
        data["reward"] = torch.full((num_envs,), float(time))
    return data


def make_rollout(*, horizon=3, batch=2, history=10, goal_frames=10, goal=False):
    """构造可自定义 H/B/L/K 的容器, 不依赖真实物理环境数量."""

    return ImaginedRollout(make_state(batch), torch.zeros(batch, 29), torch.zeros(batch, history, 99), torch.zeros(batch, 33), torch.zeros(batch, dtype=torch.int64), torch.zeros(batch, dtype=torch.int64), horizon=horizon, history_length=history, goal_frames=goal_frames, initial_goal=torch.zeros(batch, goal_frames * 73) if goal else None, check_values=True)


class ReplayBufferTest(unittest.TestCase):
    """用可追溯的时间和环境标记验证真实交互的长期保存."""

    def test_write_owns_storage_and_failed_validation_keeps_pointer(self):
        """写入复制所有字段并切断梯度, 错误输入不占用一行."""

        buffer = ReplayBuffer(4, 3, check_values=True)
        data = transition(0)
        data["action"].requires_grad_()
        data["state"]["joint_pos"].requires_grad_()
        buffer.add(data)
        with torch.no_grad():
            data["action"].fill_(999)
            data["state"]["joint_pos"].fill_(999)
        self.assertEqual(buffer.data["action"][0, 0, 0].item(), 0)
        self.assertEqual(buffer.data["state"]["joint_pos"][0, 1, 0].item(), 100)
        self.assertFalse(buffer.data["action"].requires_grad)
        self.assertIsNone(buffer.data["state"]["joint_pos"].grad_fn)
        for invalid in (transition(1, 2), transition(1, reward=True)):
            with self.assertRaises(ValueError):
                buffer.add(invalid)
            self.assertEqual((buffer.write_pos, buffer.valid_length, len(buffer)), (1, 1, 3))

    def test_ring_order_environment_axis_and_independent_samples(self):
        """多次环形覆盖后仍按旧到新采样, 每个窗口始终属于同一环境."""

        buffer = ReplayBuffer(4, 3, store_reward=True, check_values=True)
        for time in range(11):
            buffer.add(transition(time, reward=True))
        data, mask = buffer.sample_windows(128, 4, generator=torch.Generator().manual_seed(1))
        times = torch.arange(7, 11).expand(128, 4)
        envs = data["clip_id"][:, :1].expand_as(times)
        torch.testing.assert_close(data["state"]["joint_pos"][..., 0], (envs * 100 + times).float())
        torch.testing.assert_close(data["next_state"]["joint_pos"][..., 0], (envs * 100 + times + 1).float())
        torch.testing.assert_close(data["frame_id"], times + 10)
        torch.testing.assert_close(data["reward"], times.float())
        self.assertTrue(mask.all())
        self.assertEqual(set(envs[:, 0].tolist()), {0, 1, 2})
        data["state"]["joint_pos"].fill_(-1000)
        self.assertTrue((buffer.data["state"]["joint_pos"] >= 0).all())
        self.assertEqual((buffer.write_pos, buffer.valid_length, len(buffer)), (3, 4, 12))

    def test_window_masks_termination_truncation_and_counter_breaks(self):
        """结束转移保留监督, 后续回合屏蔽; 漏写结束标志时也检测步数跳变."""

        buffer = ReplayBuffer(6, 3)
        for time in range(6):
            steps = [time if time < 2 else time - 2, time if time < 3 else time - 3, time if time < 4 else time - 4]
            buffer.add(transition(time, steps=steps, terminated=(0,) if time == 1 else (), truncated=(1,) if time == 2 else ()))
        data, mask = buffer.sample_windows(128, 6, generator=torch.Generator().manual_seed(2))
        expected = torch.tensor([[True, True, False, False, False, False], [True, True, True, False, False, False], [True, True, True, True, False, False]])
        torch.testing.assert_close(mask, expected[data["clip_id"][:, 0]])
        _, single_mask = buffer.sample_windows(128, 1)
        self.assertTrue(single_mask.all())

    def test_start_history_padding_after_wrap_and_partial_resets(self):
        """按各环境自身回合补齐历史, 不越过重置或被覆盖的最旧记录."""

        buffer = ReplayBuffer(5, 3)
        records = []
        for time in range(8):
            steps = [time if time < 5 else time - 5, time if time < 4 else time - 4, time]
            records.append(transition(time, steps=steps, terminated=(0,) if time == 4 else (), truncated=(1,) if time == 3 else ()))
            buffer.add(records[-1])
        start, history = buffer.sample_starts(256, 7, generator=torch.Generator().manual_seed(3))
        observed = set()
        for sample in range(256):
            env, time = start["clip_id"][sample].item(), start["frame_id"][sample].item() - 10
            observed.add((time, env))
            earliest = time
            while earliest > 3:
                previous, current = records[earliest - 1], records[earliest]
                if previous["terminated"][env] or previous["truncated"][env] or current["episode_step"][env] != previous["episode_step"][env] + 1:
                    break
                earliest -= 1
            expected_times = torch.tensor([max(earliest, time - offset) for offset in range(6, -1, -1)])
            torch.testing.assert_close(history["frame_id"][sample], expected_times + 10)
            torch.testing.assert_close(history["state"]["joint_pos"][sample, :, 0], (env * 100 + expected_times).float())
            torch.testing.assert_close(history["previous_action"][sample, :, 0], expected_times.float() - 1)
            self.assertTrue((history["clip_id"][sample] == env).all())
            torch.testing.assert_close(history["state"]["joint_pos"][sample, -1], start["state"]["joint_pos"][sample])
        self.assertEqual(len(observed), 15)
        start, history = buffer.sample_starts(16, 1)
        torch.testing.assert_close(start["frame_id"], history["frame_id"][:, 0])

    def test_save_load_empty_partial_and_wrapped_then_continue(self):
        """保存有效数据和指针, 恢复后继续写入仍得到正确时间顺序."""

        with tempfile.TemporaryDirectory() as directory:
            for count in (0, 2, 9):
                with self.subTest(count=count):
                    metadata = {"control_dt": 0.02, "joint_names": ["test_joint"]}
                    buffer = ReplayBuffer(4, 3, store_reward=True, metadata=metadata)
                    metadata["joint_names"].append("external_change")
                    for time in range(count):
                        buffer.add(transition(time, reward=True))
                    path = Path(directory) / "nested" / "replay.pt"
                    buffer.save(path)
                    restored = ReplayBuffer.load(path)
                    self.assertEqual((restored.capacity, restored.num_envs, restored.write_pos, restored.valid_length), (4, 3, count % 4, min(count, 4)))
                    self.assertEqual(restored.metadata, {"control_dt": 0.02, "joint_names": ["test_joint"]})
                    for time in range(count, count + 2):
                        restored.add(transition(time, reward=True))
                    data, mask = restored.sample_windows(16, restored.valid_length)
                    expected = torch.arange(max(0, count + 2 - 4), count + 2).expand(16, -1)
                    torch.testing.assert_close(data["frame_id"], expected + 10)
                    self.assertTrue(mask.all())
                    self.assertFalse(list(path.parent.glob("*.tmp")))

    def test_invalid_capacity_empty_samples_and_corrupt_snapshot(self):
        """空池和无效采样长度明确报错, 不接受损坏的恢复指针."""

        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                ReplayBuffer(value, 2)
        buffer = ReplayBuffer(4, 3)
        for call in (lambda: buffer.sample_starts(2), lambda: buffer.sample_windows(2, 1), lambda: buffer.sample_windows(0, 1)):
            with self.assertRaises(ValueError):
                call()
        buffer.add(transition(0))
        with self.assertRaises(ValueError):
            buffer.sample_windows(2, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.pt"
            buffer.save(path)
            snapshot = torch.load(path, weights_only=True)
            snapshot["write_pos"] = 3
            torch.save(snapshot, path)
            with self.assertRaisesRegex(ValueError, "write_pos"):
                ReplayBuffer.load(path)


class ImaginedRolloutTest(unittest.TestCase):
    """检查预测轨迹组织与策略梯度, 不在容器内运行实际算法."""

    def test_multistep_gradient_and_detached_critic_view(self):
        """三步可微动力学的解析梯度与容器输出一致, Critic 视图无梯度."""

        rollout = make_rollout(goal=True, goal_frames=2)
        parameter = torch.tensor(0.2, requires_grad=True)
        state = make_state(2)
        for time in range(3):
            action = parameter.expand(2, 29)
            state = {**state, "joint_pos": state["joint_pos"] + action}
            reward = -state["joint_pos"].square().sum(-1)
            rollout.append(state, action, reward, parameter.sigmoid().expand(2), torch.full((2,), time + 1), goal=parameter.expand(2, 146))
        batch = rollout.as_dict()
        validate_imagined_rollout(batch, goal_frames=2, check_values=True)
        self.assertEqual(batch["states"]["joint_pos"].shape, (4, 2, 29))
        self.assertEqual(batch["actions"].shape, (3, 2, 29))
        self.assertEqual(batch["goals"].shape, (4, 2, 146))
        self.assertIs(batch, rollout.as_dict())
        detached = rollout.detached()
        for original, view in ((batch["states"]["joint_pos"], detached["states"]["joint_pos"]), (batch["rewards"], detached["rewards"]), (batch["actions"], detached["actions"])):
            self.assertFalse(view.requires_grad)
            self.assertIsNone(view.grad_fn)
            self.assertEqual(original.data_ptr(), view.data_ptr())
            self.assertTrue(original.requires_grad)
        (-batch["rewards"].sum()).backward()
        self.assertAlmostEqual(parameter.grad.item(), 2 * 2 * 29 * (1 + 4 + 9) * 0.2, places=4)

    def test_custom_lengths_single_step_and_optional_goals(self):
        """自定义 B/H/L/K, 无目标缓存时不产生多余字段."""

        rollout = make_rollout(horizon=1, batch=5, history=4, goal_frames=3)
        rollout.append(make_state(5), torch.zeros(5, 29), torch.zeros(5), torch.zeros(5), torch.ones(5, dtype=torch.int64))
        batch = rollout.as_dict()
        validate_imagined_rollout(batch, history_length=4, goal_frames=3, check_values=True)
        self.assertNotIn("goals", batch)
        self.assertEqual(batch["states"]["root_quat"].shape, (2, 5, 4))

    def test_invalid_append_does_not_advance_and_lifecycle_errors(self):
        """所有检查通过后才收集一整步, 拒绝不完整批次和已释放容器."""

        rollout = make_rollout(horizon=1)
        with self.assertRaises(RuntimeError):
            rollout.as_dict()
        with self.assertRaises(ValueError):
            rollout.append(make_state(2), torch.zeros(2, 29), torch.zeros(2), torch.full((2,), 1.1), torch.ones(2, dtype=torch.int64))
        with self.assertRaises(ValueError):
            rollout.append(make_state(2), torch.zeros(2, 29), torch.zeros(2), torch.zeros(2), torch.ones(2, dtype=torch.int64), goal=torch.zeros(2, 730))
        self.assertEqual(len(rollout), 0)
        arguments = (make_state(2), torch.zeros(2, 29), torch.zeros(2), torch.zeros(2), torch.ones(2, dtype=torch.int64))
        rollout.append(*arguments)
        with self.assertRaises(RuntimeError):
            rollout.append(*arguments)
        rollout.clear()
        rollout.clear()
        self.assertEqual(len(rollout), 0)
        for call in (rollout.as_dict, rollout.detached, lambda: rollout.append(*arguments)):
            with self.assertRaises(RuntimeError):
                call()

    def test_cached_goals_required_each_step_and_initial_shape_checked(self):
        """目标缓存方式前后一致, 历史长度不符合配置时立即报错."""

        rollout = make_rollout(horizon=1, goal=True)
        with self.assertRaises(ValueError):
            rollout.append(make_state(2), torch.zeros(2, 29), torch.zeros(2), torch.zeros(2), torch.ones(2, dtype=torch.int64))
        self.assertEqual(len(rollout), 0)
        with self.assertRaisesRegex(ValueError, "initial_observation_history"):
            ImaginedRollout(make_state(2), torch.zeros(2, 29), torch.zeros(2, 3, 99), torch.zeros(2, 33), torch.zeros(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int64), horizon=3)

    def test_clear_releases_internal_references_and_preserves_external_output(self):
        """clear 释放容器引用, 已交给上层的批次仍由上层持有和释放."""

        initial_state = make_state(2)
        initial_reference = weakref.ref(initial_state["joint_pos"])
        rollout = ImaginedRollout(initial_state, torch.zeros(2, 29), torch.zeros(2, 10, 99), torch.zeros(2, 33), torch.zeros(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int64), horizon=1)
        del initial_state
        rollout.append(make_state(2), torch.zeros(2, 29), torch.zeros(2), torch.zeros(2), torch.ones(2, dtype=torch.int64))
        batch = rollout.as_dict()
        batch_reference = weakref.ref(batch["actions"])
        rollout.clear()
        gc.collect()
        self.assertIsNone(initial_reference())
        self.assertIsNotNone(batch_reference())
        del batch
        gc.collect()
        self.assertIsNone(batch_reference())


if __name__ == "__main__":
    unittest.main()

"""验证限时评估的回合选择、成功判断及回报归一化, 不启动仿真窗口."""

from types import SimpleNamespace
import unittest

import torch

from supermimic.workflows.eval import evaluate_policy


class FakePolicy(torch.nn.Module):
    """提供固定动作, 仅用于评估流程检查."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(goal_frames=1, history_length=2)

    def predict_actions(self, history, goal):
        return torch.zeros(1, 1, 29)


class FakeMotion:
    """三段动作分别用于检查限时完成、短片段完成和跟踪失败."""

    device = torch.device("cpu")
    names = ["long", "short", "fail"]
    num_clips = 3
    clip_lengths = torch.tensor([2001, 2, 2001])

    def get_reference(self, clip, frame, include_body=False):
        return {"joint_pos": torch.zeros(1, 29)}


class FakeEnv:
    """记录重置和步数, 返回最小机器人状态."""

    num_envs, device = 1, torch.device("cpu")
    config = SimpleNamespace(control_dt=.02, show_viewer=False)

    def __init__(self):
        self.resets = self.steps = 0

    def reset(self, ids, state):
        self.resets += 1
        return state

    def step(self, action):
        self.steps += 1
        return {"joint_pos": torch.zeros(1, 29)}


class FakeMDP:
    """返回最大单步奖励, 第三个动作在指定帧触发失败."""

    motion = FakeMotion()
    config = SimpleNamespace(control_dt=.02, goal_frames=1)
    max_step_reward = 1.

    def __init__(self, fail_frame=2):
        self.fail_frame = fail_frame

    def observation(self, *args):
        return torch.zeros(1, 99)

    def goal_features(self, *args):
        return torch.zeros(1, 73)

    def reward(self, *args):
        return torch.ones(1)

    def termination(self, state, clip, frame):
        return (clip == 2) & (frame >= self.fail_frame)


class EvaluationTest(unittest.TestCase):
    """检查周期随机模式和最终顺序模式的边界行为."""

    def test_final_order_limit_failure_and_denominator(self):
        policy, env = FakePolicy(), FakeEnv()
        policy.train()
        result = evaluate_policy(env, policy, FakeMDP(), max_steps=3, verbose=False)
        records = result["per_clip"]
        self.assertEqual([r["clip_id"] for r in records], [0, 1, 2])
        self.assertEqual([r["steps"] for r in records], [3, 1, 2])
        self.assertEqual([r["end_reason"] for r in records], ["time_limit", "clip_end", "failed"])
        self.assertEqual(result["success_count"], 2)
        self.assertAlmostEqual(result["success_rate"], 2 / 3)
        self.assertAlmostEqual(records[2]["normalized_return"], 2 / 3)
        self.assertEqual(env.resets, 3)
        self.assertTrue(policy.training)

    def test_failure_on_limit_step_is_not_success(self):
        result = evaluate_policy(FakeEnv(), FakePolicy(), FakeMDP(fail_frame=3), max_steps=3, verbose=False)
        last = result["per_clip"][-1]
        self.assertEqual(last["steps"], 3)
        self.assertFalse(last["success"])
        self.assertEqual(last["end_reason"], "failed")

    def test_training_random_twenty_reproducible_and_rng_isolated(self):
        state = torch.get_rng_state().clone()
        result = evaluate_policy(FakeEnv(), FakePolicy(), FakeMDP(), mode="training", num_episodes=20, max_steps=3, seed=7, verbose=False)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        again = evaluate_policy(FakeEnv(), FakePolicy(), FakeMDP(), mode="training", num_episodes=20, max_steps=3, seed=7, verbose=False)
        selected = [r["clip_id"] for r in result["per_clip"]]
        self.assertEqual(selected, [r["clip_id"] for r in again["per_clip"]])
        self.assertEqual(len(selected), 20)
        self.assertLess(len(set(selected)), 20)
        self.assertTrue(all(0 <= i < 3 for i in selected))

    def test_default_thirty_second_limit(self):
        result = evaluate_policy(FakeEnv(), FakePolicy(), FakeMDP(), verbose=False)
        self.assertEqual(result["per_clip"][0]["steps"], 1500)
        self.assertTrue(result["per_clip"][0]["success"])


if __name__ == "__main__":
    unittest.main()

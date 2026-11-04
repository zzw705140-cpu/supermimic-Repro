"""验证参考动作库的插值, 速度和动作边界, 不启动物理仿真.

在项目根目录运行: python -m unittest discover -s tests -p test_motion.py -v
"""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_platform.data.motion import MotionLib


class MotionLibTest(unittest.TestCase):
    """用恒定线速度, 世界角速度及关节速度的合成轨迹检查数值结果."""

    def setUp(self):
        """创建两个独立动作, 大的位置偏移用于暴露跨动作求导错误."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.linear_velocity = np.array([1.0, 2.0, -0.5])
        self.joint_velocity = np.arange(1, 30) * 0.01
        self._write_clip("a.csv", frames=31, offset=0.0)
        self._write_clip("b.csv", frames=16, offset=100.0)
        self.motion = MotionLib(self.path)

    def _write_clip(self, name, frames, offset):
        """保存带四元数符号翻转及非单位模长的 30 FPS 合成 CSV."""
        times = np.arange(frames) / 30.0
        position = offset + times[:, None] * self.linear_velocity
        # 初始姿态非单位旋转, 确保角速度确实在世界系而不是局部系.
        rotations = Rotation.from_rotvec(times[:, None] * np.array([0.0, 0.0, 0.6])) * Rotation.from_rotvec([0.4, 0.0, 0.0])
        quaternion = rotations.as_quat() * 2.0
        quaternion[1::2] *= -1
        joints = times[:, None] * self.joint_velocity
        np.savetxt(self.path / name, np.concatenate((position, quaternion, joints), axis=1), delimiter=",")

    def test_interpolation_and_time_scale(self):
        """检查帧数, 位置, 关节角和球面插值保持真实运动时间尺度."""
        self.assertEqual(self.motion.names, ["a", "b"])
        self.assertEqual(self.motion.clip_lengths.tolist(), [51, 26])
        state = self.motion.get_reference(0, 17)
        time = 17 * 0.02
        np.testing.assert_allclose(state["root_pos"].numpy(), time * self.linear_velocity, atol=1e-6)
        np.testing.assert_allclose(state["joint_pos"].numpy(), time * self.joint_velocity, atol=1e-6)
        expected = Rotation.from_rotvec([0, 0, time * 0.6]) * Rotation.from_rotvec([0.4, 0, 0])
        actual = Rotation.from_quat(state["root_quat"].numpy())
        np.testing.assert_allclose(actual.as_matrix(), expected.as_matrix(), atol=1e-6)
        torch.testing.assert_close(self.motion.states["root_quat"].norm(dim=-1), torch.ones(77))

    def test_velocities_and_clip_boundaries(self):
        """验证首尾单边差分与内部中心差分, 两个动作之间不产生虚假速度."""
        reference = self.motion.get_reference([0, 0, 0, 1, 1], [0, 25, 50, 0, 25])
        np.testing.assert_allclose(reference["root_lin_vel"].numpy(), np.tile(self.linear_velocity, (5, 1)), atol=1e-6)
        np.testing.assert_allclose(reference["root_ang_vel"].numpy(), np.tile([0, 0, 0.6], (5, 1)), atol=1e-6)
        np.testing.assert_allclose(reference["joint_vel"].numpy(), np.tile(self.joint_velocity, (5, 1)), atol=1e-6)

    def test_future_stops_at_clip_end(self):
        """真实末帧保留速度, 超出末尾保持姿态并置零速度, 不进入下一个动作."""
        future = self.motion.get_future_reference(0, 49, 3)
        self.assertEqual(future["root_pos"].shape, (3, 3))
        np.testing.assert_allclose(future["root_pos"].numpy(), np.tile(self.linear_velocity, (3, 1)), atol=1e-6)
        np.testing.assert_allclose(future["root_lin_vel"][0].numpy(), self.linear_velocity, atol=1e-6)
        for key in ("root_lin_vel", "root_ang_vel", "joint_vel"):
            self.assertEqual(torch.count_nonzero(future[key][1:]).item(), 0)

    def test_batch_queries_and_copies(self):
        """检查批量输入, 标量广播和返回副本不污染原始数据."""
        reference = self.motion.get_reference([0, 1], 0)
        self.assertEqual(reference["joint_pos"].shape, (2, 29))
        reference["root_pos"][0] = -999
        torch.testing.assert_close(self.motion.get_reference(0, 0)["root_pos"], torch.zeros(3))
        self.assertEqual(self.motion.get_future_reference(0, [0, 10], 4)["root_quat"].shape, (2, 4, 4))
        future = self.motion.get_future_reference([0, 1], [49, 24], 3)
        self.assertEqual(future["joint_pos"].shape, (2, 3, 29))
        self.assertEqual(torch.count_nonzero(future["joint_vel"][:, 1:]).item(), 0)

    def test_non_aligned_end_time(self):
        """控制周期不能整除时长时保持等间隔, 不追加非等间隔的原始末帧."""
        motion = MotionLib(self.path, control_dt=0.07)
        self.assertEqual(motion.clip_lengths.tolist(), [15, 8])
        np.testing.assert_allclose(motion.get_reference(0, 14)["root_pos"].numpy(), 0.98 * self.linear_velocity, atol=1e-6)

    def test_invalid_queries(self):
        """拒绝越界帧号, 浮点索引和无效未来长度, 避免静默取错动作."""
        for clip, frame in ((-1, 0), (2, 0), (0, -1), (1, 26)):
            with self.assertRaises(IndexError):
                self.motion.get_reference(clip, frame)
        for clip, frame in ((0.0, 0), (0, 1.5), (True, 0), ([[0]], 0)):
            with self.assertRaises(ValueError):
                self.motion.get_reference(clip, frame)
        for count in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                self.motion.get_future_reference(0, 0, count)

    def test_invalid_data_and_timing(self):
        """拒绝错误频率, 列数, 非有限数值和零四元数."""
        for kwargs in ({"source_fps": 0}, {"control_dt": float("nan")}, {"control_dt": 2.0}):
            with self.assertRaises(ValueError):
                MotionLib(self.path, **kwargs)
        for case in ("columns", "nan", "quaternion"):
            raw = np.zeros((3, 36))
            raw[:, 6] = 1
            if case == "columns":
                raw = raw[:, :-1]
            elif case == "nan":
                raw[1, 0] = float("nan")
            else:
                raw[1, 3:7] = 0
            np.savetxt(self.path / "bad.csv", raw, delimiter=",")
            with self.assertRaises(ValueError):
                MotionLib(self.path)
            (self.path / "bad.csv").unlink()


if __name__ == "__main__":
    unittest.main()

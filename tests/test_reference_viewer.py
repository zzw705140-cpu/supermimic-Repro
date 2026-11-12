"""无窗口验证蓝色参考模型、P 键开关及显示状态同步, 不推进物理."""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from sim_platform.assets.robots.g1.robot import default_joint_pos_list
from sim_platform.sim.env import RobotSimEnv, RobotSimEnvConfig


class ReferenceViewerTest(unittest.TestCase):
    """使用真实 MuJoCo 显示模型, 用窗口替身避免测试打开桌面窗口."""

    def test_reference_toggle_and_pose(self):
        """参考状态只驱动显示副本, 同一回调可显示、隐藏并更新下一帧."""
        env = RobotSimEnv.__new__(RobotSimEnv)
        env.config = RobotSimEnvConfig(show_viewer=True)
        env.device, env.num_envs, env.is_closed = torch.device("cpu"), 1, False
        env._viewer = env._viewer_model = env._viewer_data = None
        env._viewer_closed, env._reference_visible, env._reference_values = False, False, None
        env.scene = SimpleNamespace(env_origins=torch.tensor([[3., 4., 0.]]))
        pose = torch.tensor([[3., 4., .76, 0., 0., 0., 1.]])
        joints = torch.tensor([default_joint_pos_list()])
        env.robot = SimpleNamespace(data=SimpleNamespace(root_link_pose_w=SimpleNamespace(torch=pose), joint_pos=SimpleNamespace(torch=joints)))
        reference = {"root_pos": torch.tensor([[.2, .3, .8]]), "root_quat": torch.tensor([[0., 0., 0., 1.]]), "root_lin_vel": torch.zeros(1, 3), "root_ang_vel": torch.zeros(1, 3), "joint_pos": joints.clone(), "joint_vel": torch.zeros(1, 29)}
        saved = {key: value.clone() for key, value in reference.items()}
        viewer = SimpleNamespace(opt=mujoco.MjvOption(), cam=mujoco.MjvCamera(), lock=nullcontext, close=lambda: None)
        env.set_reference_state(reference)
        with patch("mujoco.viewer.launch_passive", return_value=viewer) as launch:
            env._initialize_viewer()
        callback = launch.call_args.kwargs["key_callback"]
        self.assertFalse(viewer.opt.geomgroup[2])
        original_qpos = env._viewer_data.qpos[:36].copy()
        callback(ord("P"))
        env._sync_viewer_state()
        self.assertTrue(viewer.opt.geomgroup[2])
        address = env._reference_root_address
        np.testing.assert_allclose(env._viewer_data.qpos[address:address + 3], [.2, 1.3, .8], atol=1e-6)
        np.testing.assert_allclose(env._viewer_data.qpos[address + 3:address + 7], [1, 0, 0, 0])
        ghost = env._viewer_model.geom_group == 2
        self.assertTrue(ghost.any())
        self.assertTrue(np.all(env._viewer_model.geom_contype[ghost] == 0))
        self.assertTrue(np.all(env._viewer_model.geom_conaffinity[ghost] == 0))
        np.testing.assert_allclose(env._viewer_model.geom_rgba[ghost][0], [.25, .65, 1, .35])
        reference["joint_pos"] = reference["joint_pos"] + .05
        env.set_reference_state(reference)
        env._sync_viewer_state()
        np.testing.assert_allclose(env._viewer_data.qpos[env._reference_joint_addresses], reference["joint_pos"][0], atol=1e-6)
        np.testing.assert_array_equal(env._viewer_data.qpos[:36], original_qpos)
        callback(ord("P"))
        env._sync_viewer_state()
        self.assertFalse(viewer.opt.geomgroup[2])
        for key in saved:
            if key != "joint_pos":
                torch.testing.assert_close(reference[key], saved[key])
        env._close_viewer()
        self.assertIsNone(env._reference_values)


if __name__ == "__main__":
    unittest.main()

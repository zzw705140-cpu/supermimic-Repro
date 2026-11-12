"""无窗口验证蓝色参考模型、P 键开关及显示状态同步, 不推进物理."""

from contextlib import nullcontext
from collections import deque
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
        env._viewer_pan_requests = deque()
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

    def test_arrow_keys_pan_camera_in_ground_plane(self):
        """旋转视角后方向键仍沿视线前后和左右平移, 不改变相机高度、距离和角度."""
        env = RobotSimEnv.__new__(RobotSimEnv)
        env.config = RobotSimEnvConfig(show_viewer=True, camera_pan_step=.2)
        env.is_closed, env._viewer_closed, env._reference_values = False, False, None
        env._viewer_pan_requests = deque()
        env._sync_viewer_state = lambda: None
        env._viewer_key_callback = (None, None)  # 本测试不创建 GLFW 窗口, 重复事件另行验证.
        camera = mujoco.MjvCamera()
        camera.distance, camera.elevation = 3., -15.
        env._viewer = SimpleNamespace(cam=camera, lock=nullcontext, is_running=lambda: True, sync=lambda: None)
        for azimuth, expected_forward, expected_right in ((0, [.2, 0.], [0., -.2]), (90, [0., .2], [.2, 0.])):
            camera.azimuth = azimuth
            camera.lookat[:] = [0., 0., 2.]
            env._on_viewer_key(265)
            np.testing.assert_array_equal(camera.lookat, [0., 0., 2.])  # 键盘回调不直接修改窗口.
            env.render()
            np.testing.assert_allclose(camera.lookat[:2], expected_forward, atol=1e-7)
            env._on_viewer_key(264)
            env.render()
            np.testing.assert_allclose(camera.lookat[:2], [0., 0.], atol=1e-7)
            env._on_viewer_key(262)
            env.render()
            np.testing.assert_allclose(camera.lookat[:2], expected_right, atol=1e-7)
            env._on_viewer_key(263)
            env.render()
            np.testing.assert_allclose(camera.lookat, [0., 0., 2.], atol=1e-7)
            self.assertEqual((camera.distance, camera.azimuth, camera.elevation), (3., azimuth, -15.))

    def test_arrow_hold_repeat_preserves_native_keys(self):
        """长按重复事件持续入队, 松开停止, P 和其他原生快捷键只按原逻辑处理."""
        import glfw

        env = RobotSimEnv.__new__(RobotSimEnv)
        env._viewer_pan_requests = deque()
        env._viewer_key_callback = None
        env._reference_values, env._reference_visible = object(), False
        events = []

        def original(window, key, scancode, action, mods):
            """模拟 MuJoCo 仅在 PRESS 时调用用户键盘入口的原生回调."""
            events.append((key, action))
            if action == glfw.PRESS:
                env._on_viewer_key(key)

        with patch("glfw.get_current_context", return_value=object()), patch("glfw._glfw.glfwSetKeyCallback", return_value=original) as setter:
            env._on_viewer_key(glfw.KEY_UP)
            callback = env._viewer_key_callback[0]
            for _ in range(3):
                callback(None, glfw.KEY_UP, 0, glfw.REPEAT, 0)
            self.assertEqual(list(env._viewer_pan_requests), [(0, 1)] * 4)
            callback(None, glfw.KEY_UP, 0, glfw.RELEASE, 0)
            self.assertEqual(len(env._viewer_pan_requests), 4)
            callback(None, glfw.KEY_P, 0, glfw.PRESS, 0)
            self.assertTrue(env._reference_visible)
            callback(None, glfw.KEY_P, 0, glfw.REPEAT, 0)
            self.assertTrue(env._reference_visible)
            callback(None, glfw.KEY_RIGHT, 0, glfw.PRESS, 0)
            self.assertEqual(env._viewer_pan_requests[-1], (1, 0))
            self.assertEqual(setter.call_count, 1)
            self.assertEqual(len(events), 7)


if __name__ == "__main__":
    unittest.main()

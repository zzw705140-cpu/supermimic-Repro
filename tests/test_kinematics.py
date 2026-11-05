"""对照 MuJoCo 验证可微正运动学, 无窗口且不推进物理.

运行: python -m unittest discover -s tests -p test_kinematics.py -v
"""

import ast
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from sim_platform.data.kinematics import G1Kinematics


class KinematicsTest(unittest.TestCase):
    """验证真实 G1 的所有连杆, 固定连接, 非零关节锚点以及 autograd."""

    @classmethod
    def setUpClass(cls):
        """从资源代码读取关节顺序, 不导入 Isaac Lab 或启动仿真场景."""
        assets = PROJECT / "sim_platform/assets/robots/g1"
        tree = ast.parse((assets / "robot.py").read_text())
        cls.names = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "G1_JOINT_NAMES" for target in node.targets))
        cls.path = assets / "g1_29dof.xml"
        cls.model = mujoco.MjModel.from_xml_path(str(cls.path))
        cls.bodies = [cls.model.body(i).name for i in range(1, cls.model.nbody)]
        cls.kin = G1Kinematics(cls.path, cls.names, cls.bodies, dtype=torch.float64)

    def _state(self, names, model, seed=0):
        """生成包含世界系根部速度的非平凡随机状态, 返回六个 float64 输入张量."""
        rng = np.random.default_rng(seed)
        q = np.array([rng.uniform(*model.jnt_range[model.joint(name).id]) for name in names])
        values = (rng.normal(size=3), Rotation.from_rotvec(rng.normal(size=3)).as_quat(), rng.normal(size=3), rng.normal(size=3), q, rng.normal(size=len(names)))
        return tuple(torch.tensor(value, dtype=torch.float64) for value in values)

    def _compare_mujoco(self, kin, model, state):
        """将同一状态写入 MuJoCo, 用连杆位姿及原点雅可比速度对照全部输出."""
        p, quat, v, w, q, qd = [value.detach().numpy() for value in state]
        data = mujoco.MjData(model)
        free = int(np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
        qadr, vadr = model.jnt_qposadr[free], model.jnt_dofadr[free]
        data.qpos[qadr:qadr + 3] = p
        data.qpos[qadr + 3:qadr + 7] = quat[[3, 0, 1, 2]]
        data.qvel[vadr:vadr + 3] = v
        # MuJoCo 自由关节的角速度使用局部系, 接口输入则使用世界系.
        data.qvel[vadr + 3:vadr + 6] = Rotation.from_quat(quat).inv().apply(w)
        for i, name in enumerate(kin.joint_names):
            joint = model.joint(name).id
            data.qpos[model.jnt_qposadr[joint]] = q[i]
            data.qvel[model.jnt_dofadr[joint]] = qd[i]
        mujoco.mj_forward(model, data)

        expected = {key: [] for key in ("body_pos", "body_rot", "body_lin_vel", "body_ang_vel")}
        for name in kin.tracked_bodies:
            body = model.body(name).id
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jp, jr, body)
            expected["body_pos"].append(data.xpos[body].copy())
            expected["body_rot"].append(data.xmat[body].reshape(3, 3).copy())
            expected["body_lin_vel"].append(jp @ data.qvel)
            expected["body_ang_vel"].append(jr @ data.qvel)
        actual = kin.forward(*state)
        for key in expected:
            np.testing.assert_allclose(actual[key].detach().numpy(), np.array(expected[key]), atol=1e-10, rtol=1e-10, err_msg=key)

    def test_g1_against_mujoco(self):
        """随机姿态和速度下检查全部 39 个连杆, 包含手掌, 头部及传感器固定连接."""
        for seed in range(4):
            self._compare_mujoco(self.kin, self.model, self._state(self.names, self.model, seed))

    def test_joint_and_body_order(self):
        """输入关节和输出部位都可重排, 不依赖模型内部数组顺序."""
        kin = G1Kinematics(self.path, self.names[::-1], ["right_rubber_hand", "pelvis", "left_ankle_roll_link"], dtype=torch.float64)
        self._compare_mujoco(kin, self.model, self._state(kin.joint_names, self.model))
        self.assertEqual(kin.q_lo.shape, (29,))

    def test_batch_and_float32(self):
        """检查多维批量输出和默认 float32 精度, 各样本结果与单帧一致."""
        kin = G1Kinematics(self.path, self.names, self.bodies)
        state = tuple(value.float() for value in self._state(self.names, self.model))
        batch = tuple(value.expand(2, 3, *value.shape) for value in state)
        output, single = kin.forward(*batch), kin.forward(*state)
        self.assertEqual(output["body_rot"].shape, (2, 3, 39, 3, 3))
        for key in single:
            torch.testing.assert_close(output[key], single[key].expand_as(output[key]))
            torch.testing.assert_close(single[key].double(), self.kin.forward(*(value.double() for value in state))[key], atol=3e-6, rtol=3e-6)

    def test_gradients(self):
        """用有限差分验证六类输入的解析梯度, 包含零关节角避免轴角奇点."""
        state = self._state(self.names, self.model)
        for zero_joints in (False, True):
            inputs = tuple((torch.zeros_like(value) if zero_joints and i == 4 else value.clone()).requires_grad_() for i, value in enumerate(state))
            self.assertTrue(torch.autograd.gradcheck(lambda *values: tuple(self.kin.forward(*values).values()), inputs, fast_mode=True, atol=1e-5, rtol=1e-3))

    def test_joint_anchor_and_reference_angle(self):
        """合成模型覆盖非零关节锚点, 参考角和有旋转偏移的固定父连接."""
        xml = '''<mujoco><compiler angle="radian" fusestatic="false"/>
        <worldbody><body name="root"><freejoint/><geom size="0.1"/>
          <body name="fixed" pos="0.2 -0.1 0.3" euler="0.2 0.3 -0.1"><geom size="0.1"/>
            <body name="moving" pos="0.1 0.2 0.4" euler="-0.1 0.4 0.2">
              <joint name="hinge" axis="1 2 3" pos="0.1 -0.2 0.3" ref="0.3" range="-0.7 0.9"/>
              <geom size="0.1"/>
              <body name="tip" pos="0.2 0.1 -0.2"><geom size="0.05"/></body>
            </body>
          </body>
        </body></worldbody></mujoco>'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anchored.xml"
            path.write_text(xml)
            model = mujoco.MjModel.from_xml_path(str(path))
            kin = G1Kinematics(path, ["hinge"], ["root", "fixed", "moving", "tip"], dtype=torch.float64)
            for seed in range(3):
                self._compare_mujoco(kin, model, self._state(["hinge"], model, seed))

    def test_input_checks(self):
        """明确拒绝错误部位名, 关节名, 输入形状, dtype 和零四元数."""
        for names, bodies in ((self.names[:-1], self.bodies), (self.names, ["missing"])):
            with self.assertRaises(ValueError):
                G1Kinematics(self.path, names, bodies)
        state = list(self._state(self.names, self.model))
        for index, replacement in ((1, torch.zeros(4, dtype=torch.float64)), (4, torch.zeros(28, dtype=torch.float64)), (0, torch.zeros(3))):
            invalid = state.copy()
            invalid[index] = replacement
            with self.assertRaises(ValueError):
                self.kin.forward(*invalid)


if __name__ == "__main__":
    unittest.main()

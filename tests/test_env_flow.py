"""独立验证 G1 环境完整流程, 使用 MuJoCo 窗口显示 Isaac Lab 的实时状态.

窗口预览: python tests/test_env_flow.py
GPU 物理: python tests/test_env_flow.py --device cuda:0
无窗口检查: python tests/test_env_flow.py --viewer none --num-envs 2 --steps 160
物理由 Isaac Lab + Newton/MuJoCo-Warp 推进, MuJoCo 窗口只显示第 0 个环境.
默认使用 CPU 和非 CUDA 图模式, 以分别定位 GPU 与窗口相关问题.
预览使用零动作保持默认关节角, 不包含平衡策略, 跌倒后每 150 步重置.
"""

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from isaaclab.app import launch_simulation
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass, instantiate, replace
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_platform.assets.ground.ground import GROUND_VISUAL_PATH, dynamic_friction, static_friction
from sim_platform.assets.robots.g1.robot import G1_CFG, G1_JOINT_NAMES, MJCF_PATH, USD_PATH, action_scale


"""---------------- 任务入口与场景 ----------------"""


def report(message):
    """立即打印带时间的阶段信息, 便于定位初始化或步进停滞的位置."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


@dataclass
class TaskCfg:
    """测试参数, N=num_envs 为环境数, control_dt 为秒, substeps 为物理子步数."""

    num_envs: int = 1
    control_dt: float = 0.02
    substeps: int = 4
    reset_steps: int = 150


@configclass
class SceneCfg(InteractiveSceneCfg):
    """加载 G1 和本地生成的静态地面, 地面顶面 z=0, 不依赖远程资源."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.CuboidCfg(size=(100.0, 100.0, 0.1), collision_props=sim_utils.UsdPhysicsCollisionCfg(), physics_material=sim_utils.RigidBodyMaterialBaseCfg(static_friction=static_friction, dynamic_friction=dynamic_friction)), init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)))
    robot = replace(G1_CFG, prim_path="{ENV_REGEX_NS}/Robot")


"""---------------- 物理环境 ----------------"""


class FlowEnv:
    """最小物理环境, 只组装场景, 重置, 步进并读取状态.

    输入: actions [N, 29], 按 G1_JOINT_NAMES 排列的无量纲动作.
    输出: root_pose [N, 7] 为世界位置和 xyzw 四元数, root_velocity [N, 6]
    为世界线速度和角速度, q/qd [N, 29] 为关节角和角速度.
    """

    def __init__(self, cfg, sim_cfg):
        """在不创建 Newton 窗口的情况下加载场景, 初始化求解器与动作缩放."""
        self.cfg = cfg
        report("1/6 创建 SimulationContext")
        self.sim = SimulationContext(sim_cfg)
        report("2/6 加载 USD 机器人与地面")
        self.scene = instantiate(SceneCfg(num_envs=cfg.num_envs, env_spacing=2.5, replicate_physics=True))
        report("3/6 初始化物理求解器, 首次运行可能编译内核")
        self.sim.reset()
        self.robot = self.scene["robot"]
        if list(self.robot.joint_names) != G1_JOINT_NAMES:
            raise AssertionError(f"关节顺序不匹配: {self.robot.joint_names}")
        self.default_q = self.robot.data.default_joint_pos.torch.clone()
        self.alpha = torch.tensor([action_scale(name) for name in G1_JOINT_NAMES], device=sim_cfg.device)
        report("物理求解器初始化完成")

    def state(self):
        """复制当前状态并返回四个字段, 形状和坐标约定见类说明."""
        data = self.robot.data
        return {"root_pose": data.root_link_pose_w.torch.clone(), "root_velocity": data.root_link_vel_w.torch.clone(), "q": data.joint_pos.torch.clone(), "qd": data.joint_vel.torch.clone()}

    def reset(self):
        """恢复全部环境的默认位姿, 速度和 PD 目标, 返回重置后的状态."""
        pose = self.robot.data.default_root_pose.torch.clone()
        pose[:, :3] += self.scene.env_origins
        self.robot.write_root_pose_to_sim_index(root_pose=pose)
        self.robot.write_root_velocity_to_sim_index(root_velocity=self.robot.data.default_root_vel.torch.clone())
        self.robot.write_joint_position_to_sim_index(position=self.default_q)
        self.robot.write_joint_velocity_to_sim_index(velocity=self.robot.data.default_joint_vel.torch.clone())
        self.robot.actuators.target_command.set_position_index(value=self.default_q)
        self.scene.reset()
        self.sim.forward()
        self.scene.update(0.0)
        return self.state()

    def step(self, actions):
        """将 actions [N, 29] 映射成 PD 目标, 推进 0.02 秒并返回状态."""
        if actions.shape != self.default_q.shape or not torch.isfinite(actions).all():
            raise ValueError(f"动作必须为有限数值且形状为 {tuple(self.default_q.shape)}")
        self.robot.actuators.target_command.set_position_index(value=self.default_q + self.alpha * actions)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(self.cfg.control_dt)
        return self.state()


"""---------------- 状态检查与窗口数据同步 ----------------"""


def check_state(state, num_envs):
    """检查状态字段形状, 有限性与单位四元数, 异常时终止测试."""
    for key, width in (("root_pose", 7), ("root_velocity", 6), ("q", 29), ("qd", 29)):
        if state[key].shape != (num_envs, width) or not torch.isfinite(state[key]).all():
            raise AssertionError(f"状态异常: {key}, shape={tuple(state[key].shape)}")
    torch.testing.assert_close(state["root_pose"][:, 3:].norm(dim=-1), torch.ones_like(state["q"][:, 0]), atol=1e-4, rtol=1e-4)


def verify_flow(env):
    """验证默认姿态, 零动作与肘关节动作的响应差异, 以及全部状态的重置往返."""
    report("4/6 验证 reset, step 和状态读取")
    initial = env.reset()
    check_state(initial, env.cfg.num_envs)
    torch.testing.assert_close(initial["q"], env.default_q)
    actions = torch.zeros_like(env.default_q)
    report("开始第一次物理步进")
    for _ in range(10):
        baseline = env.step(actions)
        check_state(baseline, env.cfg.num_envs)
    env.reset()
    elbow = G1_JOINT_NAMES.index("left_elbow_joint")
    actions[:, elbow] = 0.25
    for _ in range(10):
        moved = env.step(actions)
        check_state(moved, env.cfg.num_envs)
    response = (moved["q"][:, elbow] - baseline["q"][:, elbow]).abs()
    if not torch.all(response > 1e-4):
        raise AssertionError(f"肘关节未产生可辨别的动作响应: {response}")
    restored = env.reset()
    for key in initial:
        torch.testing.assert_close(restored[key], initial[key], atol=1e-5, rtol=1e-5)
    report(f"PASS: 初始状态, 动作响应和重置往返, 肘关节响应差异={response.tolist()} rad")


class StateDisplay:
    """把第 0 个环境的 Isaac Lab 状态映射到 MuJoCo, 仅前向计算并绘制.

    输入: FlowEnv.state() 的批量状态字典. 不运行 MuJoCo 物理步进.
    根据关节名称建立映射, 将根部四元数由 xyzw 转成 MuJoCo 的 wxyz.
    """

    def __init__(self):
        """在内存中添加显示用地面与灯光, 不修改磁盘 MJCF 或其他资源."""
        root = ET.parse(MJCF_PATH).getroot()
        root.find("compiler").set("meshdir", str(Path(MJCF_PATH).parent))
        # 预览材质从地面资源目录读取, 不依赖 RoboForge 的本地路径.
        for section in ET.parse(GROUND_VISUAL_PATH).getroot():
            target = root.find(section.tag)
            if target is None:
                target = ET.SubElement(root, section.tag)
            target.extend(section)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        free = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise AssertionError(f"显示模型预期一个浮动基座, 实际为 {len(free)}")
        self.root_address = int(self.model.jnt_qposadr[free[0]])
        ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in G1_JOINT_NAMES]
        if -1 in ids:
            raise AssertionError("显示模型缺少所需关节")
        self.joint_addresses = self.model.jnt_qposadr[ids]

    def sync(self, state):
        """同步第 0 个环境的根部位姿和 29 个关节角, 更新显示几何并检查数值."""
        pose = state["root_pose"][0].detach().cpu().numpy()
        address = self.root_address
        self.data.qpos[address:address + 3] = pose[:3]
        self.data.qpos[address + 3:address + 7] = pose[[6, 3, 4, 5]]
        q = state["q"][0].detach().cpu().numpy()
        self.data.qpos[self.joint_addresses] = q
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.xpos).all():
            raise AssertionError("显示模型的刚体坐标包含非有限数值")
        np.testing.assert_allclose(self.data.qpos[self.joint_addresses], q, atol=1e-7)


"""---------------- 测试与预览入口 ----------------"""


def main():
    """先完成物理验证, 再按参数开启窗口并持续步进, Ctrl+C 或关闭窗口退出."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu 或 cuda:0")
    parser.add_argument("--viewer", choices=("mujoco", "none"), default="mujoco")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0, help="预览步数, 0 表示持续运行")
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 0 or (args.steps == 0 and args.viewer == "none"):
        parser.error("环境数必须为正, 步数非负, 无窗口测试必须指定正步数")
    for path in (USD_PATH, MJCF_PATH):
        if not Path(path).is_file():
            raise FileNotFoundError(f"缺少机器人资源: {path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("当前进程无法使用 CUDA, 请使用 --device cpu")

    cfg = TaskCfg(num_envs=args.num_envs)
    physics = NewtonCfg(solver_cfg=MJWarpSolverCfg(njmax=500, nconmax=128, cone="pyramidal", integrator="implicitfast", use_mujoco_contacts=True), num_substeps=cfg.substeps, use_cuda_graph=False)
    sim_cfg = sim_utils.SimulationCfg(dt=cfg.control_dt, device=args.device, physics=physics, visualizer_cfgs=[])
    report(f"启动独立测试: device={args.device}, N={cfg.num_envs}, viewer={args.viewer}, CUDA graph=off")
    with launch_simulation(sim_cfg), ExitStack() as stack:
        env = FlowEnv(cfg, sim_cfg)
        verify_flow(env)
        report("5/6 创建显示模型, 校验实时状态映射")
        display = StateDisplay()
        display.sync(env.reset())
        report("PASS: MJCF 显示模型与 Isaac Lab 状态映射")
        viewer = None
        if args.viewer == "mujoco":
            import mujoco.viewer
            report("正在打开 MuJoCo 实时窗口")
            viewer = stack.enter_context(mujoco.viewer.launch_passive(display.model, display.data))
            with viewer.lock():
                viewer.cam.lookat[:] = [0.0, 0.0, 0.7]
                viewer.cam.distance = 3.0
                viewer.cam.azimuth = 135.0
                viewer.cam.elevation = -15.0
            viewer.sync()
            report("窗口已打开, 关闭窗口或 Ctrl+C 退出")
        report("6/6 持续物理步进, 每 150 步重置, 每 50 步报告状态")
        actions = torch.zeros_like(env.default_q)
        count = 0
        while (args.steps == 0 or count < args.steps) and (viewer is None or viewer.is_running()):
            start = time.perf_counter()
            if count % cfg.reset_steps == 0:
                env.reset()
            state = env.step(actions)
            check_state(state, cfg.num_envs)
            if viewer is None:
                display.sync(state)
            else:
                with viewer.lock():
                    display.sync(state)
                viewer.sync()
            count += 1
            if count % 50 == 0:
                report(f"step={count}, root_z={state['root_pose'][0, 2].item():.3f} m")
            if viewer is not None:
                time.sleep(max(0.0, cfg.control_dt - (time.perf_counter() - start)))
        report(f"PASS: 完成 {count} 步, 物理状态与显示状态同步检查通过")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        report("用户中断, 已退出测试")


# conda activate supermimic
# cd "/home/xinyu/project/model-based RL/supermimic-Repro"
# python tests/test_env_flow.py
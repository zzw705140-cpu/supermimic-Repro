"""验证 G1 的场景组装, 重置, 动作步进和状态读取.

窗口预览: python tests/test_basic_env.py --device cuda:0 --viz newton_gl --steps 0
无窗口检查: python tests/test_basic_env.py --device cpu --steps 100
零动作通过 PD 保持默认关节角, 不包含平衡策略, 机器人可能跌倒.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import torch

from isaaclab.app import add_launcher_args, launch_simulation
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass, instantiate, replace
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_platform.assets.ground.ground import dynamic_friction, static_friction
from sim_platform.assets.robots.g1.robot import G1_CFG, G1_JOINT_NAMES, USD_PATH, action_scale


"""---------------- 场景与任务配置 ----------------"""


@configclass
class SceneCfg(InteractiveSceneCfg):
    """组装 G1 与地面, num_envs 表示并行环境数, env_spacing 表示间距."""

    # 测试采用本地生成的静态薄盒地面, 顶面 z=0, 避免默认地面访问远程 USD.
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.CuboidCfg(size=(100.0, 100.0, 0.1), collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=sim_utils.RigidBodyMaterialBaseCfg(static_friction=static_friction, dynamic_friction=dynamic_friction), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45))), init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)))
    robot = replace(G1_CFG, prim_path="{ENV_REGEX_NS}/Robot")


@dataclass
class TaskCfg:
    """测试入口参数, control_dt 为控制周期秒数, substeps 为每周期物理子步数."""

    num_envs: int = 1
    control_dt: float = 0.02
    substeps: int = 4
    reset_steps: int = 150


"""---------------- 最小环境流程 ----------------"""


class BasicEnv:
    """测试专用环境, 由 TaskCfg 和 SimulationCfg 创建场景.

    输入: step 的 actions [N, 29], N 为环境数, 关节顺序为 G1_JOINT_NAMES.
    输出: 状态字典包含 root_pose [N, 7], root_velocity [N, 6], q/qd [N, 29].
    root_pose 使用世界坐标位置与 xyzw 四元数, root_velocity 为世界坐标线速度与角速度.
    """

    def __init__(self, cfg, sim_cfg):
        """创建并初始化机器人与地面, 校验关节顺序后重置到默认状态."""
        self.cfg = cfg
        self.sim = SimulationContext(sim_cfg)
        self.sim.set_camera_view([2.5, 2.5, 1.8], [0.0, 0.0, 0.7])
        self.scene = instantiate(SceneCfg(num_envs=cfg.num_envs, env_spacing=2.5, replicate_physics=True))
        self.sim.reset()
        self.robot = self.scene["robot"]
        if list(self.robot.joint_names) != G1_JOINT_NAMES:
            raise RuntimeError(f"关节顺序不匹配: {self.robot.joint_names}")
        self.default_q = self.robot.data.default_joint_pos.torch.clone()
        self.alpha = torch.tensor([action_scale(name) for name in G1_JOINT_NAMES], device=sim_cfg.device)
        self.reset()

    def state(self):
        """复制当前状态, 返回 root_pose, root_velocity, q, qd, 形状见类说明."""
        data = self.robot.data
        return {"root_pose": data.root_link_pose_w.torch.clone(), "root_velocity": data.root_link_vel_w.torch.clone(), "q": data.joint_pos.torch.clone(), "qd": data.joint_vel.torch.clone()}

    def reset(self):
        """恢复全部环境的默认位姿与速度, 清理缓存并返回初始状态."""
        pose = self.robot.data.default_root_pose.torch.clone()
        pose[:, :3] += self.scene.env_origins
        self.robot.write_root_pose_to_sim_index(root_pose=pose)
        self.robot.write_root_velocity_to_sim_index(root_velocity=self.robot.data.default_root_vel.torch.clone())
        self.robot.write_joint_position_to_sim_index(position=self.default_q)
        self.robot.write_joint_velocity_to_sim_index(velocity=self.robot.data.default_joint_vel.torch.clone())
        self.robot.set_joint_position_target_index(target=self.default_q)
        self.scene.reset()
        self.sim.forward()
        self.scene.update(0.0)
        return self.state()

    def step(self, actions):
        """将 actions [N, 29] 转成关节目标角, 推进一个控制周期并返回状态."""
        if actions.shape != self.default_q.shape or not torch.isfinite(actions).all():
            raise ValueError(f"动作必须为有限数值且形状为 {tuple(self.default_q.shape)}")
        self.robot.set_joint_position_target_index(target=self.default_q + self.alpha * actions)
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.cfg.control_dt)
        return self.state()


"""---------------- 检查与可视化入口 ----------------"""


def check_state(state, num_envs):
    """检查状态字段的批量维度与数值有限性, 不满足时抛出异常."""
    for key, width in (("root_pose", 7), ("root_velocity", 6), ("q", 29), ("qd", 29)):
        if state[key].shape != (num_envs, width) or not torch.isfinite(state[key]).all():
            raise AssertionError(f"状态异常: {key}, shape={state[key].shape}")


def main():
    """解析参数并验证重置与动作响应, 可持续显示实时仿真直到关闭窗口."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300, help="控制步数, 0 表示持续运行")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 0 or (args.steps == 0 and not args.visualizer):
        parser.error("环境数必须为正, 步数非负, 持续运行必须选择 --viz")
    if not Path(USD_PATH).is_file():
        raise FileNotFoundError(f"请先生成 G1 USD: {USD_PATH}")

    cfg = TaskCfg(num_envs=args.num_envs)
    physics = NewtonCfg(solver_cfg=MJWarpSolverCfg(njmax=500, nconmax=128, cone="pyramidal", integrator="implicitfast", use_mujoco_contacts=True), num_substeps=cfg.substeps, use_cuda_graph=args.device.startswith("cuda"))
    sim_cfg = sim_utils.SimulationCfg(dt=cfg.control_dt, device=args.device, physics=physics)
    with launch_simulation(sim_cfg, args):
        env = BasicEnv(cfg, sim_cfg)
        initial = env.reset()
        check_state(initial, cfg.num_envs)
        torch.testing.assert_close(initial["q"], env.default_q)
        actions = torch.zeros_like(env.default_q)
        elbow = G1_JOINT_NAMES.index("left_elbow_joint")
        actions[:, elbow] = 0.1
        moved = env.step(actions)
        check_state(moved, cfg.num_envs)
        if not (moved["q"] - initial["q"]).abs().max() > 1e-7:
            raise AssertionError("物理步进后关节状态没有变化")
        restored = env.reset()
        for key in initial:
            torch.testing.assert_close(restored[key], initial[key], atol=1e-5, rtol=1e-5)
        print("PASS: 场景加载, 状态形状, 默认姿态, 动作步进与重置往返检查通过.", flush=True)

        actions.zero_()
        count = 0
        while env.sim.is_running() and (args.steps == 0 or count < args.steps):
            start = time.perf_counter()
            if count % cfg.reset_steps == 0:
                env.reset()
            state = env.step(actions)
            check_state(state, cfg.num_envs)
            count += 1
            if args.visualizer:
                time.sleep(max(0.0, cfg.control_dt - (time.perf_counter() - start)))
        print(f"PASS: 完成 {count} 个控制步, 每 {cfg.reset_steps} 步恢复默认姿态.", flush=True)


if __name__ == "__main__":
    main()

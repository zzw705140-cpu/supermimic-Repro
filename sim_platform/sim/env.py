"""G1 批量仿真适配层, 连接上层控制指令与 Isaac Lab / Newton 物理后端."""

from __future__ import annotations

from collections.abc import Mapping
from collections import deque
from dataclasses import dataclass, replace as copy_config
import math
from pathlib import Path

import torch
import warp as wp
from newton import ModelFlags

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass, instantiate, replace
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from sim_platform.assets.ground.ground import GROUND_VISUAL_PATH, dynamic_friction, static_friction
from sim_platform.assets.robots.g1.robot import G1_CFG, G1_JOINT_NAMES, MJCF_PATH, USD_PATH, action_scale

##############################################################################
#                         环境配置与场景组装                                  #
##############################################################################

NUM_JOINTS = len(G1_JOINT_NAMES)
STATE_WIDTHS = {"root_pos": 3, "root_quat": 4, "root_lin_vel": 3, "root_ang_vel": 3, "joint_pos": NUM_JOINTS, "joint_vel": NUM_JOINTS}


@dataclass
class RobotSimEnvConfig:
    """公共仿真配置, 启动脚本可覆盖默认值; 不包含任务与训练参数.

    control_dt 为每次 step 推进的秒数, 每个物理子步为 control_dt / substeps.
    上层创建 MotionLib 和 TrackingMDP 时应传入同一个 control_dt.
    """

    num_envs: int = 1              # 并行环境数量, 编号为 0 到 num_envs-1
    device: str = "cpu"            # 物理和返回状态所在设备, 可设为 cuda:0
    control_dt: float = 0.02       # 控制周期, 默认 50 Hz
    substeps: int = 4              # 每个控制周期的物理子步数
    env_spacing: float = 2.5       # 并行环境原点间距, 单位为米
    show_viewer: bool = False      # 是否显示 MuJoCo 窗口, 训练与测评脚本均可选择
    viewer_env_id: int = 0         # 窗口显示哪个并行环境, 不影响其他环境的物理步进
    reference_offset_y: float = 1.0  # 蓝色参考机器人沿世界 y 轴的显示偏移, 单位为米
    camera_pan_step: float = 0.1   # 点按或长按重复事件的相机地面平移距离, 单位为米

    # 参考帧, 回合长度, 奖励和随机化采样规则由上层任务配置负责.

    def __post_init__(self):
        """在加载仿真前检查配置, 统一设备名称和时间参数."""

        for name in ("num_envs", "substeps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须为正整数")
        if not math.isfinite(self.control_dt) or self.control_dt <= 0:
            raise ValueError("control_dt 必须为正的有限数值")
        if not math.isfinite(self.env_spacing) or self.env_spacing < 0:
            raise ValueError("env_spacing 必须为非负的有限数值")
        if not math.isfinite(self.reference_offset_y):
            raise ValueError("reference_offset_y 必须为有限数值")
        if not math.isfinite(self.camera_pan_step) or self.camera_pan_step <= 0:
            raise ValueError("camera_pan_step 必须为正的有限数值")
        device = torch.device(self.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("仿真设备必须为 cpu 或 cuda")
        self.device = "cuda:0" if str(device) == "cuda" else str(device)
        if not isinstance(self.show_viewer, bool):
            raise ValueError("show_viewer 必须为 bool")
        if isinstance(self.viewer_env_id, bool) or not isinstance(self.viewer_env_id, int) or not 0 <= self.viewer_env_id < self.num_envs:
            raise ValueError("viewer_env_id 必须为有效的环境编号")


# 设置 Isaac Lab 的设备、控制周期、物理子步和求解器，它在创建环境前调用，不打开窗口
def make_sim_config(config: RobotSimEnvConfig | None = None) -> sim_utils.SimulationCfg:
    """只构造 Isaac Lab 物理配置, 不启用其显示后端; MuJoCo 窗口由环境 render 管理."""

    config = copy_config(config) if config is not None else RobotSimEnvConfig()
    physics = NewtonCfg(solver_cfg=MJWarpSolverCfg(njmax=500, nconmax=128, cone="pyramidal", integrator="implicitfast", use_mujoco_contacts=True), num_substeps=config.substeps, use_cuda_graph=config.device.startswith("cuda"))
    return sim_utils.SimulationCfg(dt=config.control_dt, device=config.device, physics=physics, visualizer_cfgs=[])


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """引用 G1 资源并创建本地静态地面, 顶面 z=0, 无需下载远程地面 USD."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.CuboidCfg(size=(1000.0, 1000.0, 0.1), collision_props=sim_utils.UsdPhysicsCollisionCfg(), physics_material=sim_utils.RigidBodyMaterialBaseCfg(static_friction=static_friction, dynamic_friction=dynamic_friction)), init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)))
    robot = replace(G1_CFG, prim_path="{ENV_REGEX_NS}/Robot")


##############################################################################
#                         基础仿真环境                                        #
##############################################################################


class RobotSimEnv:
    """只负责状态读写, 关节控制和物理步进, 不管理参考动作或经验池.

    输入:
        step(action)         : float32 [N,29], 按 G1_JOINT_NAMES 排列的无量纲动作.
        reset(env_ids, state): 编号 [B], 六字段状态各为 [B,D], 按编号顺序逐行对应.
        env_ids=None         : 选中全部环境, state=None 使用机器人资源中的默认状态.
        set_physical_parameters: 全部环境的摩擦 [N,1] 和躯干质心偏移 [N,3].
        set_joint_offsets      : 指定环境的关节目标偏移 [B,29].

    输出:
        state / step / reset 均返回全部 N 个环境的六字段字典, 不返回奖励或结束标志.
        root_pos       [N,3]: 减去各环境原点的位置, 坐标轴仍与世界系一致.
        root_quat      [N,4]: xyzw 四元数.
        root_lin_vel   [N,3]: 根部连杆原点的世界系线速度.
        root_ang_vel   [N,3]: 根部的世界系角速度.
        joint_pos/vel [N,29]: 关节角和关节角速度.

    生命周期与数据:
        在 launch_simulation 上下文中创建; 首次使用时由本类恢复默认状态.
        返回数据不承诺跨 step/reset 保持原值, 不应原地修改; 上层需要快照时自行复制.
        reset 只写入状态并刷新运动学, 不推进物理时间; step 推进一个 control_dt.
        show_viewer=True 时使用 MuJoCo 被动窗口, 仅显示 viewer_env_id 对应环境的位姿.
        窗口不执行 MuJoCo 物理步进; 手动关闭后不自动重开, render 返回窗口是否仍运行.

    用法:
        config = RobotSimEnvConfig(num_envs=2, device="cpu", show_viewer=False)
        sim_cfg = make_sim_config(config)
        with launch_simulation(sim_cfg):
            env = RobotSimEnv(config, sim_cfg=sim_cfg)
            state = env.step(torch.zeros(2, 29, device=config.device))
            # 需要重置时由上层选择参考帧, 再调用 env.reset(env_ids, reference_state).
            env.close()
    """

    # ---------------------------------- 初始化 ---------------------------------- #

    def __init__(self, config: RobotSimEnvConfig | None = None, *, sim_cfg: sim_utils.SimulationCfg | None = None):
        """检查配置, 加载场景与执行器, 准备默认状态和动作缩放."""

        self.config = copy_config(config) if config is not None else RobotSimEnvConfig()
        self.num_envs, self.device = self.config.num_envs, torch.device(self.config.device)
        sim_cfg = make_sim_config(self.config) if sim_cfg is None else sim_cfg
        if not math.isclose(sim_cfg.dt, self.config.control_dt, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("仿真 dt 必须与环境 control_dt 一致")
        if torch.device(sim_cfg.device) != self.device:
            raise ValueError("仿真与环境的 device 必须一致")
        if not isinstance(sim_cfg.physics, NewtonCfg) or not isinstance(sim_cfg.physics.solver_cfg, MJWarpSolverCfg) or sim_cfg.physics.num_substeps != self.config.substeps:
            raise ValueError("需要子步数一致的 Newton / MuJoCo-Warp 配置")
        if sim_cfg.visualizer_cfgs:
            raise ValueError("本环境使用 MuJoCo 窗口, 请通过 make_sim_config 禁用 Isaac Lab 显示后端")
        if not Path(USD_PATH).is_file():
            raise FileNotFoundError(f"请先生成机器人 USD: {USD_PATH}")
        if SimulationContext.instance() is not None:
            raise RuntimeError("当前进程已有仿真上下文, 请先关闭原环境")

        self.is_closed = False
        self._viewer = None
        self._viewer_model = self._viewer_data = None
        self._viewer_closed = False
        self._reference_values = None
        self._reference_visible = False
        self._viewer_pan_requests = deque()
        self._viewer_key_callback = None
        self.sim = SimulationContext(sim_cfg)
        try:
            self._initialize_simulation()
            self.reset()
        except Exception:
            self.close()
            raise

    def _initialize_simulation(self):
        """实例化场景, 初始化物理后端, 校验关节顺序并缓存固定控制参数."""

        scene_config = RobotSceneCfg(num_envs=self.num_envs, env_spacing=self.config.env_spacing, replicate_physics=True)
        # 地面覆盖并行环境网格, 额外预留机器人运动空间; 几何仍为本地静态薄盒.
        width = max(100.0, (math.ceil(math.sqrt(self.num_envs)) + 1) * self.config.env_spacing + 100.0)
        scene_config.ground.spawn.size = (width, width, 0.1)
        self.scene = instantiate(scene_config)
        self.sim.reset()
        self.robot = self.scene["robot"]
        if list(self.robot.joint_names) != G1_JOINT_NAMES:
            raise RuntimeError(f"关节顺序与 G1_JOINT_NAMES 不一致: {self.robot.joint_names}")
        self.default_joint_pos = self.robot.data.default_joint_pos.torch
        self.alpha = torch.tensor([action_scale(name) for name in G1_JOINT_NAMES], device=self.device, dtype=torch.float32)
        self._all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.joint_offsets = torch.zeros((self.num_envs, NUM_JOINTS), device=self.device)
        self._torso_id = self.robot.find_bodies("torso_link")[0][0]
        self._default_torso_com = self.robot.data.body_com_pos_b.torch[:, self._torso_id, :].clone()

    # ---------------------------------- 对外接口 ---------------------------------- #

    @torch.no_grad()
    def set_physical_parameters(self, friction: torch.Tensor, torso_com_offset: torch.Tensor):
        """设置所有环境的摩擦 [N,1] 和躯干质心偏移 [N,3], 不推进物理时间.

        偏移相对初始躯干质心, 摩擦写入机器人碰撞形状; 地面取采样下限 0.3.
        调用方负责采样及记录参数, 本接口不改变状态或隐式生成随机数.
        """
        self._check_open()
        self._check_tensor(friction, (self.num_envs, 1), "friction")
        self._check_tensor(torso_com_offset, (self.num_envs, 3), "torso_com_offset")
        if torch.any(friction < 0.3):
            raise ValueError("摩擦系数不能小于地面摩擦 0.3")
        manager = self.sim.backend_manager
        model = manager.get_model()
        shape_mu = wp.to_torch(model.shape_material_mu)
        shape_body = wp.to_torch(model.shape_body)
        robot_mu = wp.to_torch(self.robot._root_view.get_attribute("shape_material_mu", model)[:, 0])
        shape_mu[shape_body < 0] = 0.3
        robot_mu[:] = friction
        manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
        self.robot.set_coms_index(coms=(self._default_torso_com + torso_com_offset).unsqueeze(1), body_ids=[self._torso_id])

    @torch.no_grad()
    def set_joint_offsets(self, env_ids, offsets: torch.Tensor):
        """为指定环境设置关节目标偏移 [B,29], B 与编号行顺序一致."""
        self._check_open()
        ids = self._resolve_env_ids(env_ids)
        self._check_tensor(offsets, (ids.numel(), NUM_JOINTS), "offsets")
        self.joint_offsets[ids] = offsets

    def state(self) -> dict[str, torch.Tensor]:
        """读取当前六字段状态, 不推进时间, 不额外 clone 保存快照."""

        self._check_open()
        data = self.robot.data
        pose, velocity = data.root_link_pose_w.torch, data.root_link_vel_w.torch
        return {"root_pos": pose[:, :3] - self.scene.env_origins, "root_quat": pose[:, 3:7], "root_lin_vel": velocity[:, :3], "root_ang_vel": velocity[:, 3:6], "joint_pos": data.joint_pos.torch, "joint_vel": data.joint_vel.torch}

    @torch.no_grad()
    # 重置指定环境
    def reset(self, env_ids=None, state: Mapping[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """
        恢复所选机器人状态并返回整个批量; 不选择参考帧, 不重置其他环境.

        例如 env_ids=[2,5] 时, 输入状态第 0 行写到环境 2, 第 1 行写到环境 5.
        单个环境也接受无批量维的 [D] 状态, 原始输入不会被修改.
        """

        self._check_open()
        ids = self._resolve_env_ids(env_ids)
        if ids.numel() == 0:
            return self.state()
        initial = self._default_state(ids) if state is None else self._validate_state(state, ids.numel())
        self._write_state(ids, initial)
        self._refresh_state()
        if self.config.show_viewer:
            self.render()
        return self.state()

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """接收动作, 推进一个控制周期, 仅返回执行后的六字段状态."""

        self._check_open()
        self._check_tensor(action, (self.num_envs, NUM_JOINTS), "action")
        self._apply_action(action)
        self._advance_simulation()
        if self.config.show_viewer:
            self.render()
        return self.state()

    def render(self) -> bool:
        """
        若打开了仿真窗口：
        检查是否需要显示 → 准备窗口 → 同步状态 → 刷新画面
        
        无窗口时返回 False, 不推进物理时间.
        同步 MuJoCo 被动窗口并返回运行标志;
        """

        self._check_open()
        if not self.config.show_viewer or self._viewer_closed:
            return False
        if self._viewer is None:
            self._initialize_viewer()
        if not self._viewer.is_running():
            self._close_viewer()
            return False
        with self._viewer.lock():
            self._sync_viewer_state()
            # 回调仅入队; 在刷新线程持锁时移动相机, 避免键盘线程重复获取窗口锁.
            while self._viewer_pan_requests:
                sideways, forward = self._viewer_pan_requests.popleft()
                camera = self._viewer.cam
                angle, step = math.radians(camera.azimuth), self.config.camera_pan_step
                camera.lookat[0] += step * (forward * math.cos(angle) + sideways * math.sin(angle))
                camera.lookat[1] += step * (forward * math.sin(angle) - sideways * math.cos(angle))
                camera.type = 0  # mjCAMERA_FREE, 保留高度、距离、方位角和俯仰角.
        self._viewer.sync()
        return self._viewer.is_running()

    @torch.no_grad()
    def set_reference_state(self, state: Mapping[str, torch.Tensor]):
        """接收上层当前参考帧的六字段 [N,D], 仅缓存所选环境的显示位姿.

        不选取参考帧、不写入物理状态; 无窗口时跳过, P 键切换蓝色参考机器人.
        """
        self._check_open()
        if not self.config.show_viewer or self._viewer_closed:
            return
        reference = self._validate_state(state, self.num_envs)
        index = self.config.viewer_env_id
        self._reference_values = torch.cat((reference["root_pos"][index], reference["root_quat"][index], reference["joint_pos"][index])).detach().cpu().numpy()

    def close(self):
        """释放本环境持有的仿真和窗口资源, 重复调用无副作用."""

        if not self.is_closed:
            try:
                self._close_viewer()
            finally:
                try:
                    if SimulationContext.instance() is self.sim:
                        SimulationContext.clear_instance()
                finally:
                    self.is_closed = True

    # ---------------------------------- MuJoCo 状态显示 ---------------------------------- #


    """
    职责：创建显示窗口，读取选定机器人的位姿并刷新画面，不推进物理
    原因：显示功能则需要访问已经创建的机器人状态，并持有窗口对象，所以放在 RobotSimEnv 内部
    
    make_sim_config() 和 env.step(action)的关系：
    make_sim_config() → 创建物理仿真配置
                           ↓
                    env.step(action)
                           ↓
                    机器人状态发生变化
                           ↓
          render() → MuJoCo 窗口显示这个状态

          
    显示部分由公开的 render() 统一组织：
    render()
        ├── 尚未创建窗口 → _initialize_viewer()
        ├── 窗口已被关闭 → _close_viewer()
        └── 窗口正常运行 → _sync_viewer_state() → 刷新画面
    """
    def _initialize_viewer(self):
        """首次显示时加载独立 MJCF 和地面视觉资源, 不修改原始模型文件. 打开 MuJoCo 窗口并设置视角"""

        import xml.etree.ElementTree as ET
        from copy import deepcopy
        import mujoco
        import mujoco.viewer

        root = ET.parse(MJCF_PATH).getroot()
        root.find("compiler").set("meshdir", str(Path(MJCF_PATH).parent))
        for section in ET.parse(GROUND_VISUAL_PATH).getroot():
            target = root.find(section.tag)
            if target is None:
                target = ET.SubElement(root, section.tag)
            target.extend(section)
        # 副本仅属于显示模型, 不进入 Isaac Lab 的物理场景.
        world = root.find("worldbody")
        ghost = deepcopy(world.find("body"))
        for element in ghost.iter():
            if "name" in element.attrib:
                element.set("name", "reference_" + element.get("name"))
            if element.tag == "geom":
                element.set("contype", "0")
                element.set("conaffinity", "0")
                element.set("group", "2")
                element.set("rgba", "0.25 0.65 1.0 0.35")
                element.attrib.pop("material", None)
        world.append(ghost)
        self._viewer_model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self._viewer_data = mujoco.MjData(self._viewer_model)
        free_ids = [i for i in range(self._viewer_model.njnt) if self._viewer_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free_ids) != 2:
            raise ValueError("显示模型必须包含策略与参考机器人两个浮动根部")
        self._viewer_root_address = int(self._viewer_model.jnt_qposadr[free_ids[0]])
        self._reference_root_address = int(self._viewer_model.jnt_qposadr[free_ids[1]])
        # 按名称映射显示关节, 不假定 MJCF 与 Isaac Lab 内部的存储顺序相同.
        self._viewer_joint_addresses = [int(self._viewer_model.joint(name).qposadr[0]) for name in G1_JOINT_NAMES]
        self._reference_joint_addresses = [int(self._viewer_model.joint("reference_" + name).qposadr[0]) for name in G1_JOINT_NAMES]
        self._sync_viewer_state()
        self._viewer = mujoco.viewer.launch_passive(self._viewer_model, self._viewer_data, key_callback=self._on_viewer_key)
        with self._viewer.lock():
            self._viewer.opt.geomgroup[2] = 0
            self._viewer.cam.lookat[:] = [0.0, 0.0, 0.7]
            self._viewer.cam.distance = 3.0
            self._viewer.cam.azimuth = 135.0
            self._viewer.cam.elevation = -15.0

    def _on_viewer_key(self, keycode):
        """P 切换参考显示, 方向键支持点按和长按平移, 刷新由 render 负责."""
        if keycode in (ord("P"), ord("p")) and self._reference_values is not None:
            self._reference_visible = not self._reference_visible
        # GLFW: 右 262, 左 263, 下 264, 上 265; 不修改机器人状态.
        direction = {262: (1, 0), 263: (-1, 0), 264: (0, -1), 265: (0, 1)}.get(keycode)
        if direction is not None:
            self._enable_viewer_key_repeat()
            self._viewer_pan_requests.append(direction)

    def _enable_viewer_key_repeat(self):
        """在窗口按键线程接入 GLFW 重复事件, 保留 MuJoCo 原有按键处理."""
        if getattr(self, "_viewer_key_callback", None) is not None:
            return
        import glfw

        # MuJoCo 的 key_callback 只有键码, 不转发 REPEAT; 此处只补方向键重复事件.
        # 回调正在窗口线程执行, current context 就是该 MuJoCo 窗口, 不搜索其他窗口.
        window = glfw.get_current_context()
        if not window:
            return

        def key_event(window, key, scancode, action, mods):
            """先保持原生快捷键行为, 再将方向键重复事件交给已有平移队列."""
            if original:
                original(window, key, scancode, action, mods)
            if action == glfw.REPEAT and key in (glfw.KEY_LEFT, glfw.KEY_RIGHT, glfw.KEY_UP, glfw.KEY_DOWN):
                self._on_viewer_key(key)

        # Python GLFW 的公开 setter 不返回 MuJoCo 在 C++ 中注册的旧回调, 因此使用其原生绑定.
        # 保存 C 回调引用, 避免被回收; 窗口异步关闭期间也继续保留它.
        callback = glfw._GLFWkeyfun(key_event)
        original = glfw._glfw.glfwSetKeyCallback(window, callback)
        self._viewer_key_callback = (callback, original)

    def _sync_viewer_state(self):
        """只将所选环境的位姿搬到显示模型, 通过 mj_forward 更新几何, 不调用 mj_step."""

        import mujoco

        index = self.config.viewer_env_id
        pose = self.robot.data.root_link_pose_w.torch[index]
        joints = self.robot.data.joint_pos.torch[index]
        values = torch.cat((pose[:3] - self.scene.env_origins[index], pose[3:7], joints)).detach().cpu().numpy()
        address = self._viewer_root_address
        self._viewer_data.qpos[address:address + 3] = values[:3]
        # Isaac Lab 使用 xyzw, MuJoCo 浮动关节使用 wxyz.
        self._viewer_data.qpos[address + 3:address + 7] = values[[6, 3, 4, 5]]
        self._viewer_data.qpos[self._viewer_joint_addresses] = values[7:]
        if self._reference_values is not None:
            reference = self._reference_values
            address = self._reference_root_address
            self._viewer_data.qpos[address:address + 3] = reference[:3]
            self._viewer_data.qpos[address + 1] += self.config.reference_offset_y
            self._viewer_data.qpos[address + 3:address + 7] = reference[[6, 3, 4, 5]]
            self._viewer_data.qpos[self._reference_joint_addresses] = reference[7:]
        if self._viewer is not None:
            self._viewer.opt.geomgroup[2] = int(self._reference_visible and self._reference_values is not None)
        mujoco.mj_forward(self._viewer_model, self._viewer_data)

    def _close_viewer(self):
        """
        释放   MuJoCo   窗口及独立显示模型,          释放显示模型及相关引用，不停止或重置物理机器人.
        环境的 close()  也会调用 _close_viewer()，  确保退出时释放窗口。
        """
        # 如果显示窗口存在，就关闭它。
        # 清除窗口引用。
        # 清除显示专用的模型和数据引用。
        # 标记窗口已关闭，供后续代码判断。
        if self._viewer is not None:
            self._viewer.close()
        self._viewer = None
        self._viewer_model = self._viewer_data = None
        self._reference_values = None
        self._reference_visible = False
        self._viewer_pan_requests.clear()
        self._viewer_closed = True

    # ---------------------------------- 状态写入与刷新 ---------------------------------- #
    """
    职责：
    三个函数共同服务于 reset()：准备重置状态 → 写入仿真 → 刷新相关数据，整个过程不推进物理时间
    
    reset(env_ids, state)
        ├── 未提供 state → _default_state(env_ids)
        ├── _write_state(env_ids, state)
        └── _refresh_state()
    """

    # 获取默认重置状态
    def _default_state(self, env_ids):
        """读取所选环境的资源默认状态, 与上层参考状态使用相同六字段格式."""

        data = self.robot.data
        pose, velocity = data.default_root_pose.torch[env_ids], data.default_root_vel.torch[env_ids]
        return {"root_pos": pose[:, :3], "root_quat": pose[:, 3:7], "root_lin_vel": velocity[:, :3], "root_ang_vel": velocity[:, 3:6], "joint_pos": self.default_joint_pos[env_ids], "joint_vel": data.default_joint_vel.torch[env_ids]}

    # 实际重置指定机器人，把准备好的控制命令提交到底层仿真，不推进物理时间
    def _write_state(self, env_ids, state):
        """清理所选环境的执行器缓存, 写入根部和关节状态, 对齐 PD 目标角."""

        # 构造底层需要的组合张量, 不原地修改上层参考状态, 不额外 clone.
        pose = torch.cat((state["root_pos"] + self.scene.env_origins[env_ids], state["root_quat"]), dim=-1)
        velocity = torch.cat((state["root_lin_vel"], state["root_ang_vel"]), dim=-1)
        self.scene.reset(env_ids)
        self.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=state["joint_pos"], env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=state["joint_vel"], env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)
        self.robot.actuators.target_command.set_position_index(value=state["joint_pos"], env_ids=env_ids)
        self.scene.write_data_to_sim()

    # 根据新写入的根部和关节状态，重新计算各连杆的位置、姿态等派生量
    def _refresh_state(self):
        """根据写入状态更新运动学和读取缓存, 不调用物理 step."""

        self.sim.forward()
        self.scene.update(0.0)

    # ---------------------------------- 动作应用与物理步进 ---------------------------------- #

    def _apply_action(self, action):
        """沿用已验证的动作映射, PD 参数由 robot.py 提供, 不额外裁剪动作."""

        target = self.default_joint_pos + self.joint_offsets + self.alpha * action
        self.robot.actuators.target_command.set_position_index(value=target)

    def _advance_simulation(self):
        """提交执行器目标并推进 control_dt, 物理子步由 Newton 内部完成."""

        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(self.config.control_dt)

    # ---------------------------------- 输入检查 ---------------------------------- #

    def _check_open(self):
        """阻止访问已关闭的底层仿真资源."""

        if self.is_closed:
            raise RuntimeError("环境已关闭")

    def _check_tensor(self, value, shape, name):
        """要求状态和动作已在仿真设备上, 避免每步隐式 CPU/GPU 搬运."""

        if not isinstance(value, torch.Tensor) or value.shape != shape or value.device != self.device or value.dtype != torch.float32:
            raise ValueError(f"{name} 必须为设备 {self.device} 上的 float32 张量, 形状为 {shape}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} 必须全部为有限数值")

    def _resolve_env_ids(self, env_ids):
        """检查无重复的一维整数环境编号; 空数组不重置任何环境."""

        if env_ids is None:
            return self._all_env_ids
        ids = torch.as_tensor(env_ids, device=self.device)
        if ids.ndim != 1 or (ids.numel() and (ids.dtype == torch.bool or ids.is_floating_point() or ids.is_complex())):
            raise ValueError("env_ids 必须为一维整数数组")
        ids = ids.long()
        if torch.any((ids < 0) | (ids >= self.num_envs)) or ids.unique().numel() != ids.numel():
            raise ValueError("env_ids 超出范围或包含重复编号")
        return ids

    def _validate_state(self, state, count):
        """写入前校验六字段及单位四元数; 不复制输入, 允许额外的参考身体缓存字段."""

        result = {}
        for key, width in STATE_WIDTHS.items():
            if key not in state:
                raise ValueError(f"重置状态缺少 {key}")
            value = state[key]
            if isinstance(value, torch.Tensor) and count == 1 and value.shape == (width,):
                value = value.unsqueeze(0)
            self._check_tensor(value, (count, width), key)
            result[key] = value
        norm = result["root_quat"].norm(dim=-1)
        if not torch.allclose(norm, torch.ones_like(norm), atol=1e-4, rtol=1e-4):
            raise ValueError("root_quat 必须为 xyzw 单位四元数")
        return result

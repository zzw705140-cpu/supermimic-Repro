#########################################################################
#      从 MJCF 读取结构, 用可微 PyTorch 运算计算指定连杆的世界系位姿和速度       #
#########################################################################


from pathlib import Path
from collections.abc import Sequence

import mujoco
import numpy as np
import torch


class G1Kinematics:
    """
    加载并重采样参考动作,按动作编号和帧号提供参考状态.

    输入:
        CSV 动作数据 [T, 36]:根部位置,xyzw 四元数和 29 个关节角.
        单帧查询:clip_id [],frame_id [].
        批量查询:clip_id [B],frame_id [B].

    输出:
        根部位置,线速度和角速度:   [3] 或 [B, 3].
        根部 xyzw 四元数:        [4] 或 [B, 4].
        关节角和关节速度:         [29] 或 [B, 29].
        可选身体位置,线速度和角速度:[N, 3] 或 [B, N, 3].
        可选身体旋转矩阵:         [N, 3, 3] 或 [B, N, 3, 3].
        查询未来 K 帧时,各字段增加 K 维,如 [K, 3] 或 [B, K, 3].

    T 为原始帧数,B 为批量大小,N 为跟踪部位数;身体状态均为世界系.
    """

    """---------------- ① 初始化与模型结构读取 ----------------"""

    # 保存配置、加载模型、检查浮动根部
    def __init__(self, mjcf_path: str | Path, joint_names: Sequence[str], tracked_bodies: Sequence[str], device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype 必须为 torch.float32 或 torch.float64")

        # 保存设备和精度配置
        self.device, self.dtype = torch.device(device), dtype
        # 保存关节和跟踪部位名称，列表顺序决定输入、输出顺序
        self.joint_names, self.tracked_bodies = list(joint_names), list(tracked_bodies)
        
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names 必须非空且不重复")
        if not self.tracked_bodies or len(set(self.tracked_bodies)) != len(self.tracked_bodies):
            raise ValueError("tracked_bodies 必须非空且不重复")

        # 从 XML 文件加载机器人模型
        model = mujoco.MjModel.from_xml_path(str(Path(mjcf_path).resolve()))
        # 查找自由关节
        free_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)

        if len(free_ids) != 1:
            raise ValueError("模型必须恰好有一个浮动根部")
        root_id = int(model.jnt_bodyid[free_ids[0]])
        if model.body_parentid[root_id] != 0:
            raise ValueError("浮动根部必须直接连接世界")

        # 按名称映射输入关节, 不依赖 MuJoCo 内部的关节存储顺序.
        hinge_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
        hinges = {model.joint(int(j)).name: int(j) for j in hinge_ids}

        if len(hinges) != model.njnt - 1 or set(hinges) != set(self.joint_names):
            raise ValueError("模型必须只包含浮动根部和 joint_names 指定的转动关节")

        # 建立“关节名称 → 输入索引”的映射，保存关节数量
        joint_indices = {name: i for i, name in enumerate(self.joint_names)}
        self.num_joints = len(self.joint_names)
        # 收集所有连杆名称，记录连杆数和跟踪部位数。跳过编号 0 的世界
        self.body_names = [model.body(b).name for b in range(1, model.nbody)]
        self.num_bodies, self.num_tracked = len(self.body_names), len(self.tracked_bodies)
        if any(name not in self.body_names for name in self.tracked_bodies):
            raise ValueError("tracked_bodies 包含模型中不存在的连杆")

        # 保存跟踪部位和根连杆的内部索引。由于移除了世界，编号统一减一。
        self.tracked_indices = torch.tensor([self.body_names.index(name) for name in self.tracked_bodies], dtype=torch.long, device=self.device)
        self.root_index = root_id - 1

        # MuJoCo 按父节点在前的顺序存储, 固定连接也保留, 不跳过中间坐标变换.
        self.parents, self.joint_indices = [], []
        axes, anchors, references = [], [], []

        #遍历连杆，获取各自的父节点
        for body in range(1, model.nbody):
            parent = int(model.body_parentid[body])
            if parent == 0 and body != root_id:
                raise ValueError("暂不支持浮动机器人以外的独立世界物体")
            if parent >= body or model.body_jntnum[body] > 1:
                raise ValueError("要求父节点在前且每个连杆至多有一个关节")
            self.parents.append(parent - 1)
            if body == root_id or model.body_jntnum[body] == 0:
                self.joint_indices.append(-1)
                axes.append(np.zeros(3))
                anchors.append(np.zeros(3))
                references.append(0.0)
            else:
                joint = int(model.body_jntadr[body])
                self.joint_indices.append(joint_indices[model.joint(joint).name])
                axes.append(model.jnt_axis[joint].copy())
                anchors.append(model.jnt_pos[joint].copy())
                references.append(model.qpos0[model.jnt_qposadr[joint]])

        # 将转轴、锚点和参考角转成张量，统一设备与精度。
        self.axes = torch.tensor(np.array(axes), dtype=dtype, device=self.device)
        self.anchors = torch.tensor(np.array(anchors), dtype=dtype, device=self.device)
        self.joint_references = torch.tensor(references, dtype=dtype, device=self.device)
        # 保存各连杆相对父连杆的固定位置偏移，跳过世界节点。
        self.offset_pos = torch.tensor(model.body_pos[1:].copy(), dtype=dtype, device=self.device)
        # 读取固定姿态，将四元数从 wxyz 调整为 xyzw，再转成旋转矩阵。
        quaternion = torch.tensor(model.body_quat[1:, [1, 2, 3, 0]].copy(), dtype=dtype, device=self.device)
        self.offset_rot = self._quat_to_matrix(quaternion)

        # 未启用限位的关节用无穷范围表示, 避免把默认零范围误当成锁死关节.
        limits = [model.jnt_range[hinges[name]] if model.jnt_limited[hinges[name]] else [-np.inf, np.inf] for name in self.joint_names]
        limits = torch.tensor(np.array(limits), dtype=dtype, device=self.device)
        self.q_lo, self.q_hi = limits[:, 0], limits[:, 1]

    """---------------- ② 旋转计算工具 ----------------"""

    @staticmethod
    # 将四元数归一化并转换为旋转矩阵。
    def _quat_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
        """将 xyzw 四元数 [..., 4] 归一化并转换为旋转矩阵 [..., 3, 3], 保留梯度."""
        norm = quaternion.norm(dim=-1, keepdim=True)
        if torch.any(norm < 1e-8):
            raise ValueError("根部或模型结构包含零四元数")
        x, y, z, w = (quaternion / norm).unbind(-1)
        row0 = torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)), dim=-1)
        row1 = torch.stack((2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)), dim=-1)
        row2 = torch.stack((2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)), dim=-1)
        return torch.stack((row0, row1, row2), dim=-2)

    @staticmethod
    # 根据关节转轴和旋转角度计算旋转矩阵。
    def _axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """用 Rodrigues 公式将单位关节轴 [3] 和角度 [...] 转换为 [..., 3, 3]."""
        x, y, z = axis.unbind(-1)
        zero = torch.zeros_like(x)
        skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
        identity = torch.eye(3, device=angle.device, dtype=angle.dtype)
        return identity + angle.sin()[..., None, None] * skew + (1 - angle.cos())[..., None, None] * (skew @ skew)

    """---------------- ③ 正运动学与速度递推 ----------------"""

    # 根据根部和关节状态，计算指定身体部位在世界系中的位姿与速度。
    def forward(self, root_pos: torch.Tensor, root_quat: torch.Tensor, root_lin_vel: torch.Tensor, root_ang_vel: torch.Tensor, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> dict[str, torch.Tensor]:
        
        inputs = {"root_pos": (root_pos, 3), "root_quat": (root_quat, 4), "root_lin_vel": (root_lin_vel, 3), "root_ang_vel": (root_ang_vel, 3), "joint_pos": (joint_pos, self.num_joints), "joint_vel": (joint_vel, self.num_joints)}
        if not isinstance(root_pos, torch.Tensor) or root_pos.ndim < 1:
            raise ValueError("root_pos 必须为至少一维的 PyTorch 张量")
        leading_shape = root_pos.shape[:-1]
        for name, (value, width) in inputs.items():
            if not isinstance(value, torch.Tensor) or value.shape != leading_shape + (width,):
                raise ValueError(f"{name} 必须为形状 {leading_shape + (width,)} 的张量")
            if value.device != self.offset_pos.device or value.dtype != self.dtype:
                raise ValueError(f"{name} 的设备和 dtype 必须与 G1Kinematics 一致")

        positions, rotations, linear, angular = [], [], [], []
        for body, parent in enumerate(self.parents):
            if body == self.root_index:
                positions.append(root_pos)
                rotations.append(self._quat_to_matrix(root_quat))
                linear.append(root_lin_vel)
                angular.append(root_ang_vel)
                continue

            # 先应用父节点到当前连杆的固定变换.
            parent_rot = rotations[parent]
            fixed_rot  = parent_rot @ self.offset_rot[body]
            position   = positions[parent] + (parent_rot @ self.offset_pos[body, :, None]).squeeze(-1)
            rotation   = fixed_rot
            velocity   = linear[parent] + torch.cross(angular[parent], position - positions[parent], dim=-1)
            omega      = angular[parent]
            joint      = self.joint_indices[body]

            # 再围绕关节轴旋转, 同时处理非零关节锚点和模型参考角偏移.
            if joint >= 0:
                turn               = self._axis_angle_to_matrix(self.axes[body], joint_pos[..., joint] - self.joint_references[body])
                rotation           = fixed_rot @ turn
                correction         = (fixed_rot @ (self.anchors[body, :, None] - turn @ self.anchors[body, :, None])).squeeze(-1)
                position           = position + correction
                joint_omega        = (fixed_rot @ self.axes[body, :, None]).squeeze(-1) * joint_vel[..., joint, None]
                origin_from_anchor = -(rotation @ self.anchors[body, :, None]).squeeze(-1)
                velocity           = velocity + torch.cross(angular[parent], correction, dim=-1) + torch.cross(joint_omega, origin_from_anchor, dim=-1)
                omega              = omega + joint_omega

            positions.append(position)
            rotations.append(rotation)
            linear.append(velocity)
            angular.append(omega)

        """---------------- ④ 整理身体状态输出 ----------------"""

        # 沿身体维堆叠并筛选, 原样保留批量维和 autograd 计算图.
        return {
            "body_pos": torch.stack(positions, dim=-2).index_select(-2, self.tracked_indices),
            "body_rot": torch.stack(rotations, dim=-3).index_select(-3, self.tracked_indices),
            "body_lin_vel": torch.stack(linear, dim=-2).index_select(-2, self.tracked_indices),
            "body_ang_vel": torch.stack(angular, dim=-2).index_select(-2, self.tracked_indices),
        }

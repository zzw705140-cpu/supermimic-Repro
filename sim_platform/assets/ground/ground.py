"""地面的基础物理属性和预览资源路径, 训练时的随机化由环境层负责."""

from pathlib import Path


# 静摩擦系数与动摩擦系数, 均为无量纲参数.
static_friction = 1.0
dynamic_friction = 1.0

# MuJoCo 预览的棋盘格材质, 天空与基础照明统一保存在资源目录.
GROUND_VISUAL_PATH = Path(__file__).resolve().with_name("ground_visual.xml")

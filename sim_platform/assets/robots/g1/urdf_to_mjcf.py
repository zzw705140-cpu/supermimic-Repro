"""将配套 G1 URDF 转为具有浮动基座的 MuJoCo XML。

在安装了 mujoco 的 Python 环境中直接运行本脚本。
"""

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
URDF_PATH = ROOT / "g1_29dof_rev_1_0.urdf"
MJCF_PATH = ROOT / "g1_29dof.xml"


def convert() -> Path:
    """保留连杆坐标系和可视几何，生成并检查 MJCF。"""
    import mujoco

    tree = ET.parse(URDF_PATH)
    root = tree.getroot()
    for mesh in root.iter("mesh"):
        path = ROOT / mesh.attrib["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"缺少机器人几何模型：{path}")

    # 延续参考项目的导入设置；不修改原始 URDF。
    extension = root.find("mujoco")
    if extension is None:
        extension = ET.SubElement(root, "mujoco")
    compiler = extension.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(extension, "compiler")
    compiler.attrib.update(
        meshdir=".", balanceinertia="true", discardvisual="false",
        fusestatic="false", strippath="false",
    )

    # 临时文件与资源放在同一目录，保持 meshes/ 相对路径有效。
    with tempfile.TemporaryDirectory(prefix=".g1_conversion_", dir=ROOT) as temporary:
        temporary = Path(temporary)
        source = temporary / "g1.urdf"
        compiler.set("meshdir", str(ROOT))
        tree.write(source, encoding="utf-8", xml_declaration=True)
        model = mujoco.MjModel.from_xml_path(str(source))
        output = temporary / "g1.xml"
        mujoco.mj_saveLastXML(str(output), model)

        mjcf = ET.parse(output)
        pelvis = mjcf.getroot().find("worldbody/body[@name='pelvis']")
        if pelvis is None:
            raise ValueError("转换结果缺少 pelvis 根刚体。")
        if pelvis.find("freejoint") is None:
            ET.SubElement(pelvis, "freejoint", name="floating_base_joint")
        # 最终 XML 与 meshes/ 同级，使用可随项目移动的相对路径。
        mjcf.getroot().find("compiler").set("meshdir", ".")
        candidate = ROOT / (temporary.name + ".xml")
        try:
            mjcf.write(candidate, encoding="utf-8", xml_declaration=True)
            model = mujoco.MjModel.from_xml_path(str(candidate))
            if (model.nq, model.nv) != (36, 35):
                raise ValueError(f"预期 29 关节加浮动基座，实际 nq={model.nq}, nv={model.nv}")
            candidate.replace(MJCF_PATH)
        finally:
            candidate.unlink(missing_ok=True)

    print(f"已生成 {MJCF_PATH}")
    print(f"nq={model.nq}, nv={model.nv}, nbody={model.nbody}, ngeom={model.ngeom}")
    return MJCF_PATH


if __name__ == "__main__":
    convert()

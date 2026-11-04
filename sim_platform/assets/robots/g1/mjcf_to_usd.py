"""使用独立 Isaac Sim MJCF 导入器将 G1 XML 转为 USD。

先运行 urdf_to_mjcf.py，再在具备参考项目所用独立导入器的环境中运行。
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
MJCF_PATH = ROOT / "g1_29dof.xml"
USD_DIR = ROOT / "usd"


def convert():
    """注册导入器 schema，并将转换资源统一输出到 usd/。"""
    if not MJCF_PATH.is_file():
        raise FileNotFoundError(f"缺少 {MJCF_PATH}，请先运行 urdf_to_mjcf.py。")

    from pxr import Plug
    import isaacsim.asset.importer.utils.impl as importer_utils

    # 与参考项目一致：在导入 MJCF 导入器之前注册独立运行所需的 schema。
    schemas = Path(importer_utils.__file__).resolve().parent / "schemas"
    Plug.Registry().RegisterPlugins(str(schemas))
    from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

    USD_DIR.mkdir(parents=True, exist_ok=True)
    config = MJCFImporterConfig(mjcf_path=str(MJCF_PATH), usd_path=str(USD_DIR))
    result = MJCFImporter(config).import_mjcf()
    print(result)
    print(f"USD 输出目录：{USD_DIR}")
    return result


if __name__ == "__main__":
    convert()

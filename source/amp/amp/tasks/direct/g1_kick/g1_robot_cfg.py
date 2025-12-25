# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
宇树G1机器人配置。

使用Isaac Lab内置的G1机器人配置。
"""

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

# 尝试从isaaclab_assets导入G1配置
G1_CFG = None
G1_CFG_NAME = None

# 尝试多种可能的导入路径
import_paths = [
    ("isaaclab_assets.robots.unitree.g1", "G1_CFG"),
    ("isaaclab_assets.robots.unitree", "G1_CFG"),
    ("isaaclab_assets.robots.unitree_g1", "G1_CFG"),
    ("isaaclab_assets.robots", "G1_CFG"),
    ("isaaclab_assets.robots.unitree.g1", "UNITREE_G1_CFG"),
    ("isaaclab_assets.robots.unitree", "UNITREE_G1_CFG"),
]

for module_path, cfg_name in import_paths:
    try:
        module = __import__(module_path, fromlist=[cfg_name])
        if hasattr(module, cfg_name):
            G1_CFG = getattr(module, cfg_name)
            G1_CFG_NAME = f"{module_path}.{cfg_name}"
            break
    except (ImportError, AttributeError):
        continue

if G1_CFG is not None:
    # 使用Isaac Lab内置的G1配置
    G1RobotCfg = G1_CFG
    # 在运行时输出信息（通过__init__方法）
    _USING_BUILTIN_G1 = True
    _G1_CFG_SOURCE = G1_CFG_NAME
else:
    # 如果找不到内置配置，尝试使用Isaac Lab中的其他人形机器人配置作为临时替代
    _USING_BUILTIN_G1 = False
    _G1_CFG_SOURCE = None

    # 尝试使用其他人形机器人配置
    try:
        from isaaclab_assets.robots.humanoid import HUMANOID_CFG
        G1RobotCfg = HUMANOID_CFG
        print("[WARNING] 未找到G1配置，使用HUMANOID_CFG作为临时替代。")
        print("[WARNING] 请确保isaaclab_assets已正确安装，或手动指定G1的USD文件路径。")
    except ImportError:
        # 如果都找不到，创建一个基础配置类
        @configclass
        class G1RobotCfg(ArticulationCfg):
            """宇树G1机器人配置（备用方案）"""

            # 如果找不到内置配置，需要手动指定USD文件路径
            # 示例：
            # usd_path = "/path/to/g1.usd"
            # 或者使用Isaac Lab的其他机器人配置作为替代
            pass

        print("[ERROR] 未找到G1配置，且无法使用备用机器人配置。")
        print("[ERROR] 请确保：")
        print("[ERROR]   1. isaaclab_assets已正确安装")
        print("[ERROR]   2. 或手动在g1_robot_cfg.py中指定G1的USD文件路径")


def get_g1_config_info():
    """获取G1配置信息（用于调试）"""
    if _USING_BUILTIN_G1:
        return {
            "using_builtin": True,
            "source": _G1_CFG_SOURCE,
            "config": G1RobotCfg,
        }
    else:
        return {
            "using_builtin": False,
            "source": None,
            "config": G1RobotCfg,
            "message": "使用备用配置，需要手动指定USD文件路径",
        }


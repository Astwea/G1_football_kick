# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
宇树G1机器人配置。

使用项目中的g1_29dof_rev_1_0.usd文件（29自由度，不含灵巧手）。
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# G1机器人USD文件路径
g1_usd_path = "/home/astwea/amp/source/amp/assets/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd"

# G1机器人配置
G1RobotCfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=g1_usd_path),
    actuators={"joint_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)

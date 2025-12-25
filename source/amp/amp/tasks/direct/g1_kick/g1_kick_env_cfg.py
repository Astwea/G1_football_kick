# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
宇树G1踢球环境配置。
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .g1_robot_cfg import G1RobotCfg


@configclass
class G1KickEnvCfg(DirectRLEnvCfg):
    """宇树G1踢球环境配置"""

    # 环境基本参数
    decimation = 2  # 控制频率降采样
    episode_length_s = 10.0  # 每回合时长（秒）

    # 观察空间和动作空间
    # G1机器人有29个自由度
    # 观察空间包括：关节位置(29) + 关节速度(29) + 根位置(3) + 根旋转(4) + 根线速度(3) + 根角速度(3) 
    #              + 球绝对位置(3) + 球相对位置(3) + 球旋转(4) + 球线速度(3) + 球角速度(3)
    # = 29 + 29 + 3 + 4 + 3 + 3 + 3 + 3 + 4 + 3 + 3 = 87
    action_space = 29  # 29个自由度的动作
    observation_space = 87  # 观察空间维度（87维）
    state_space = 0  # AMP不需要显式状态空间

    # 仿真配置
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # 机器人配置
    robot_cfg: ArticulationCfg = G1RobotCfg.replace(prim_path="/World/envs/env_.*/Robot")

    # 球体配置（按照Isaac Lab官方示例的方式）
    ball_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.11,  # 标准足球半径（米）
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.43),  # 标准足球质量（kg）
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2),  # 红色
                metallic=0.0,
                roughness=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.0, 0.0, 0.1),  # 球体初始位置（机器人前方1米）
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )

    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,  # 并行环境数量（从4096减少到1024以节省GPU内存）
        env_spacing=4.0,  # 环境间距
        replicate_physics=True,
    )

    # 奖励权重
    # 踢球奖励
    rew_scale_ball_velocity = 1.0  # 球的速度奖励
    rew_scale_ball_forward = 2.0  # 球向前移动的奖励
    rew_scale_ball_distance = 0.5  # 球距离目标的奖励

    # 动作模仿奖励（由AMP判别器提供）
    # 这些权重在AMP配置中设置

    # 平衡和稳定性奖励
    rew_scale_alive = 1.0  # 存活奖励
    rew_scale_terminated = -10.0  # 终止惩罚
    rew_scale_upright = 1.0  # 保持直立的奖励
    rew_scale_energy = -0.01  # 能量消耗惩罚

    # 任务完成奖励
    rew_scale_kick_success = 10.0  # 成功踢球的奖励

    # 重置参数
    # 机器人初始位置随机化
    initial_robot_pos_range = [(-0.1, 0.1), (-0.1, 0.1), (0.95, 1.05)]  # x, y, z范围
    initial_robot_rot_range = [(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)]  # roll, pitch, yaw范围（弧度）

    # 球体初始位置随机化
    initial_ball_pos_range = [(0.8, 1.2), (-0.2, 0.2), (0.08, 0.12)]  # x, y, z范围

    # 终止条件
    max_robot_tilt = 0.5  # 最大倾斜角度（弧度）
    max_robot_height = 2.0  # 最大高度（米）
    min_robot_height = 0.3  # 最小高度（米，低于此值视为跌倒）

    # 动作缩放
    action_scale = 1.0  # 动作缩放因子（对于位置控制，通常为1.0）

    # 参考动作数据路径（用于AMP训练）
    motion_file_path = ""  # 将在训练时通过命令行参数指定


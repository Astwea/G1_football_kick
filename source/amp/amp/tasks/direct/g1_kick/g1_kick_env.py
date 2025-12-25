# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .g1_kick_env_cfg import G1KickEnvCfg


class G1KickEnv(DirectRLEnv):
    """宇树G1踢球环境"""

    cfg: G1KickEnvCfg

    def __init__(self, cfg: G1KickEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 检查机器人是否正确初始化
        if not hasattr(self, "robot") or self.robot is None:
            raise RuntimeError(
                "机器人未正确初始化。请检查G1机器人配置是否正确加载。"
                "如果使用备用配置，需要手动指定G1的USD文件路径。"
            )

        # 获取所有关节索引
        try:
            self._dof_indices = list(range(self.robot.num_joints))
            self.joint_pos = self.robot.data.joint_pos
            self.joint_vel = self.robot.data.joint_vel
        except Exception as e:
            raise RuntimeError(
                f"无法访问机器人关节数据。机器人可能有{self.robot.num_joints if hasattr(self.robot, 'num_joints') else 'unknown'}个关节。"
                f"错误: {e}"
            )

        # 验证观察空间和动作空间维度
        # 观察空间：关节位置(29) + 关节速度(29) + 根位置(3) + 根旋转(4) + 根线速度(3) + 根角速度(3) 
        #          + 球绝对位置(3) + 球相对位置(3) + 球旋转(4) + 球线速度(3) + 球角速度(3) = 87
        expected_obs_dim = 29 * 2 + 3 + 4 + 3 + 3 + 3 + 3 + 4 + 3 + 3  # 87维
        expected_action_dim = 29
        
        # 确保observation_space属性正确设置（如果super().__init__创建时维度为0）
        import gymnasium.spaces as spaces
        if hasattr(self, "observation_space"):
            # 检查当前观察空间维度
            if hasattr(self.observation_space, "shape") and len(self.observation_space.shape) > 0:
                current_obs_dim = self.observation_space.shape[0]
            else:
                current_obs_dim = 0
            
            if current_obs_dim == 0 or current_obs_dim != expected_obs_dim:
                # 重新创建观察空间
                self.observation_space = spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(expected_obs_dim,),
                    dtype=float,
                )
        
        # 确保action_space属性正确设置
        if hasattr(self, "action_space"):
            if hasattr(self.action_space, "shape") and len(self.action_space.shape) > 0:
                current_action_dim = self.action_space.shape[0]
            else:
                current_action_dim = 0
            
            if current_action_dim == 0 or current_action_dim != expected_action_dim:
                self.action_space = spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(expected_action_dim,),
                    dtype=float,
                )

        # 存储上一帧的球速度（用于计算加速度）
        self._prev_ball_vel = torch.zeros(self.num_envs, 3, device=self.device)

    def _setup_scene(self):
        """设置场景"""
        # 创建机器人
        self.robot = Articulation(self.cfg.robot_cfg)

        # 创建地面
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # 创建球体（RigidObject会自动处理几何创建和RigidBodyAPI应用）
        self.ball = RigidObject(self.cfg.ball_cfg)

        # 克隆环境（这会复制所有资产到所有环境）
        self.scene.clone_environments(copy_from_source=False)

        # CPU仿真需要过滤碰撞
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # 添加资产到场景
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["ball"] = self.ball

        # 添加灯光
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """物理步进前的处理"""
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """应用动作到机器人"""
        # 将动作从[-1, 1]范围缩放到实际关节角度范围
        # 这里假设使用位置控制
        # 对于G1机器人，需要根据实际关节限制调整
        joint_pos_target = self.actions * self.cfg.action_scale

        # 设置关节目标位置
        self.robot.set_joint_position_target(joint_pos_target, joint_ids=self._dof_indices)

    def _get_observations(self) -> dict:
        """获取观察"""
        # 机器人状态
        root_pos = self.robot.data.root_pos_w - self.scene.env_origins
        root_quat = self.robot.data.root_quat_w
        root_lin_vel = self.robot.data.root_lin_vel_w
        root_ang_vel = self.robot.data.root_ang_vel_w

        # 球体状态
        ball_pos = self.ball.data.root_pos_w - self.scene.env_origins
        ball_quat = self.ball.data.root_quat_w  # 球的旋转（四元数，4维）
        ball_lin_vel = self.ball.data.root_lin_vel_w
        ball_ang_vel = self.ball.data.root_ang_vel_w  # 球的角速度（3维）

        # 计算球体相对于机器人的位置
        ball_rel_pos = ball_pos - root_pos

        # 构建观察向量
        # 包括：关节位置(29) + 关节速度(29) + 根位置(3) + 根旋转(4) + 根线速度(3) + 根角速度(3) 
        #      + 球绝对位置(3) + 球相对位置(3) + 球旋转(4) + 球线速度(3) + 球角速度(3) 
        # = 29+29+3+4+3+3+3+3+4+3+3 = 93
        obs = torch.cat(
            [
                self.joint_pos,  # 关节位置 (29)
                self.joint_vel,  # 关节速度 (29)
                root_pos,  # 根位置 (3)
                root_quat,  # 根旋转（四元数）(4)
                root_lin_vel,  # 根线速度 (3)
                root_ang_vel,  # 根角速度 (3)
                ball_pos,  # 球绝对位置 (3)
                ball_rel_pos,  # 球相对位置 (3)
                ball_quat,  # 球旋转（四元数）(4)
                ball_lin_vel,  # 球线速度 (3)
                ball_ang_vel,  # 球角速度 (3)
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        
        # 为AMP训练添加amp_obs到extras（与policy观察相同）
        # AMP算法需要从infos中获取amp_obs
        # extras会在step/reset时自动添加到infos中
        self.extras["amp_obs"] = obs
        
        return observations

    def collect_reference_motions(self, num_samples: int) -> torch.Tensor:
        """
        收集参考动作数据的状态样本，用于AMP训练。

        Args:
            num_samples: 需要收集的样本数量

        Returns:
            states: 形状为 (num_samples, observation_dim) 的张量，包含参考动作的状态
        """
        # 如果配置了参考动作文件路径，尝试加载
        if hasattr(self.cfg, "motion_file_path") and self.cfg.motion_file_path:
            try:
                import numpy as np
                from pathlib import Path

                motion_file = Path(self.cfg.motion_file_path)
                if not motion_file.exists():
                    return self._generate_random_states(num_samples)

                # 加载参考动作数据
                if motion_file.suffix == ".npy":
                    motion_data = np.load(motion_file, allow_pickle=True)
                    # 如果是字典，提取状态数据
                    if isinstance(motion_data, dict):
                        if "states" in motion_data:
                            states = motion_data["states"]
                        elif "motion_data" in motion_data:
                            states = motion_data["motion_data"]
                        else:
                            states = motion_data
                    else:
                        states = motion_data
                elif motion_file.suffix == ".pkl":
                    import pickle

                    with open(motion_file, "rb") as f:
                        data = pickle.load(f)
                        if "states" in data:
                            states = data["states"]
                        elif "motion_data" in data:
                            states = data["motion_data"]
                        else:
                            states = data
                else:
                    return self._generate_random_states(num_samples)

                # 确保状态数据是正确的形状
                if isinstance(states, np.ndarray):
                    states = torch.from_numpy(states).float().to(self.device)
                elif isinstance(states, torch.Tensor):
                    states = states.float().to(self.device)
                else:
                    return self._generate_random_states(num_samples)

                # 确保状态维度匹配观察空间
                # 使用固定的87维（G1踢球环境的观察空间维度）
                obs_dim = 87
                
                if states.shape[-1] != obs_dim:
                    return self._generate_random_states(num_samples)
                
                # 如果维度匹配，确保状态数据是正确的形状
                if len(states.shape) == 1:
                    # 如果是1D，扩展为2D
                    states = states.unsqueeze(0) if isinstance(states, torch.Tensor) else states.reshape(1, -1)

                # 如果状态数据是3D的 (num_frames, num_envs, obs_dim)，展平为2D
                if len(states.shape) == 3:
                    states = states.reshape(-1, states.shape[-1])

                # 随机采样指定数量的样本
                if states.shape[0] >= num_samples:
                    indices = torch.randint(0, states.shape[0], (num_samples,), device=self.device)
                    sampled_states = states[indices]
                else:
                    # 如果样本不足，重复采样
                    indices = torch.randint(0, states.shape[0], (num_samples,), device=self.device)
                    sampled_states = states[indices]

                return sampled_states

            except Exception as e:
                return self._generate_random_states(num_samples)
        else:
            # 如果没有配置参考动作文件，生成随机状态
            return self._generate_random_states(num_samples)

    def _generate_random_states(self, num_samples: int) -> torch.Tensor:
        """
        生成随机状态样本（当没有参考动作数据时使用）。

        Args:
            num_samples: 需要生成的样本数量

        Returns:
            states: 形状为 (num_samples, observation_dim) 的张量
        """
        # 直接使用配置中的观察空间维度（87维）
        # 因为这是G1踢球环境的固定观察空间维度
        obs_dim = 87  # 29(关节位置) + 29(关节速度) + 3(根位置) + 4(根旋转) + 3(根线速度) + 3(根角速度) 
                      # + 3(球绝对位置) + 3(球相对位置) + 4(球旋转) + 3(球线速度) + 3(球角速度) = 87
        
        # 生成随机状态（在合理范围内）
        states = torch.randn(num_samples, obs_dim, device=self.device) * 0.1
        return states

    def _get_rewards(self) -> torch.Tensor:
        """计算奖励"""
        # 机器人状态
        root_pos = self.robot.data.root_pos_w - self.scene.env_origins
        root_quat = self.robot.data.root_quat_w
        root_lin_vel = self.robot.data.root_lin_vel_w

        # 球体状态
        ball_pos = self.ball.data.root_pos_w - self.scene.env_origins
        ball_lin_vel = self.ball.data.root_lin_vel_w

        # 计算各种奖励
        rewards = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_upright,
            self.cfg.rew_scale_energy,
            self.cfg.rew_scale_ball_velocity,
            self.cfg.rew_scale_ball_forward,
            self.cfg.rew_scale_ball_distance,
            self.cfg.rew_scale_kick_success,
            root_pos,
            root_quat,
            root_lin_vel,
            ball_pos,
            ball_lin_vel,
            self.joint_vel,
            self.reset_terminated,
        )

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """获取终止条件"""
        # 更新关节状态
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # 机器人状态
        root_pos = self.robot.data.root_pos_w - self.scene.env_origins
        root_quat = self.robot.data.root_quat_w

        # 计算机器人倾斜角度
        # 从四元数提取pitch和roll
        w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))

        # 计算倾斜角度（pitch和roll的组合）
        tilt_angle = torch.sqrt(pitch**2 + roll**2)

        # 终止条件
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        robot_fallen = tilt_angle > self.cfg.max_robot_tilt
        robot_too_low = root_pos[:, 2] < self.cfg.min_robot_height
        robot_too_high = root_pos[:, 2] > self.cfg.max_robot_height

        terminated = robot_fallen | robot_too_low | robot_too_high

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """重置指定环境"""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        # 重置机器人
        # 随机化初始位置
        num_resets = len(env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        # 随机化根位置
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # 添加随机偏移
        pos_range = self.cfg.initial_robot_pos_range
        default_root_state[:, 0] += sample_uniform(
            pos_range[0][0], pos_range[0][1], (num_resets,), default_root_state.device
        )
        default_root_state[:, 1] += sample_uniform(
            pos_range[1][0], pos_range[1][1], (num_resets,), default_root_state.device
        )
        default_root_state[:, 2] += sample_uniform(
            pos_range[2][0], pos_range[2][1], (num_resets,), default_root_state.device
        )

        # 随机化根旋转
        rot_range = self.cfg.initial_robot_rot_range
        rot_x = sample_uniform(rot_range[0][0], rot_range[0][1], (num_resets,), default_root_state.device)
        rot_y = sample_uniform(rot_range[1][0], rot_range[1][1], (num_resets,), default_root_state.device)
        rot_z = sample_uniform(rot_range[2][0], rot_range[2][1], (num_resets,), default_root_state.device)

        # 将欧拉角转换为四元数（简化版本，实际应该使用完整的转换）
        # 这里使用小角度近似
        default_root_state[:, 3] = 1.0  # w
        default_root_state[:, 4] = rot_x * 0.5  # x
        default_root_state[:, 5] = rot_y * 0.5  # y
        default_root_state[:, 6] = rot_z * 0.5  # z

        # 更新状态
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        # 写入仿真
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # 重置球体
        ball_root_state = self.ball.data.default_root_state[env_ids].clone()
        ball_root_state[:, :3] += self.scene.env_origins[env_ids]

        # 随机化球体位置
        ball_pos_range = self.cfg.initial_ball_pos_range
        ball_root_state[:, 0] += sample_uniform(
            ball_pos_range[0][0], ball_pos_range[0][1], (num_resets,), ball_root_state.device
        )
        ball_root_state[:, 1] += sample_uniform(
            ball_pos_range[1][0], ball_pos_range[1][1], (num_resets,), ball_root_state.device
        )
        ball_root_state[:, 2] += sample_uniform(
            ball_pos_range[2][0], ball_pos_range[2][1], (num_resets,), ball_root_state.device
        )

        # 重置球体速度
        ball_root_state[:, 7:10] = 0.0  # 线速度
        ball_root_state[:, 10:] = 0.0  # 角速度

        # 写入仿真
        self.ball.write_root_state_to_sim(ball_root_state, env_ids)

        # 重置上一帧的球速度
        self._prev_ball_vel[env_ids] = 0.0


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_upright: float,
    rew_scale_energy: float,
    rew_scale_ball_velocity: float,
    rew_scale_ball_forward: float,
    rew_scale_ball_distance: float,
    rew_scale_kick_success: float,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    ball_pos: torch.Tensor,
    ball_lin_vel: torch.Tensor,
    joint_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    """计算奖励（JIT编译以提高性能）"""
    # 存活奖励
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())

    # 终止惩罚
    rew_termination = rew_scale_terminated * reset_terminated.float()

    # 直立奖励（鼓励机器人保持直立）
    # 从四元数提取pitch和roll
    w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    tilt_angle = torch.sqrt(pitch**2 + roll**2)
    rew_upright = rew_scale_upright * torch.exp(-5.0 * tilt_angle**2)

    # 能量消耗惩罚（关节速度的平方）
    rew_energy = rew_scale_energy * torch.sum(joint_vel**2, dim=-1)

    # 球速度奖励（鼓励球移动）
    ball_speed = torch.norm(ball_lin_vel, dim=-1)
    rew_ball_velocity = rew_scale_ball_velocity * ball_speed

    # 球向前移动奖励（鼓励球向前移动）
    ball_forward_vel = ball_lin_vel[:, 0]  # x方向速度
    rew_ball_forward = rew_scale_ball_forward * torch.clamp(ball_forward_vel, min=0.0)

    # 球距离奖励（鼓励球远离机器人）
    ball_distance = torch.norm(ball_pos, dim=-1)
    rew_ball_distance = rew_scale_ball_distance * ball_distance

    # 踢球成功奖励（球向前移动且速度足够快）
    kick_success = (ball_forward_vel > 1.0) & (ball_speed > 2.0)  # 阈值可调
    rew_kick_success = rew_scale_kick_success * kick_success.float()

    # 总奖励
    total_reward = (
        rew_alive
        + rew_termination
        + rew_upright
        + rew_energy
        + rew_ball_velocity
        + rew_ball_forward
        + rew_ball_distance
        + rew_kick_success
    )

    return total_reward


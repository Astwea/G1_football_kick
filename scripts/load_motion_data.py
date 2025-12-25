#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
加载参考动作数据，用于AMP训练。

将转换后的机器人关节角度数据加载到AMP的motion_dataset中。
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch


def load_motion_data(file_path: str, device: str = "cpu") -> dict:
    """
    加载动作数据文件。

    Args:
        file_path: 动作数据文件路径（.pkl或.npy）
        device: 设备（'cpu'或'cuda'）

    Returns:
        包含动作数据的字典
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if file_path.suffix == ".pkl":
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    elif file_path.suffix == ".npy":
        # 如果是numpy文件，假设只包含关节角度
        joint_positions = np.load(file_path)
        data = {
            "joint_positions": joint_positions,
            "joint_velocities": None,  # 需要计算
            "num_frames": joint_positions.shape[0],
            "robot_dof": joint_positions.shape[1],
        }
        # 计算关节速度（如果不存在）
        if data["joint_velocities"] is None and data["num_frames"] > 1:
            dt = 1.0 / 30.0  # 假设30 FPS
            velocities = np.zeros_like(joint_positions)
            velocities[1:] = (joint_positions[1:] - joint_positions[:-1]) / dt
            data["joint_velocities"] = velocities
    else:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")

    print(f"[INFO] 加载动作数据: {data['num_frames']} 帧, {data['robot_dof']} 自由度")
    print(f"[INFO] 关节位置形状: {data['joint_positions'].shape}")
    if data["joint_velocities"] is not None:
        print(f"[INFO] 关节速度形状: {data['joint_velocities'].shape}")

    return data


def prepare_amp_motion_data(motion_data: dict, observation_dim: int) -> np.ndarray:
    """
    准备AMP训练所需的动作数据格式。

    AMP需要的是观察数据（state），而不是动作数据。
    这里我们需要根据关节角度构建状态向量。

    Args:
        motion_data: 动作数据字典
        observation_dim: 观察空间维度

    Returns:
        形状为 (num_samples, observation_dim) 的状态数据
    """
    joint_positions = motion_data["joint_positions"]
    joint_velocities = motion_data.get("joint_velocities")

    num_frames = joint_positions.shape[0]
    robot_dof = joint_positions.shape[1]

    # 构建状态向量
    # 状态包括：关节位置、关节速度、根位置、根旋转、根速度等
    # 这里我们简化处理，只使用关节位置和速度
    # 实际使用时，需要根据环境的观察空间定义来构建

    states = []
    for i in range(num_frames):
        state = joint_positions[i].copy()

        # 添加关节速度（如果存在）
        if joint_velocities is not None:
            state = np.concatenate([state, joint_velocities[i]])

        # 如果状态维度不够，用零填充
        # 如果状态维度过多，截断
        if len(state) < observation_dim:
            state = np.pad(state, (0, observation_dim - len(state)), mode="constant")
        elif len(state) > observation_dim:
            state = state[:observation_dim]

        states.append(state)

    states = np.array(states, dtype=np.float32)

    print(f"[INFO] 准备AMP状态数据: {states.shape}")
    return states


def save_amp_motion_data(states: np.ndarray, output_path: str):
    """
    保存AMP格式的动作数据。

    Args:
        states: 状态数据数组
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存为numpy文件
    np.save(output_path, states)
    print(f"[INFO] AMP动作数据已保存到: {output_path}")

    # 同时保存为pickle文件（包含元数据）
    pkl_path = output_path.with_suffix(".pkl")
    data = {
        "states": states,
        "num_samples": states.shape[0],
        "state_dim": states.shape[1],
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[INFO] AMP动作数据（pickle格式）已保存到: {pkl_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="加载和准备AMP训练的动作数据")
    parser.add_argument("--input", type=str, required=True, help="输入动作数据文件路径")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径（可选）")
    parser.add_argument(
        "--observation_dim", type=int, default=None, help="观察空间维度（如果已知）"
    )

    args = parser.parse_args()

    # 加载数据
    motion_data = load_motion_data(args.input)

    # 准备AMP格式数据
    # 如果未指定观察维度，使用关节数据的维度
    if args.observation_dim is None:
        # 估算观察维度：关节位置 + 关节速度 + 根状态等
        # 这里使用一个合理的默认值
        observation_dim = motion_data["robot_dof"] * 2 + 13  # 关节位置+速度+根状态(7+6)
    else:
        observation_dim = args.observation_dim

    amp_states = prepare_amp_motion_data(motion_data, observation_dim)

    # 保存数据
    if args.output is None:
        input_path = Path(args.input)
        args.output = input_path.parent / f"{input_path.stem}_amp.npy"

    save_amp_motion_data(amp_states, args.output)


if __name__ == "__main__":
    main()


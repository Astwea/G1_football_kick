#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
将人体姿态数据转换为机器人关节角度数据。

将MediaPipe提取的人体关键点映射到宇树G1机器人的关节空间。
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import scipy.spatial.transform as transform


class HumanToRobotConverter:
    """将人体姿态转换为机器人关节角度"""

    def __init__(self, robot_dof: int = 29):
        """
        初始化转换器。

        Args:
            robot_dof: 机器人自由度数量（G1有29个自由度）
        """
        self.robot_dof = robot_dof
        # MediaPipe关键点索引（33个关键点）
        # 0-10: 面部和上身
        # 11-16: 左臂
        # 17-22: 右臂
        # 23-28: 左腿
        # 29-32: 右腿
        self.keypoint_indices = {
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_elbow": 13,
            "right_elbow": 14,
            "left_wrist": 15,
            "right_wrist": 16,
            "left_hip": 23,
            "right_hip": 24,
            "left_knee": 25,
            "right_knee": 26,
            "left_ankle": 27,
            "right_ankle": 28,
            "left_foot": 31,
            "right_foot": 32,
        }

    def compute_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        计算三点之间的角度（p2为顶点）。

        Args:
            p1: 第一个点
            p2: 顶点
            p3: 第三个点

        Returns:
            角度（弧度）
        """
        v1 = p1 - p2
        v2 = p3 - p2
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm < 1e-6 or v2_norm < 1e-6:
            return 0.0
        cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
        return np.arccos(cos_angle)

    def compute_joint_angles_from_pose(self, landmarks: np.ndarray) -> np.ndarray:
        """
        从人体关键点计算关节角度。

        Args:
            landmarks: 形状为 (33, 3) 的关键点数组

        Returns:
            关节角度数组，形状为 (robot_dof,)
        """
        # 提取关键点
        left_shoulder = landmarks[self.keypoint_indices["left_shoulder"]]
        right_shoulder = landmarks[self.keypoint_indices["right_shoulder"]]
        left_elbow = landmarks[self.keypoint_indices["left_elbow"]]
        right_elbow = landmarks[self.keypoint_indices["right_elbow"]]
        left_wrist = landmarks[self.keypoint_indices["left_wrist"]]
        right_wrist = landmarks[self.keypoint_indices["right_wrist"]]
        left_hip = landmarks[self.keypoint_indices["left_hip"]]
        right_hip = landmarks[self.keypoint_indices["right_hip"]]
        left_knee = landmarks[self.keypoint_indices["left_knee"]]
        right_knee = landmarks[self.keypoint_indices["right_knee"]]
        left_ankle = landmarks[self.keypoint_indices["left_ankle"]]
        right_ankle = landmarks[self.keypoint_indices["right_ankle"]]

        # 计算关节角度
        joint_angles = np.zeros(self.robot_dof, dtype=np.float32)

        # 左肩关节（俯仰和偏航）
        if np.linalg.norm(left_shoulder - left_elbow) > 1e-6:
            shoulder_angle = self.compute_angle(left_hip, left_shoulder, left_elbow)
            joint_angles[0] = shoulder_angle - np.pi / 2  # 归一化到合理范围

        # 左肘关节
        if np.linalg.norm(left_elbow - left_wrist) > 1e-6:
            elbow_angle = self.compute_angle(left_shoulder, left_elbow, left_wrist)
            joint_angles[1] = np.pi - elbow_angle  # 肘关节弯曲角度

        # 右肩关节
        if np.linalg.norm(right_shoulder - right_elbow) > 1e-6:
            shoulder_angle = self.compute_angle(right_hip, right_shoulder, right_elbow)
            joint_angles[2] = shoulder_angle - np.pi / 2

        # 右肘关节
        if np.linalg.norm(right_elbow - right_wrist) > 1e-6:
            elbow_angle = self.compute_angle(right_shoulder, right_elbow, right_wrist)
            joint_angles[3] = np.pi - elbow_angle

        # 左髋关节（俯仰）
        if np.linalg.norm(left_hip - left_knee) > 1e-6:
            hip_angle = self.compute_angle(left_shoulder, left_hip, left_knee)
            joint_angles[4] = hip_angle - np.pi / 2

        # 左膝关节
        if np.linalg.norm(left_knee - left_ankle) > 1e-6:
            knee_angle = self.compute_angle(left_hip, left_knee, left_ankle)
            joint_angles[5] = np.pi - knee_angle

        # 左踝关节
        if np.linalg.norm(left_knee - left_ankle) > 1e-6:
            # 计算脚部相对于小腿的角度
            ankle_angle = self.compute_angle(left_knee, left_ankle, left_ankle + np.array([0, 0, 0.1]))
            joint_angles[6] = ankle_angle - np.pi / 2

        # 右髋关节
        if np.linalg.norm(right_hip - right_knee) > 1e-6:
            hip_angle = self.compute_angle(right_shoulder, right_hip, right_knee)
            joint_angles[7] = hip_angle - np.pi / 2

        # 右膝关节
        if np.linalg.norm(right_knee - right_ankle) > 1e-6:
            knee_angle = self.compute_angle(right_hip, right_knee, right_ankle)
            joint_angles[8] = np.pi - knee_angle

        # 右踝关节
        if np.linalg.norm(right_knee - right_ankle) > 1e-6:
            ankle_angle = self.compute_angle(right_knee, right_ankle, right_ankle + np.array([0, 0, 0.1]))
            joint_angles[9] = ankle_angle - np.pi / 2

        # 注意：这里只计算了10个主要关节角度
        # 对于G1机器人的完整29个自由度，需要根据实际的URDF配置
        # 添加更多关节（如脊柱、颈部、手腕等）
        # 剩余的关节角度保持为0（可以根据需要扩展）

        return joint_angles

    def convert_motion_data(self, motion_data: np.ndarray, target_fps: float = 30.0, source_fps: float = None) -> dict:
        """
        转换完整的动作数据。

        Args:
            motion_data: 形状为 (num_frames, 33, 3) 的人体关键点数据
            target_fps: 目标帧率（用于重采样）
            source_fps: 源帧率（如果提供）

        Returns:
            包含转换后数据的字典
        """
        num_frames = motion_data.shape[0]
        joint_angles = []

        print(f"[INFO] 开始转换 {num_frames} 帧数据...")

        for i in range(num_frames):
            landmarks = motion_data[i]
            angles = self.compute_joint_angles_from_pose(landmarks)
            joint_angles.append(angles)

            if (i + 1) % 100 == 0:
                print(f"[INFO] 已转换 {i + 1}/{num_frames} 帧")

        joint_angles = np.array(joint_angles, dtype=np.float32)  # (num_frames, robot_dof)

        # 计算关节速度（通过差分）
        joint_velocities = np.zeros_like(joint_angles)
        if num_frames > 1:
            dt = 1.0 / target_fps if target_fps else 1.0 / 30.0
            joint_velocities[1:] = (joint_angles[1:] - joint_angles[:-1]) / dt

        print(f"[INFO] 转换完成: {joint_angles.shape}")

        return {
            "joint_positions": joint_angles,
            "joint_velocities": joint_velocities,
            "num_frames": num_frames,
            "robot_dof": self.robot_dof,
            "fps": target_fps,
        }


def convert_motion_file(input_path: str, output_path: str = None, robot_dof: int = 29, target_fps: float = 30.0):
    """
    转换动作数据文件。

    Args:
        input_path: 输入文件路径（.pkl或.npy）
        output_path: 输出文件路径
        robot_dof: 机器人自由度
        target_fps: 目标帧率
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 加载数据
    if input_path.suffix == ".pkl":
        with open(input_path, "rb") as f:
            data = pickle.load(f)
            motion_data = data["motion_data"]
            source_fps = data.get("fps", 30.0)
    elif input_path.suffix == ".npy":
        motion_data = np.load(input_path)
        source_fps = 30.0
    else:
        raise ValueError(f"不支持的文件格式: {input_path.suffix}")

    print(f"[INFO] 加载动作数据: {motion_data.shape}, 源帧率: {source_fps} FPS")

    # 转换数据
    converter = HumanToRobotConverter(robot_dof=robot_dof)
    converted_data = converter.convert_motion_data(motion_data, target_fps=target_fps, source_fps=source_fps)

    # 保存数据
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_robot.pkl"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(converted_data, f)

    print(f"[INFO] 转换后的数据已保存到: {output_path}")

    # 同时保存为numpy格式
    npy_path = output_path.with_suffix(".npy")
    np.save(npy_path, converted_data["joint_positions"])
    print(f"[INFO] 关节角度数据（numpy格式）已保存到: {npy_path}")

    return converted_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="将人体姿态数据转换为机器人关节角度")
    parser.add_argument("--input", type=str, required=True, help="输入动作数据文件路径（.pkl或.npy）")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径（可选）")
    parser.add_argument("--robot_dof", type=int, default=29, help="机器人自由度数量（默认: 29，G1机器人）")
    parser.add_argument("--target_fps", type=float, default=30.0, help="目标帧率（默认: 30.0）")

    args = parser.parse_args()

    convert_motion_file(args.input, args.output, args.robot_dof, args.target_fps)


if __name__ == "__main__":
    main()


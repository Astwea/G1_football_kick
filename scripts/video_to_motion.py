#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
从视频中提取人体姿态数据，用于AMP训练。

使用MediaPipe Pose从视频中提取33个人体关键点，并保存为时间序列数据。
"""

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

# 尝试导入MediaPipe，提供清晰的错误信息
try:
    import mediapipe as mp
except ImportError as e:
    print("[ERROR] 无法导入MediaPipe。请确保已正确安装：")
    print("  pip install mediapipe")
    print(f"  详细错误: {e}")
    sys.exit(1)

# 检查MediaPipe版本和导入
mp_pose = None
try:
    # 标准导入方式
    if hasattr(mp, 'solutions'):
        mp_pose = mp.solutions.pose
    else:
        raise AttributeError("mp.solutions不存在")
except AttributeError:
    # 如果标准方式失败，尝试其他导入方式
    print("[WARNING] 标准导入方式失败，尝试备用导入方式...")
    try:
        from mediapipe.python.solutions import pose as mp_pose
    except ImportError as e:
        print("[ERROR] MediaPipe导入失败。可能的原因：")
        print("  1. MediaPipe未正确安装，请运行: pip install --upgrade mediapipe")
        print("  2. MediaPipe版本过旧，请运行: pip install --upgrade mediapipe")
        print("  3. 项目中存在名为 'mediapipe.py' 的文件导致命名冲突")
        print("  4. Python版本不兼容（MediaPipe支持Python 3.8-3.11）")
        print(f"  详细错误: {e}")
        # 提供诊断信息
        if hasattr(mp, '__file__'):
            print(f"  MediaPipe模块位置: {mp.__file__}")
            # 检查是否是项目中的文件（命名冲突）
            if 'mediapipe.py' in str(mp.__file__):
                print("  [警告] 检测到可能的命名冲突：导入的是项目中的mediapipe.py文件")
        print(f"  MediaPipe模块属性: {[x for x in dir(mp) if not x.startswith('_')][:20]}")
        sys.exit(1)

if mp_pose is None:
    print("[ERROR] 无法找到MediaPipe Pose模块")
    sys.exit(1)


def extract_pose_from_video(video_path: str, output_path: str = None, min_detection_confidence: float = 0.5):
    """
    从视频中提取人体姿态数据。

    Args:
        video_path: 输入视频文件路径
        output_path: 输出文件路径（.pkl或.npy格式）
        min_detection_confidence: MediaPipe检测置信度阈值

    Returns:
        motion_data: 形状为 (num_frames, 33, 3) 的numpy数组，包含关键点坐标
    """
    # 初始化MediaPipe Pose
    pose = mp_pose.Pose(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=0.5,
        model_complexity=2,  # 使用完整模型（33个关键点）
    )

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] 视频信息: {total_frames} 帧, {fps:.2f} FPS")

    # 存储所有关键点数据
    all_landmarks = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 转换BGR到RGB（MediaPipe需要RGB格式）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # 检测姿态
        results = pose.process(rgb_frame)

        # 提取关键点
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            all_landmarks.append(landmarks)
        else:
            # 如果没有检测到姿态，使用零值填充
            print(f"[WARNING] 第 {frame_count} 帧未检测到姿态，使用零值填充")
            all_landmarks.append([[0.0, 0.0, 0.0] for _ in range(33)])

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"[INFO] 已处理 {frame_count}/{total_frames} 帧")

    cap.release()

    # 转换为numpy数组
    motion_data = np.array(all_landmarks, dtype=np.float32)  # (num_frames, 33, 3)

    print(f"[INFO] 成功提取 {motion_data.shape[0]} 帧的姿态数据")
    print(f"[INFO] 数据形状: {motion_data.shape}")

    # 保存数据
    if output_path is None:
        video_path_obj = Path(video_path)
        output_path = video_path_obj.parent / f"{video_path_obj.stem}_motion.pkl"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存为pickle文件（包含元数据）
    output_data = {
        "motion_data": motion_data,
        "fps": fps,
        "num_frames": motion_data.shape[0],
        "num_keypoints": 33,
        "keypoint_dim": 3,
    }

    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)

    print(f"[INFO] 动作数据已保存到: {output_path}")

    # 同时保存为numpy文件（便于直接加载）
    npy_path = output_path.with_suffix(".npy")
    np.save(npy_path, motion_data)
    print(f"[INFO] 动作数据（numpy格式）已保存到: {npy_path}")

    return motion_data, output_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="从视频中提取人体姿态数据")
    parser.add_argument("--video", type=str, required=True, help="输入视频文件路径")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径（可选）")
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.5,
        help="MediaPipe检测置信度阈值（默认: 0.5）",
    )

    args = parser.parse_args()

    # 检查视频文件是否存在
    if not Path(args.video).exists():
        raise FileNotFoundError(f"视频文件不存在: {args.video}")

    # 提取姿态数据
    extract_pose_from_video(args.video, args.output, args.min_detection_confidence)


if __name__ == "__main__":
    main()


# 宇树G1踢球AMP训练系统使用指南

本文档说明如何使用基于宇树G1机器人的踢球动作模仿学习与强化学习系统。

## 目录

1. [系统概述](#系统概述)
2. [环境准备](#环境准备)
3. [视频处理](#视频处理)
4. [动作数据转换](#动作数据转换)
5. [机器人配置](#机器人配置)
6. [训练配置](#训练配置)
7. [开始训练](#开始训练)
8. [测试和评估](#测试和评估)
9. [常见问题](#常见问题)

## 系统概述

本系统实现了以下功能：

1. **视频到动作数据转换**：使用MediaPipe从视频中提取人体姿态数据
2. **动作映射**：将人体姿态映射到G1机器人关节空间
3. **踢球环境**：在Isaac Lab中创建包含G1机器人和球体的仿真环境
4. **AMP训练**：使用Adversarial Motion Priors进行模仿学习和强化学习

## 环境准备

### 1. 安装依赖

确保已安装Isaac Lab和必要的Python包：

```bash
# 安装Isaac Lab（参考Isaac Lab官方文档）
# ...

# 重要：Isaac Lab需要NumPy < 2.0
# 如果当前环境使用NumPy 2.x，需要先降级
pip install "numpy<2.0"

# 安装Python依赖
pip install -r requirements.txt
# 或者手动安装：
# pip install "numpy<2.0" mediapipe opencv-python scipy
```

**注意**：Isaac Lab及其依赖（如`nlopt`）是用NumPy 1.x编译的，与NumPy 2.x不兼容。如果遇到 `AttributeError: _ARRAY_API not found` 错误，请确保使用NumPy 1.x版本。

### 2. 安装项目

```bash
# 在项目根目录下
cd /home/astwea/amp
python -m pip install -e source/amp
```

### 3. G1机器人模型

**重要**：系统会自动尝试使用Isaac Lab内置的G1机器人配置。

- 如果Isaac Lab已包含G1配置（通过`isaaclab_assets`），系统会自动使用
- 如果找不到内置配置，系统会使用备用配置，此时需要：
  - 手动指定G1的USD文件路径
  - 或从宇树官方获取G1的URDF/USD文件
  - 在 `source/amp/amp/tasks/direct/g1_kick/g1_robot_cfg.py` 中配置USD路径

## 视频处理

### 1. 准备视频文件

准备包含踢球动作的视频文件，建议：
- 视频清晰，动作完整
- 人体姿态可见
- 建议使用MP4格式

### 2. 提取人体姿态

使用 `video_to_motion.py` 从视频中提取人体关键点：

```bash
python scripts/video_to_motion.py --video /path/to/kick_video.mp4 --output /path/to/output_motion.pkl
```

参数说明：
- `--video`: 输入视频文件路径
- `--output`: 输出文件路径（可选，默认为视频同目录下的 `*_motion.pkl`）
- `--min_detection_confidence`: MediaPipe检测置信度阈值（默认0.5）

输出文件：
- `*_motion.pkl`: 包含完整元数据的pickle文件
- `*_motion.npy`: numpy格式的关键点数据

### 3. 检查提取结果

提取的数据包含33个人体关键点的3D坐标，形状为 `(num_frames, 33, 3)`。

## 动作数据转换

### 1. 转换为机器人关节角度

使用 `motion_converter.py` 将人体姿态转换为G1机器人关节角度：

```bash
python scripts/motion_converter.py --input /path/to/output_motion.pkl --output /path/to/robot_motion.pkl --robot_dof 29
```

参数说明：
- `--input`: 输入动作数据文件（.pkl或.npy）
- `--output`: 输出文件路径（可选）
- `--robot_dof`: 机器人自由度数量（G1为29）
- `--target_fps`: 目标帧率（默认30.0）

输出文件：
- `*_robot.pkl`: 包含关节角度和速度的pickle文件
- `*_robot.npy`: numpy格式的关节角度数据

### 2. 准备AMP训练数据

使用 `load_motion_data.py` 准备AMP训练所需的数据格式：

```bash
python scripts/load_motion_data.py --input /path/to/robot_motion.pkl --output /path/to/amp_motion.npy --observation_dim 77
```

参数说明：
- `--input`: 输入动作数据文件
- `--output`: 输出文件路径（可选）
- `--observation_dim`: 观察空间维度（如果已知）

**注意**：观察空间维度需要与环境配置匹配。对于G1踢球环境，观察空间包括：
- 关节位置（29维）
- 关节速度（29维）
- 根位置（3维）
- 根旋转（4维，四元数）
- 根速度（6维）
- 球相对位置（3维）
- 球速度（3维）
- 总计：约77维（29+29+3+4+6+3+3，具体取决于环境配置）

## 机器人配置

### 1. G1机器人配置

系统会自动尝试从`isaaclab_assets`导入G1配置。如果成功，无需额外配置。

如果自动导入失败，可以：

**方法1：检查isaaclab_assets安装**
```bash
# 确保isaaclab_assets已正确安装
pip install isaaclab-assets
```

**方法2：手动指定USD文件路径**

编辑 `source/amp/amp/tasks/direct/g1_kick/g1_robot_cfg.py`，在备用配置中添加：

```python
@configclass
class G1RobotCfg(ArticulationCfg):
    usd_path = "/path/to/g1.usd"  # 指定G1的USD文件路径
    # ... 其他配置
```

### 2. 调整关节限制（可选）

如果需要自定义G1机器人的关节参数，可以在环境配置中调整关节范围、阻尼、刚度等参数。

## 训练配置

### 1. 环境配置

环境配置在 `source/amp/amp/tasks/direct/g1_kick/g1_kick_env_cfg.py` 中定义。

可以调整的参数：
- `num_envs`: 并行环境数量（默认4096）
- `episode_length_s`: 每回合时长（默认10秒）
- 奖励权重：各种奖励的缩放因子

### 2. AMP训练配置

AMP训练配置在 `source/amp/amp/tasks/direct/g1_kick/agents/skrl_amp_cfg.yaml` 中定义。

关键参数：
- `task_reward_weight`: 任务奖励权重（踢球，默认0.3）
- `style_reward_weight`: 风格奖励权重（动作模仿，默认0.7）
- `trainer.timesteps`: 训练时间步数（默认1000000）

## 开始训练

### 1. 基本训练命令

```bash
python scripts/skrl/train.py \
    --task G1-Kick-Direct-v0 \
    --algorithm AMP \
    --motion_file /path/to/amp_motion.npy \
    --num_envs 4096 \
    --max_iterations 10000
```

### 2. 完整训练命令示例

```bash
python scripts/skrl/train.py \
    --task G1-Kick-Direct-v0 \
    --algorithm AMP \
    --motion_file /path/to/amp_motion.npy \
    --num_envs 4096 \
    --device cuda:0 \
    --seed 42 \
    --max_iterations 10000 \
    --video \
    --video_length 500 \
    --video_interval 5000
```

参数说明：
- `--task`: 任务名称（G1-Kick-Direct-v0）
- `--algorithm`: 算法（AMP）
- `--motion_file`: 参考动作数据文件路径
- `--num_envs`: 并行环境数量
- `--device`: 设备（cuda:0 或 cpu）
- `--seed`: 随机种子
- `--max_iterations`: 最大训练迭代次数
- `--video`: 启用视频录制
- `--video_length`: 视频长度（帧数）
- `--video_interval`: 视频录制间隔（步数）

### 3. 训练输出

训练日志和模型检查点保存在：
```
logs/skrl/g1_kick_amp_run/<timestamp>_amp_torch/
├── params/
│   ├── env.yaml
│   └── agent.yaml
├── checkpoints/
│   └── *.pt
└── videos/
    └── train/
        └── *.mp4
```

## 测试和评估

### 1. 使用随机动作测试环境

```bash
python scripts/random_agent.py --task G1-Kick-Direct-v0 --num_envs 1
```

### 2. 使用零动作测试环境

```bash
python scripts/zero_agent.py --task G1-Kick-Direct-v0 --num_envs 1
```

### 3. 加载训练好的模型进行测试

```bash
python scripts/skrl/play.py \
    --task G1-Kick-Direct-v0 \
    --checkpoint /path/to/checkpoint.pt \
    --num_envs 1
```

## 常见问题

### 1. MediaPipe无法检测到姿态

- 检查视频质量，确保人体清晰可见
- 降低 `--min_detection_confidence` 阈值
- 尝试使用不同的视频角度

### 2. 动作映射不准确

- 检查 `motion_converter.py` 中的关节映射关系
- 根据G1机器人的实际关节结构调整映射
- 可能需要使用逆运动学（IK）进行更精确的转换

### 3. 训练不收敛

- 检查奖励权重配置，平衡任务奖励和风格奖励
- 增加训练时间步数
- 调整学习率
- 检查参考动作数据质量

### 4. 机器人无法保持平衡

- 调整奖励函数中的平衡奖励权重
- 检查机器人初始姿态配置
- 调整关节PD控制参数

### 5. 球体不出现或无法交互

- 检查球体配置中的prim_path是否正确
- 确认碰撞检测已启用
- 检查球体的物理属性（质量、摩擦等）

### 6. URDF文件加载失败

- 确认URDF文件路径正确
- 检查URDF文件格式是否正确
- 如果Isaac Lab不支持直接加载URDF，需要先转换为USD格式

### 7. NumPy版本兼容性错误

如果遇到以下错误：
```
AttributeError: _ARRAY_API not found
```
或
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

**解决方法**：
```bash
# 降级NumPy到1.x版本
pip install "numpy<2.0"

# 验证版本
python -c "import numpy; print(numpy.__version__)"
# 应该显示类似 1.26.x 的版本号
```

**原因**：Isaac Lab及其依赖（如`nlopt`、`dex_retargeting`）是用NumPy 1.x编译的，与NumPy 2.x不兼容。

## 下一步

1. **优化动作映射**：根据实际G1机器人结构优化人体到机器人的映射
2. **调整奖励函数**：根据训练效果调整各种奖励的权重
3. **增加数据多样性**：使用多个视频提取更多样化的动作数据
4. **超参数调优**：根据训练效果调整AMP算法的超参数
5. **部署到真实机器人**：将训练好的模型部署到真实的G1机器人上

## 参考资源

- [Isaac Lab文档](https://isaac-sim.github.io/IsaacLab/)
- [skrl文档](https://skrl.readthedocs.io/)
- [MediaPipe Pose文档](https://google.github.io/mediapipe/solutions/pose)
- [宇树G1机器人文档](https://www.unitree.com/)

## 技术支持

如遇到问题，请：
1. 检查本文档的常见问题部分
2. 查看训练日志中的错误信息
3. 参考Isaac Lab和skrl的官方文档
4. 在项目GitHub仓库提交Issue


# PI0 测试代码使用说明

## 概述

`test_pi0_ros.py` 是用于测试 PI0 模型的脚本，支持多种数据源模式：
- **Bag 模式**: 从 rosbag 文件离线读取数据（推荐测试用）
- **ZMQ 模式**: 通过 ZMQ 桥接实时读取 ROS 数据（推荐实时部署）
- **ROS 模式**: 直接从 ROS topics 读取（有 Python 版本兼容问题）
- **Dummy 模式**: 使用随机图像测试模型加载

> ⚠️ **重要**: 由于 openpi 使用 Python 3.11，而 ROS Noetic 使用 Python 3.8，直接使用 `ros` 模式会卡住。**推荐使用 `zmq` 模式进行实时测试**。

---

## 数据格式

### State/Action 格式 (14维)

```
[left_ee_x, left_ee_y, left_ee_z, left_ee_roll, left_ee_pitch, left_ee_yaw, left_gripper,
 right_ee_x, right_ee_y, right_ee_z, right_ee_roll, right_ee_pitch, right_ee_yaw, right_gripper]
```

| 维度 | 名称 | 单位 | 说明 |
|------|------|------|------|
| 0-2 | left xyz | 米 (m) | 左臂末端位置 |
| 3-5 | left euler | 弧度 (rad) | 左臂末端姿态 (roll, pitch, yaw) |
| 6 | left gripper | 0-1 | 左夹爪开合度 (0=闭合, 1=张开) |
| 7-9 | right xyz | 米 (m) | 右臂末端位置 |
| 10-12 | right euler | 弧度 (rad) | 右臂末端姿态 |
| 13 | right gripper | 0-1 | 右夹爪开合度 |

---

## 快速开始

### 方式一：Bag 文件模式（离线测试推荐）

从录制的 rosbag 文件读取数据，**不需要 ROS 运行**：

```bash
cd /home/pine/yzj/src

# 运行测试
./run_test_pi0.sh bag /home/pine/yzj/pour/r1_data_20260202_172011.bag
```

### 方式二：ZMQ 桥接模式（实时测试推荐）⭐

解决 Python 版本兼容问题，使用两个进程：
- `ros_bridge.py`: 使用系统 Python 3.8 读取 ROS 数据
- `test_pi0_ros.py`: 使用 openpi Python 3.11 运行 PI0 模型

#### 步骤 1: 启动 ROS 环境

```bash
# ===== 终端1: roscore =====
export ROS_MASTER_URI=http://192.168.123.15:11311
export ROS_IP=192.168.123.15
roscore

# ===== 终端2: 手腕相机 =====
roslaunch hands_d405 hands_d405.launch

# ===== 终端3: 头部相机 (15Hz) =====
roslaunch record_15hz.launch
```

#### 步骤 2: 启动 ROS 桥接

```bash
# ===== 终端4: ROS 桥接（使用系统 Python）=====
export ROS_MASTER_URI=http://192.168.123.15:11311
export ROS_IP=192.168.123.15
/usr/bin/python3 /home/pine/yzj/src/ros_bridge.py
```

#### 步骤 3: 运行 PI0 测试

```bash
# ===== 终端5: PI0 测试（使用 openpi 虚拟环境）=====
cd /home/pine/yzj/src

# 仅打印 action 结果
./run_test_pi0.sh zmq

# 打印 action + 计算 IK（输出关节角度）
./run_test_pi0.sh zmq_ik
```

### 方式三：Dummy 模式（模型加载测试）

```bash
cd /home/pine/yzj/src
./run_test_pi0.sh dummy
```

---

## 详细用法

### 使用 run_test_pi0.sh 脚本

```bash
./run_test_pi0.sh <mode> [bag_file]

# mode 可选值:
#   dummy   - 使用随机图像测试模型加载
#   bag     - 从 rosbag 文件读取（需要 bag_file 参数）
#   ros     - 从 ROS topics 直接读取（⚠️ 有兼容性问题）
#   ros_ik  - 同上 + 计算 IK
#   zmq     - 通过 ZMQ 桥接读取 ROS 数据（✅ 推荐）
#   zmq_ik  - 同上 + 计算 IK（✅ 推荐）
```

### 直接使用 Python 脚本

```bash
# 激活虚拟环境
cd /home/pine/yzj/RoboTwin/policy/pi0
source .venv/bin/activate

# Bag 文件模式
python /home/pine/yzj/src/test_pi0_ros.py \
    --train_config_name R1_FFT_pour_35_0130_5k \
    --checkpoint_path /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/10000 \
    --task_prompt "pour water" \
    --bag_file /home/pine/yzj/pour/r1_data_20260202_172011.bag \
    --n_iterations 20

# ZMQ 模式 + IK 计算（推荐）
python /home/pine/yzj/src/test_pi0_ros.py \
    --train_config_name R1_FFT_pour_35_0130_5k \
    --checkpoint_path /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/10000 \
    --task_prompt "pour water" \
    --zmq_mode \
    --compute_ik
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_config_name` | `R1_FFT_pour_35_0130_5k` | 训练配置名称 |
| `--checkpoint_path` | `.../checkpoint/10000` | 模型 checkpoint 路径 |
| `--pi0_step` | `10` | 每次推理返回的动作数量 |
| `--task_prompt` | `"pour water"` | 任务语言指令 |
| `--n_iterations` | `10` | 测试迭代次数 |
| `--ros_mode` | `False` | 使用 ROS topics 直接读取 |
| `--zmq_mode` | `False` | 使用 ZMQ 桥接读取（推荐）|
| `--zmq_host` | `localhost` | ZMQ 服务器地址 |
| `--zmq_port` | `5555` | ZMQ 端口 |
| `--bag_file` | `None` | Rosbag 文件路径 |
| `--dummy_mode` | `False` | 使用随机图像测试 |
| `--print_only` | `True` | 仅打印结果 |
| `--compute_ik` | `False` | 计算 IK 输出关节角度 |

---

## ROS Topics

### 需要订阅的 Topics

| Topic | 类型 | 说明 |
|-------|------|------|
| `/hdas/camera_head/rgb/image_rect_color/compressed` | CompressedImage | 头部 RGB |
| `/left/camera/color/image_raw/compressed` | CompressedImage | 左手腕 RGB |
| `/right/camera/color/image_raw/compressed` | CompressedImage | 右手腕 RGB |
| `/hdas/feedback_arm_left_low` | JointState | 左臂关节反馈 |
| `/hdas/feedback_arm_right_low` | JointState | 右臂关节反馈 |
| `/hdas/feedback_gripper_left_low` | JointState | 左夹爪反馈 |
| `/hdas/feedback_gripper_right_low` | JointState | 右夹爪反馈 |

---

## 输出示例

```
============================================================
Frame 1/10
============================================================

[Controller] State vector (14 dims, format: xyz+euler+gripper):
    Left:  pos=[0.5319 0.1605 1.1556], euler=[-0.4387  0.2591 -1.0832], gripper=0.9786
    Right: pos=[ 0.561  -0.1636  1.1246], euler=[0.4878 0.1542 1.2129], gripper=0.9710

[Controller] Received 10 actions from model:
  Action 0:
    Left:  pos=[0.5351 0.1618 1.1634], euler=[-0.4419  0.2169 -1.0968], gripper=0.7528
    Right: pos=[ 0.5626 -0.1688  1.1251], euler=[0.4595 0.1772 1.2221], gripper=0.9385
  Action 1:
    Left:  pos=[0.5387 0.1613 1.1591], euler=[-0.4372  0.215  -1.0838], gripper=0.7354
    Right: pos=[ 0.5652 -0.1701  1.1239], euler=[0.4616 0.1762 1.2143], gripper=0.9410
  ...
```

---

## 配置信息

| 配置项 | 值 |
|--------|-----|
| 训练配置 | `R1_FFT_pour_35_0130_5k` |
| Checkpoint | `/home/pine/yzj/RoboTwin/policy/pi0/checkpoint/10000` |
| URDF | `/home/pine/yzj/R1_urdf/galaxea_sim/assets/r1/robot.urdf` |
| Torso 固定 | `[0.25, -0.4, -0.85, 0]` |

---

## 故障排除

### 1. ROS 模式卡住 (rospy.init_node 挂起)

**原因**: openpi 使用 Python 3.11，而 ROS Noetic 需要 Python 3.8，两者不兼容。

**解决方案**: 使用 ZMQ 桥接模式：
```bash
# 终端1: 使用系统 Python 运行 ROS 桥接
/usr/bin/python3 /home/pine/yzj/src/ros_bridge.py

# 终端2: 使用 openpi 运行 PI0 测试
./run_test_pi0.sh zmq_ik
```

### 2. 模型加载 OOM (Out of Memory)

脚本已包含内存优化设置，如仍有问题可手动设置：
```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export XLA_FLAGS="--xla_gpu_memory_fraction=0.7"
```

### 3. rosbag 模块找不到

```bash
cd /home/pine/yzj/RoboTwin/policy/pi0
source .venv/bin/activate
pip install pycryptodome python-gnupg rospkg
```

### 4. kinpy 找不到

```bash
pip install kinpy
```

### 5. ROS 连接失败

确保设置了正确的环境变量：
```bash
export ROS_MASTER_URI=http://192.168.123.15:11311
export ROS_IP=192.168.123.15
```

### 6. ZMQ 连接失败

确保两边都安装了 pyzmq：
```bash
# 系统 Python
pip3 install pyzmq

# openpi 虚拟环境
cd /home/pine/yzj/RoboTwin/policy/pi0
source .venv/bin/activate
pip install pyzmq
```

---

## 文件列表

```
/home/pine/yzj/src/
├── test_pi0_ros.py       # 主测试脚本
├── run_test_pi0.sh       # 运行脚本
├── ros_bridge.py         # ROS-ZMQ 桥接（系统Python）
├── h52eepose.py          # FK 计算参考
├── bag2h5x_yzj_speed.py  # Rosbag 转 HDF5
└── README_PI0_TEST.md    # 本文档
```

# PI0 测试代码 - ROS 模式使用指南

## 概述

现在 `test_pi0_with_ik.py` 支持从 ROS topics 读取数据，这些 topics 与数据收集时使用的完全一致。

## ROS Topics 订阅列表

### 相机 Topics
| Topic | 消息类型 | 说明 |
|-------|---------|------|
| `/hdas/camera_head/rgb/image_rect_color/compressed` | CompressedImage | 头部RGB相机（640x480） |
| `/hdas/camera_head/depth/depth_registered` | Image | 头部深度相机 |
| `/left/camera/color/image_raw/compressed` | CompressedImage | 左腕RGB相机 |
| `/left/camera/depth/image_rect_raw` | Image | 左腕深度相机 |
| `/right/camera/color/image_raw/compressed` | CompressedImage | 右腕RGB相机 |
| `/right/camera/depth/image_rect_raw` | Image | 右腕深度相机 |

### 机器人状态 Topics
| Topic | 消息类型 | 说明 |
|-------|---------|------|
| `/hdas/feedback_arm_left` | Float64MultiArray | 左臂关节位置反馈 |
| `/hdas/feedback_arm_right` | Float64MultiArray | 右臂关节位置反馈 |
| `/hdas/feedback_gripper_left` | Float64MultiArray | 左夹爪位置反馈 |
| `/hdas/feedback_gripper_right` | Float64MultiArray | 右夹爪位置反馈 |

### GT Action Topics（仅用于数据收集，测试时不需要）
- `/motion_target/target_joint_state_arm_left`
- `/motion_target/target_joint_state_arm_right`

## 使用方法

### 前置条件

1. **ROS 环境**
   ```bash
   # 确保 ROS 已安装并 source
   source /opt/ros/noetic/setup.bash  # 或你的ROS版本
   ```

2. **Python 依赖**
   ```bash
   pip install rospkg
   pip install opencv-python
   ```

3. **确认 Topics 正在发布**
   ```bash
   # 查看可用的 topics
   rostopic list
   
   # 检查特定 topic 是否有数据
   rostopic echo /hdas/camera_head/rgb/image_rect_color/compressed --noarr
   rostopic hz /hdas/feedback_arm_left
   ```

### 运行 ROS 模式

#### 方法 1: 使用脚本

编辑 `run_pi0_test.sh`，确保 ROS 模式已启用（默认）：

```bash
chmod +x run_pi0_test.sh
./run_pi0_test.sh
```

#### 方法 2: 直接命令行

```bash
# 左臂控制
python test_pi0_with_ik.py \
    --train_config_name your_config \
    --model_name your_model \
    --task_prompt "Pick up the apple" \
    --side left \
    --ros_mode

# 右臂控制
python test_pi0_with_ik.py \
    --train_config_name your_config \
    --model_name your_model \
    --task_prompt "Place the cup" \
    --side right \
    --ros_mode
```

## 数据流程

```
ROS Topics (相机 + 机器人状态)
         ↓
ROSDataSubscriber (订阅并缓存)
         ↓
get_observation() (返回最新图像)
         ↓
PI0TestController (模型推理)
         ↓
IK Solver (末端位置 → 关节角度)
         ↓
execute_joint_action() (发送到机器人)
```

## ROSDataSubscriber 类说明

### 功能
- **自动订阅** 所有必需的 ROS topics
- **线程安全** 数据缓存（使用 threading.Lock）
- **图像自动处理**：
  - Compressed 图像自动解码
  - 所有图像统一 resize 到 640x480
  - 深度图使用最近邻插值（保持精度）
  - RGB 图使用线性插值
- **等待机制** 确保初始数据到达后才开始

### 主要方法

#### `get_observation()`
返回当前观察数据：
```python
main_image, wrist_image = ros_subscriber.get_observation()
# main_image: 头部 RGB (PIL Image, 640x480)
# wrist_image: 腕部 RGB (PIL Image, 640x480, 根据 side 选择)
```

#### `get_current_joint_state()`
返回当前关节状态（可用于验证或记录）：
```python
arm_pos, gripper_pos = ros_subscriber.get_current_joint_state()
# arm_pos: numpy array, shape (6,) - 6个手臂关节角度
# gripper_pos: float - 夹爪位置
```

## 三种运行模式对比

| 模式 | 参数 | 数据源 | 适用场景 |
|------|------|--------|---------|
| **Dummy** | `--dummy_mode` | 虚拟图像 | 代码测试、调试 |
| **ROS** | `--ros_mode` | ROS topics | 真实机器人（推荐） |
| **RealSense** | 无特殊参数 | RealSense相机 | 使用 RealSense D435i/L515 |

## 配置参数

### 必需参数
```bash
--train_config_name    # 训练配置名称
--model_name          # 模型名称
```

### 可选参数
```bash
--checkpoint_id       # 检查点ID (默认: latest)
--pi0_step           # 预测步数 (默认: 10)
--task_prompt        # 任务提示文本
--n_iterations       # 测试迭代次数 (默认: 50)
--chunk_size         # 每次执行的动作数 (默认: 10)
--side               # 控制哪只手臂 (left/right, 默认: left)
--use_ik             # 使用IK转换 (默认: True)
```

## 初始位置设置

代码已经设置了初始关节角度和夹爪值：

### 左手
```python
关节角度: [-0.1674, 2.0109, -0.6594, 2.0021, 0.3938, -1.7194]
夹爪: 1.0 (开启状态)
```

### 右手
```python
关节角度: [0.1923, 1.8926, -0.6874, -1.6057, -0.1015, 1.3085]
夹爪: 1.0 (开启状态)
```

## Torso 固定值

与数据收集保持一致：
```python
TORSO_FIXED = [0.25, -0.4, -0.85, 0]
```

## 故障排查

### 问题 1: ROS not available
```
[Warning] ROS not available. Install rospy and cv_bridge for ROS mode.
```
**解决**：
```bash
# 确认 ROS 环境
source /opt/ros/noetic/setup.bash
# 安装依赖
pip install rospkg
sudo apt-get install ros-noetic-cv-bridge  # 或你的ROS版本
```

### 问题 2: Timeout waiting for ROS data
```
RuntimeError: Timeout waiting for ROS data
```
**解决**：
1. 检查 topics 是否在发布：
   ```bash
   rostopic list
   rostopic hz /hdas/camera_head/rgb/image_rect_color/compressed
   ```
2. 确认相机和机器人节点正在运行
3. 检查网络连接（如果使用远程 ROS master）

### 问题 3: 图像解码错误
```
[ROSSubscriber] Error decoding head RGB: ...
```
**解决**：
- 检查 compressed image 格式是否正确
- 确认 cv2 版本兼容：`pip install opencv-python --upgrade`

### 问题 4: 无法获取关节状态
```
arm_pos is None
```
**解决**：
- 检查 `/hdas/feedback_arm_left` 或 `right` 是否在发布
- 确认消息类型为 `Float64MultiArray`

## 性能优化建议

1. **降低图像频率**（如果推理速度慢）
   - 在 callback 中添加采样逻辑
   
2. **调整缓冲区大小**
   ```python
   rospy.Subscriber(..., queue_size=1)  # 只保留最新消息
   ```

3. **使用多线程**
   - ROSDataSubscriber 已经是线程安全的
   - 可以在主循环中并行处理图像

## 与数据收集的一致性

✅ 使用相同的 ROS topics  
✅ 相同的图像分辨率 (640x480)  
✅ 相同的图像处理方式（compressed 解码 + resize）  
✅ 相同的 Torso 固定值  
✅ 相同的关节状态格式

这确保了测试环境与训练数据收集环境完全一致！

## 示例：完整运行流程

```bash
# 1. 启动 ROS 核心（如果还没运行）
roscore &

# 2. 启动机器人和相机节点（你的具体启动命令）
# roslaunch your_robot robot.launch

# 3. 检查 topics
rostopic list | grep -E "(camera|feedback)"

# 4. 运行 PI0 测试（ROS 模式）
python test_pi0_with_ik.py \
    --train_config_name my_config \
    --model_name my_model \
    --task_prompt "Pick up the red cube" \
    --side left \
    --n_iterations 50 \
    --ros_mode

# 5. 观察输出
# [ROSSubscriber] Initializing ROS node...
# [ROSSubscriber] Subscribing to camera topics...
# [ROSSubscriber] Subscribing to robot state topics...
# [ROSSubscriber] Waiting for initial data...
# [ROSSubscriber] Ready!
# [Controller] Starting test loop...
```

## 高级用法

### 记录运行数据

可以在 ROS 模式下同时记录输入输出：

```bash
# 开始记录
rosbag record -a -O test_run.bag &

# 运行测试
python test_pi0_with_ik.py --ros_mode ...

# 停止记录
# Ctrl+C
```

### 可视化

使用 rviz 可视化相机和机器人状态：

```bash
rviz &
# 添加 Image 显示相机画面
# 添加 RobotModel 显示机器人姿态
```

---

**版本**: 2.0  
**更新日期**: 2026-02-01  
**新增**: ROS 模式支持

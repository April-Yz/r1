# PI0 测试代码使用说明

## 概述

`test_pi0_with_ik.py` 是一个集成了 PI0 模型推理和逆运动学（IK）计算的测试代码，用于将模型输出的末端位置转换为关节角度并控制机器人。

## 主要特性

1. **集成 PI0 模型**：使用 RoboTwin 的 PI0 策略进行推理
2. **逆运动学（IK）计算**：将末端位置（x, y, z, 姿态）转换为关节角度
3. **正运动学（FK）验证**：基于 `h52eepose.py` 的运动学链
4. **固定 Torso**：torso 关节固定为 `[0.25, -0.4, -0.85, 0]`（与 FK 保持一致）
5. **初始位置为 0**：机器人初始关节角度全部设为 0
6. **支持虚拟模式**：可以在没有真实硬件的情况下测试代码逻辑

## 代码结构

### 1. IKSolver 类

负责逆运动学求解：

- **正运动学（FK）**：`forward_kinematics()` - 从关节角度计算末端位置
- **逆运动学（IK）**：`inverse_kinematics()` - 从末端位置计算关节角度
- **URDF 配置**：使用 `R1_urdf/galaxea_sim/assets/r1/robot.urdf`
- **固定 Torso**：`TORSO_FIXED = [0.25, -0.4, -0.85, 0]`

### 2. PI0TestController 类

主控制器，集成模型推理和 IK：

- **模型初始化**：加载 PI0 模型
- **观察准备**：将图像和状态转换为模型输入格式
- **动作获取**：从模型获取末端位置动作
- **IK 转换**：将末端位置转换为关节角度
- **动作执行**：发送关节角度命令到机器人

### 3. 主循环

`run_test_loop()` 方法实现测试循环：

```
初始化（关节角度全部为 0）
  ↓
获取观察（相机图像）
  ↓
计算当前末端位置（FK）
  ↓
准备模型输入
  ↓
模型推理 → 末端位置动作
  ↓
IK 计算 → 关节角度
  ↓
执行关节角度命令
  ↓
重复
```

## 使用方法

### 基本用法

```bash
python test_pi0_with_ik.py \
    --train_config_name your_train_config \
    --model_name your_model \
    --checkpoint_id latest \
    --pi0_step 10 \
    --task_prompt "Pick up the object" \
    --n_iterations 50 \
    --chunk_size 10 \
    --side left \
    --use_ik
```

### 参数说明

- `--train_config_name`: PI0 训练配置名称（必需）
- `--model_name`: 模型名称（必需）
- `--checkpoint_id`: 检查点 ID（默认：latest）
- `--pi0_step`: 预测步数（默认：10）
- `--task_prompt`: 任务提示文本（默认：Pick up the object）
- `--n_iterations`: 测试迭代次数（默认：50）
- `--chunk_size`: 每次执行的动作数量（默认：10）
- `--use_ik`: 使用 IK 转换（默认：True）
- `--side`: 控制左臂或右臂（默认：left，可选：right）
- `--dummy_mode`: 虚拟模式，不需要真实硬件

### 虚拟模式测试

在没有真实硬件的情况下测试代码逻辑：

```bash
python test_pi0_with_ik.py \
    --train_config_name test_config \
    --model_name test_model \
    --task_prompt "Test task" \
    --n_iterations 10 \
    --dummy_mode
```

### 真实模式运行

需要先集成真实的相机和控制器：

```bash
# 确保相机已连接
python test_pi0_with_ik.py \
    --train_config_name your_config \
    --model_name your_model \
    --task_prompt "Pick up the apple" \
    --n_iterations 50
```

### 使用脚本运行

```bash
chmod +x run_pi0_test.sh
./run_pi0_test.sh
```

## 集成到真实系统

### 1. 集成相机

修改 `main()` 函数中的相机部分：

```python
from pyrealsense_image import initialize_camera, get_L515_image, get_D435_image

pipelines, align_objects = initialize_camera()

def get_real_observation():
    main_image = get_L515_image(pipelines)
    wrist_image = get_D435_image(pipelines)
    return main_image, wrist_image

get_observation_fn = get_real_observation
```

### 2. 集成机器人控制器

修改 `PI0TestController.__init__()` 传入控制器：

```python
from controller_eef import A1ArmController

controller = A1ArmController()
pi0_controller = PI0TestController(
    model_config=model_config,
    controller=controller,  # 传入真实控制器
    side='left'
)
```

并实现 `execute_joint_action()` 方法中的控制器调用：

```python
def execute_joint_action(self, joint_angles, gripper):
    if self.controller is not None:
        # 构建完整命令
        full_command = np.concatenate([
            self.torso_fixed,    # [0.25, -0.4, -0.85, 0]
            joint_angles,        # 6个手臂关节
            [gripper]            # 夹爪
        ])
        
        # 调用控制器（根据你的接口调整）
        self.controller.execute_joints(full_command, self.side)
```

## 动作格式说明

模型输出的动作应该是以下格式之一：

### 格式 1：位置 + 欧拉角
```python
action = [x, y, z, roll, pitch, yaw, gripper]  # 7维
```

### 格式 2：位置 + 四元数
```python
action = [x, y, z, qw, qx, qy, qz, gripper]  # 8维
```

代码会自动识别并转换为关节角度。

## 运动学配置

### URDF 路径
```python
URDF_PATH = "/home/pine/yzj/R1_urdf/galaxea_sim/assets/r1/robot.urdf"
```

### Torso 固定值（与 h52eepose.py 一致）
```python
TORSO_FIXED = [0.25, -0.4, -0.85, 0]
```

### 关节链配置
- **Left Arm**: `base_link` → `left_gripper_link`
- **Right Arm**: `base_link` → `right_gripper_link`
- **关节顺序**: torso(4) + arm(6) = 10个关节

## IK 求解器说明

使用 `scipy.optimize.minimize` 进行优化求解：

- **方法**：L-BFGS-B（有界优化）
- **目标**：最小化末端位置误差
- **约束**：关节角度限制 [-π, π]
- **初始猜测**：上一时刻的关节角度（提高收敛速度）

## 注意事项

1. **URDF 文件**：确保 URDF 路径正确，否则无法构建运动学链
2. **依赖安装**：需要安装 `kinpy`, `scipy`, `numpy`, `torch`, `PIL`
3. **GPU 内存**：PI0 模型可能需要较大 GPU 内存
4. **IK 收敛**：如果目标位置不可达，IK 可能不收敛，会打印警告
5. **关节限制**：当前使用默认限制 [-π, π]，可根据实际机器人调整
6. **Torso 固定**：确保与训练数据的 torso 值一致

## 故障排查

### 问题 1：URDF 文件未找到
```
FileNotFoundError: URDF file not found at ...
```
**解决**：检查 `URDF_PATH` 是否正确

### 问题 2：IK 不收敛
```
Warning: IK did not converge well. Error: 0.xxxx
```
**解决**：
- 检查目标位置是否在工作空间内
- 调整 IK 参数（`max_iterations`, `tolerance`）
- 使用更好的初始猜测

### 问题 3：模型加载失败
```
Error loading PI0 model...
```
**解决**：
- 检查模型路径和配置
- 确认 `train_config_name` 和 `model_name` 正确

### 问题 4：相机初始化失败
**解决**：
- 确认相机已连接
- 检查 `pyrealsense_image` 模块

## 扩展功能

### 1. 添加关节速度控制

在 `execute_joint_action()` 中添加速度：

```python
def execute_joint_action(self, joint_angles, gripper, velocities=None):
    if velocities is None:
        velocities = np.zeros(6)
    # 发送位置和速度
```

### 2. 添加轨迹平滑

使用插值平滑关节轨迹：

```python
from scipy.interpolate import interp1d

def smooth_trajectory(self, start, end, num_steps=10):
    t = np.linspace(0, 1, num_steps)
    trajectory = interp1d([0, 1], np.vstack([start, end]), axis=0, kind='cubic')
    return trajectory(t)
```

### 3. 添加碰撞检测

集成碰撞检测库（如 FCL）。

## 参考文件

- `deploy_pi0_R1.py` - 原始真机代码
- `eval_policy.py` - RoboTwin 评估流程
- `deploy_policy.py` - PI0 策略接口
- `h52eepose.py` - 正运动学参考

## 联系与支持

如有问题，请检查：
1. URDF 文件路径
2. 模型配置
3. 依赖库版本
4. GPU 内存

---

**版本**: 1.0  
**日期**: 2026-02-01  
**作者**: Based on deploy_pi0_R1.py

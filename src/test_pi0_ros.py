#!/usr/bin/env python3
"""
PI0 测试代码 - 使用 ROS Topics 作为输入
基于 test_pi0_with_ik.py，输出末端位置并通过 IK 计算关节角度

用法:
    # 仅输出 PI0 的 action 结果（不执行）
    python test_pi0_ros.py --train_config_name R1_FFT_pour_35_0130_5k \
        --checkpoint_path /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/10000 \
        --task_prompt "pour water" --print_only

    # 完整运行（需要 ROS topics）
    python test_pi0_ros.py --train_config_name R1_FFT_pour_35_0130_5k \
        --checkpoint_path /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/10000 \
        --task_prompt "pour water" --ros_mode

Author: Based on deploy_pi0_R1.py, h52eepose.py
"""

# 首先添加 ROS Python 路径（必须在其他 import 之前）
import sys
ROS_PYTHON_PATH = '/opt/ros/noetic/lib/python3/dist-packages'
if ROS_PYTHON_PATH not in sys.path:
    sys.path.insert(0, ROS_PYTHON_PATH)

import os
import numpy as np
import time
import traceback
import argparse
import cv2
import threading
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

# ROS imports
try:
    import rospy
    from sensor_msgs.msg import Image as ROSImage, CompressedImage
    from std_msgs.msg import Float64MultiArray
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    print("[Warning] ROS not available. Install rospy and cv_bridge for ROS mode.")
    ROS_AVAILABLE = False

# Kinpy for FK/IK
try:
    import kinpy as kp
    KINPY_AVAILABLE = True
except ImportError:
    print("[Warning] kinpy not available. Install kinpy for IK functionality.")
    KINPY_AVAILABLE = False

# Curobo for high-performance IK
try:
    from urdfik import URDFInverseKinematics
    CUROBO_AVAILABLE = True
except ImportError:
    print("[Warning] curobo not available. Install curobo for faster IK.")
    CUROBO_AVAILABLE = False

# rosbag for reading bag files
try:
    import rosbag
    ROSBAG_AVAILABLE = True
except ImportError:
    print("[Warning] rosbag not available. Install rosbag for bag file mode.")
    ROSBAG_AVAILABLE = False

# ZMQ for bridge mode
try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    print("[Warning] zmq not available. Install pyzmq for ZMQ bridge mode.")
    ZMQ_AVAILABLE = False

# 添加必要的路径
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)
sys.path.append(os.path.join(parent_directory, "../RoboTwin"))
sys.path.append(os.path.join(parent_directory, "../RoboTwin/policy/pi0"))
sys.path.append(os.path.join(parent_directory, "../RoboTwin/policy/pi0/src"))


class PI0Model:
    """PI0 模型封装 - 直接加载 checkpoint"""
    
    def __init__(self, train_config_name, checkpoint_path, pi0_step=10):
        """初始化 PI0 模型
        
        Args:
            train_config_name: 训练配置名称，如 "R1_FFT_pour_35_0130_5k"
            checkpoint_path: checkpoint 路径，如 "/home/pine/yzj/RoboTwin/policy/pi0/checkpoint/10000"
            pi0_step: 预测步数
        """
        self.train_config_name = train_config_name
        self.checkpoint_path = checkpoint_path
        self.pi0_step = pi0_step
        
        # 加载模型
        self._load_model()
        
        self.img_size = (224, 224)
        self.observation_window = None
        self.instruction = None
    
    def _load_model(self):
        """加载 PI0 模型"""
        import jax.numpy as jnp
        from openpi.models import model as _model
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config
        from openpi.training import checkpoints as _checkpoints
        import openpi.transforms as transforms
        import pathlib
        
        print(f"[PI0Model] Loading config: {self.train_config_name}")
        config = _config.get_config(self.train_config_name)
        
        # 查找 assets_id
        assets_path = os.path.join(self.checkpoint_path, "assets")
        if os.path.exists(assets_path):
            entries = os.listdir(assets_path)
            assets_id = entries[0] if entries else None
        else:
            assets_id = None
        print(f"[PI0Model] Checkpoint path: {self.checkpoint_path}")
        print(f"[PI0Model] Found assets_id: {assets_id}")
        
        # 创建 policy（使用 pathlib.Path 确保路径格式正确）
        checkpoint_path = pathlib.Path(self.checkpoint_path)
        self.policy = _policy_config.create_trained_policy(
            config,
            checkpoint_path,
            robotwin_repo_id=assets_id
        )
        print("[PI0Model] Model loaded successfully!")
    
    def set_language(self, instruction):
        """设置语言指令"""
        self.instruction = instruction
        print(f"[PI0Model] Set instruction: {instruction}")
    
    def update_observation_window(self, img_arr, state):
        """更新观察窗口
        
        Args:
            img_arr: [img_front, img_right, img_left] 三个相机图像 (H, W, C) RGB
            state: 当前状态向量
        """
        img_front, img_right, img_left = img_arr[0], img_arr[1], img_arr[2]
        
        # 转换为 (C, H, W) 格式
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))
        
        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }
    
    def get_action(self):
        """获取模型预测的动作"""
        assert self.observation_window is not None, "Please call update_observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]
    
    def reset(self):
        """重置模型状态"""
        self.instruction = None
        self.observation_window = None
        print("[PI0Model] Reset observation window and instruction")


class IKSolver:
    """逆运动学求解器 - 基于 Curobo 的高性能实现"""
    
    # URDF 路径
    URDF_PATH = "/home/pine/yzj/R1_urdf/galaxea_sim/assets/r1/robot.urdf"
    
    # 固定的 torso 值（与 h52eepose.py 保持一致）
    TORSO_FIXED = np.array([0.25, -0.4, -0.85, 0], dtype=np.float32)
    
    def __init__(self, side='left'):
        """初始化 IK 求解器
        
        Args:
            side: 'left' 或 'right'，指定左臂或右臂
        """
        if not CUROBO_AVAILABLE:
            raise RuntimeError("curobo is not available. Please install curobo.")
        
        self.side = side
        
        # Curobo solver
        try:
            ee_link = 'left_gripper_link' if side == 'left' else 'right_gripper_link'
            self.curobo_solver = URDFInverseKinematics(
                urdf_file=self.URDF_PATH,
                base_link='base_link',
                ee_link=ee_link
            )
            print(f"[IKSolver] Initialized Curobo solver for {side} arm")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Curobo solver: {e}")
    
    
    def forward_kinematics(self, joint_angles):
        """正运动学：从关节角度计算末端位置和姿态
        
        Args:
            joint_angles: 6个手臂关节角度（不包含 torso）
            
        Returns:
            pos: 3D 位置 [x, y, z]
            quat: 四元数 [w, x, y, z]
            euler: 欧拉角 [roll, pitch, yaw] (zyx顺序)
        """
        # 拼接 torso 和 arm 关节角度
        joint_angles_full = np.concatenate([self.TORSO_FIXED, joint_angles])
        
        # 使用 Curobo FK
        pos, quat, euler = self.curobo_solver.forward_kinematics(joint_angles_full)
        
        return pos, quat, euler
    
    def inverse_kinematics(self, target_pos, target_euler=None, initial_guess=None, 
                          max_iterations=500, tolerance=1e-3):
        """逆运动学：从末端位置（和可选的姿态）计算关节角度
        
        Args:
            target_pos: 目标位置 [x, y, z]
            target_euler: 目标欧拉角 [roll, pitch, yaw]，可选（zyx顺序）
            initial_guess: 初始关节角度猜测（6个），如果为 None 则使用零位
            max_iterations: 最大迭代次数 (unused for Curobo)
            tolerance: 收敛容差（位置误差，单位：米）(unused for Curobo)
            
        Returns:
            joint_angles: 6个手臂关节角度
            success: 是否成功求解
        """
        if target_euler is None:
            print(f"[IKSolver] Warning: target_euler is None, IK may not work properly")
            return np.zeros(6), False
        
        try:
            # 将欧拉角转换为四元数 (Curobo 需要 quaternion)
            r = R.from_euler('zyx', target_euler)
            quat_xyzw = r.as_quat()  # scipy 返回 [x, y, z, w]
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            
            # 准备完整的关节角度（包括 torso）
            if initial_guess is not None:
                current_joints_full = np.concatenate([self.TORSO_FIXED, initial_guess])
            else:
                current_joints_full = None
            
            # 调用 Curobo IK
            result = self.curobo_solver.solve_ik(
                target_position=target_pos,
                target_orientation=quat_wxyz,
                current_joints=current_joints_full
            )
            
            if result is not None and result.success.cpu().numpy().all():
                # 提取关节角度
                solution_raw = result.solution.cpu().numpy()
                print(f"[IKSolver] DEBUG: solution_raw shape = {solution_raw.shape}")
                
                # Curobo 返回 [batch_size, num_joints]
                if solution_raw.shape[0] == 0:
                    print(f"[IKSolver] Error: Empty solution returned")
                    return np.zeros(6), False
                
                joint_angles_full = solution_raw[0]  # 取第一个 batch
                print(f"[IKSolver] DEBUG: joint_angles_full shape = {joint_angles_full.shape}, value = {joint_angles_full}")
                
                # 确保至少有 4 个关节（torso）
                if len(joint_angles_full) < 4:
                    print(f"[IKSolver] Error: Expected at least 4 joints (torso), got {len(joint_angles_full)}")
                    return np.zeros(6), False
                
                joint_angles = joint_angles_full[4:]  # 只返回手臂关节（跳过 torso）
                print(f"[IKSolver] DEBUG: joint_angles (arm only) shape = {joint_angles.shape}")
                
                # 如果手臂关节不足 6 个，返回失败
                if len(joint_angles) < 6:
                    print(f"[IKSolver] Error: Expected 6 arm joints, got {len(joint_angles)}")
                    return np.zeros(6), False
                
                return joint_angles[:6], True  # 确保只返回前 6 个手臂关节
            else:
                print(f"[IKSolver] Curobo IK failed")
                return np.zeros(6), False
        except Exception as e:
            print(f"[IKSolver] Curobo IK error: {e}")
            return np.zeros(6), False


class ROSDataSubscriber:
    """ROS数据订阅器 - 订阅相机和机器人状态 topics
    
    订阅的 topics (与 bag2h5x_yzj_speed.py 保持一致):
    - /hdas/camera_head/rgb/image_rect_color/compressed  头部RGB
    - /hdas/camera_head/depth/depth_registered           头部深度
    - /left/camera/color/image_raw/compressed            左腕RGB
    - /left/camera/depth/image_rect_raw                  左腕深度  
    - /right/camera/color/image_raw/compressed           右腕RGB
    - /right/camera/depth/image_rect_raw                 右腕深度
    - /hdas/feedback_arm_left_low                        左臂状态
    - /hdas/feedback_arm_right_low                       右臂状态
    - /hdas/feedback_gripper_left_low                    左夹爪状态
    - /hdas/feedback_gripper_right_low                   右夹爪状态
    """
    
    def __init__(self):
        """初始化 ROS 订阅器"""
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS is not available. Cannot use ROS mode.")
        
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        
        # 数据缓存
        self.head_rgb = None
        self.head_depth = None
        self.left_wrist_rgb = None
        self.left_wrist_depth = None
        self.right_wrist_rgb = None
        self.right_wrist_depth = None
        self.arm_left_pos = None
        self.arm_right_pos = None
        self.gripper_left_pos = None
        self.gripper_right_pos = None
        
        print("[ROSSubscriber] Initializing ROS node...")
        try:
            rospy.init_node('pi0_test_node', anonymous=True, disable_signals=True)
        except rospy.exceptions.ROSException:
            print("[ROSSubscriber] ROS node already initialized")
        
        # 订阅相机 topics
        print("[ROSSubscriber] Subscribing to camera topics...")
        rospy.Subscriber("/hdas/camera_head/rgb/image_rect_color/compressed", 
                        CompressedImage, self._head_rgb_callback)
        rospy.Subscriber("/hdas/camera_head/depth/depth_registered", 
                        ROSImage, self._head_depth_callback)
        rospy.Subscriber("/left/camera/color/image_raw/compressed", 
                        CompressedImage, self._left_rgb_callback)
        rospy.Subscriber("/left/camera/depth/image_rect_raw", 
                        ROSImage, self._left_depth_callback)
        rospy.Subscriber("/right/camera/color/image_raw/compressed", 
                        CompressedImage, self._right_rgb_callback)
        rospy.Subscriber("/right/camera/depth/image_rect_raw", 
                        ROSImage, self._right_depth_callback)
        
        # 订阅机器人状态 topics
        print("[ROSSubscriber] Subscribing to robot state topics...")
        rospy.Subscriber("/hdas/feedback_arm_left_low", 
                        Float64MultiArray, self._arm_left_callback)
        rospy.Subscriber("/hdas/feedback_arm_right_low", 
                        Float64MultiArray, self._arm_right_callback)
        rospy.Subscriber("/hdas/feedback_gripper_left_low", 
                        Float64MultiArray, self._gripper_left_callback)
        rospy.Subscriber("/hdas/feedback_gripper_right_low", 
                        Float64MultiArray, self._gripper_right_callback)
        
        # 等待数据
        print("[ROSSubscriber] Waiting for initial data...")
        self._wait_for_data()
        print("[ROSSubscriber] Ready!")
    
    def _head_rgb_callback(self, msg):
        """头部 RGB 相机回调"""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                if img_bgr.shape[1] != 640 or img_bgr.shape[0] != 480:
                    img_bgr = cv2.resize(img_bgr, (640, 480), interpolation=cv2.INTER_LINEAR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.head_rgb = img_rgb
        except Exception as e:
            print(f"[ROSSubscriber] Error decoding head RGB: {e}")
    
    def _head_depth_callback(self, msg):
        """头部深度相机回调"""
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if depth_img.shape[1] != 640 or depth_img.shape[0] != 480:
                depth_img = cv2.resize(depth_img, (640, 480), interpolation=cv2.INTER_NEAREST)
            with self.lock:
                self.head_depth = depth_img
        except Exception as e:
            print(f"[ROSSubscriber] Error decoding head depth: {e}")
    
    def _left_rgb_callback(self, msg):
        """左腕 RGB 相机回调"""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                if img_bgr.shape[1] != 640 or img_bgr.shape[0] != 480:
                    img_bgr = cv2.resize(img_bgr, (640, 480), interpolation=cv2.INTER_LINEAR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.left_wrist_rgb = img_rgb
        except Exception as e:
            print(f"[ROSSubscriber] Error decoding left RGB: {e}")
    
    def _left_depth_callback(self, msg):
        """左腕深度相机回调"""
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if depth_img.shape[1] != 640 or depth_img.shape[0] != 480:
                depth_img = cv2.resize(depth_img, (640, 480), interpolation=cv2.INTER_NEAREST)
            with self.lock:
                self.left_wrist_depth = depth_img
        except Exception as e:
            print(f"[ROSSubscriber] Error decoding left depth: {e}")
    
    def _right_rgb_callback(self, msg):
        """右腕 RGB 相机回调"""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                if img_bgr.shape[1] != 640 or img_bgr.shape[0] != 480:
                    img_bgr = cv2.resize(img_bgr, (640, 480), interpolation=cv2.INTER_LINEAR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.right_wrist_rgb = img_rgb
        except Exception as e:
            print(f"[ROSSubscriber] Error decoding right RGB: {e}")
    
    def _right_depth_callback(self, msg):
        """右腕深度相机回调"""
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if depth_img.shape[1] != 640 or depth_img.shape[0] != 480:
                depth_img = cv2.resize(depth_img, (640, 480), interpolation=cv2.INTER_NEAREST)
            with self.lock:
                self.right_wrist_depth = depth_img
        except Exception as e:
            print(f"[ROSSubscriber] Error decoding right depth: {e}")
    
    def _arm_left_callback(self, msg):
        """左臂关节位置回调"""
        with self.lock:
            self.arm_left_pos = np.array(msg.data)
    
    def _arm_right_callback(self, msg):
        """右臂关节位置回调"""
        with self.lock:
            self.arm_right_pos = np.array(msg.data)
    
    def _gripper_left_callback(self, msg):
        """左夹爪位置回调"""
        with self.lock:
            self.gripper_left_pos = np.array(msg.data)
    
    def _gripper_right_callback(self, msg):
        """右夹爪位置回调"""
        with self.lock:
            self.gripper_right_pos = np.array(msg.data)
    
    def _wait_for_data(self, timeout=10.0):
        """等待初始数据到达"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                if (self.head_rgb is not None and 
                    self.arm_left_pos is not None and 
                    self.arm_right_pos is not None):
                    return True
            time.sleep(0.1)
        print("[ROSSubscriber] Warning: Timeout waiting for all ROS data")
        return False
    
    def get_images(self):
        """获取三个相机的 RGB 图像
        
        Returns:
            head_rgb: 头部 RGB 图像 (H, W, C)
            left_rgb: 左腕 RGB 图像 (H, W, C)
            right_rgb: 右腕 RGB 图像 (H, W, C)
        """
        with self.lock:
            return self.head_rgb, self.left_wrist_rgb, self.right_wrist_rgb
    
    def get_robot_state(self):
        """获取机器人状态
        
        Returns:
            arm_left: 左臂关节角度
            arm_right: 右臂关节角度
            gripper_left: 左夹爪位置
            gripper_right: 右夹爪位置
        """
        with self.lock:
            return (
                self.arm_left_pos.copy() if self.arm_left_pos is not None else None,
                self.arm_right_pos.copy() if self.arm_right_pos is not None else None,
                self.gripper_left_pos.copy() if self.gripper_left_pos is not None else None,
                self.gripper_right_pos.copy() if self.gripper_right_pos is not None else None
            )


class PI0TestController:
    """PI0 测试控制器 - 整合模型推理、FK/IK 计算"""
    
    def __init__(self, model, ik_solver_left=None, ik_solver_right=None):
        """初始化控制器
        
        Args:
            model: PI0Model 实例
            ik_solver_left: 左臂 IK 求解器
            ik_solver_right: 右臂 IK 求解器
        """
        self.model = model
        self.ik_solver_left = ik_solver_left
        self.ik_solver_right = ik_solver_right
        
        # 当前关节角度（用于 IK 初始猜测）
        self.current_joint_left = None
        self.current_joint_right = None
        
        # 当前夹爪值（用于计算 delta）
        self.current_gripper_left = None
        self.current_gripper_right = None
        
        np.set_printoptions(precision=4, suppress=True)
    
    def compute_state_vector(self, arm_left, arm_right, gripper_left, gripper_right):
        """计算状态向量（eepose 格式）
        
        根据 convert_aloha_data_to_lerbot_r1.py 中的定义:
        state_dims = [
            "left_ee_x", "left_ee_y", "left_ee_z",
            "left_ee_roll", "left_ee_pitch", "left_ee_yaw",
            "left_gripper",
            "right_ee_x", "right_ee_y", "right_ee_z",
            "right_ee_roll", "right_ee_pitch", "right_ee_yaw",
            "right_gripper",
        ]
        
        Args:
            arm_left: 左臂关节角度 (6,) 或 (7,) 包含夹爪
            arm_right: 右臂关节角度 (6,) 或 (7,) 包含夹爪
            gripper_left: 左夹爪位置 (0-1)
            gripper_right: 右夹爪位置 (0-1)
            
        Returns:
            state: 状态向量 (14,) 或 (32,) - [left_eepose(7), right_eepose(7), ...]
                   每个臂: [x, y, z, roll, pitch, yaw, gripper]
        """
        # 根据 norm_stats.json 的维度，这里使用 32 维（可能有额外的状态）
        state = np.zeros(32, dtype=np.float32)
        
        # 左臂末端位姿
        if self.ik_solver_left is not None and arm_left is not None:
            joint_left = arm_left[:6] if len(arm_left) > 6 else arm_left
            pos_left, quat_left, _ = self.ik_solver_left.forward_kinematics(joint_left)
            # 四元数转欧拉角 (roll, pitch, yaw)
            # kinpy 返回 [w, x, y, z], scipy 需要 [x, y, z, w]
            # 注意: 与 h52eepose.py 和 IKSolver 保持一致，使用 'zyx' 顺序
            r = R.from_quat([quat_left[1], quat_left[2], quat_left[3], quat_left[0]])
            euler_left = r.as_euler('zyx')  # 与 h52eepose.py 一致
            
            # [x, y, z, roll, pitch, yaw, gripper]
            state[0:3] = pos_left
            state[3:6] = euler_left
            # 夹爪值从 0-100 归一化到 0-1
            gripper_val = gripper_left[0] / 100.0 if gripper_left is not None and len(gripper_left) > 0 else 0.0
            state[6] = float(gripper_val)
            self.current_joint_left = joint_left
            self.current_gripper_left = gripper_val
        
        # 右臂末端位姿
        if self.ik_solver_right is not None and arm_right is not None:
            joint_right = arm_right[:6] if len(arm_right) > 6 else arm_right
            pos_right, quat_right, _ = self.ik_solver_right.forward_kinematics(joint_right)
            # 四元数转欧拉角 (roll, pitch, yaw)
            # 注意: 与 h52eepose.py 和 IKSolver 保持一致，使用 'zyx' 顺序
            r = R.from_quat([quat_right[1], quat_right[2], quat_right[3], quat_right[0]])
            euler_right = r.as_euler('zyx')  # 与 h52eepose.py 一致
            
            # [x, y, z, roll, pitch, yaw, gripper]
            state[7:10] = pos_right
            state[10:13] = euler_right
            # 夹爪值从 0-100 归一化到 0-1
            gripper_val = gripper_right[0] / 100.0 if gripper_right is not None and len(gripper_right) > 0 else 0.0
            state[13] = float(gripper_val)
            self.current_joint_right = joint_right
            self.current_gripper_right = gripper_val
        
        # 剩余维度填零（根据训练数据格式，可能有额外状态维度）
        
        return state
    
    def action_to_joint_angles(self, action, side='left'):
        """将 eepose action 转换为关节角度
        
        Action 格式 (根据 convert_aloha_data_to_lerbot_r1.py):
        action_dims = [
            "left_ee_x", "left_ee_y", "left_ee_z",
            "left_ee_roll", "left_ee_pitch", "left_ee_yaw",
            "left_gripper",
            "right_ee_x", "right_ee_y", "right_ee_z",
            "right_ee_roll", "right_ee_pitch", "right_ee_yaw",
            "right_gripper",
        ]
        
        Args:
            action: 完整的 14 维动作向量 [left(7), right(7)]
                    每个臂: [x, y, z, roll, pitch, yaw, gripper]
            side: 'left' 或 'right'
            
        Returns:
            joint_angles: 6个关节角度
            gripper: 夹爪位置 (0-1)
            success: IK 是否成功
        """
        if side == 'left':
            ik_solver = self.ik_solver_left
            initial_guess = self.current_joint_left
            action_slice = action[0:7]  # 左臂: 前 7 维
        else:
            ik_solver = self.ik_solver_right
            initial_guess = self.current_joint_right
            action_slice = action[7:14]  # 右臂: 后 7 维
        
        if ik_solver is None:
            print(f"[Controller] IK solver for {side} arm not available")
            return None, None, False
        
        # 解析 action: [x, y, z, roll, pitch, yaw, gripper]
        target_pos = action_slice[:3]
        target_euler = action_slice[3:6]  # roll, pitch, yaw (已经是欧拉角)
        gripper = action_slice[6]  # 夹爪位置 (0-1)
        
        # IK 求解
        joint_angles, success = ik_solver.inverse_kinematics(
            target_pos,
            target_euler=target_euler,
            initial_guess=initial_guess
        )
        
        return joint_angles, gripper, success
    
    def action_to_robot_command(self, action):
        """将 eepose action 转换为机器人命令格式
        
        Args:
            action: 完整的 14 维动作向量 [left(7), right(7)]
            
        Returns:
            left_joints: 左臂 6 个关节角度 (rad)
            left_gripper_raw: 左夹爪原始值 (0-100)
            right_joints: 右臂 6 个关节角度 (rad)
            right_gripper_raw: 右夹爪原始值 (0-100)
            success: (left_success, right_success)
        """
        left_joints, left_gripper, left_success = self.action_to_joint_angles(action, 'left')
        right_joints, right_gripper, right_success = self.action_to_joint_angles(action, 'right')
        
        # 夹爪从 0-1 转换回 0-100
        left_gripper_raw = left_gripper * 100.0 if left_gripper is not None else None
        right_gripper_raw = right_gripper * 100.0 if right_gripper is not None else None
        
        return left_joints, left_gripper_raw, right_joints, right_gripper_raw, (left_success, right_success)
    
    def run_single_inference(self, head_rgb, left_rgb, right_rgb, 
                            arm_left, arm_right, gripper_left, gripper_right,
                            task_prompt, print_only=True, show_joint_delta=False,
                            gt_action=None, compare_gt=False):
        """运行单次推理
        
        Args:
            head_rgb: 头部 RGB 图像 (H, W, C)
            left_rgb: 左腕 RGB 图像 (H, W, C)
            right_rgb: 右腕 RGB 图像 (H, W, C)
            arm_left: 左臂关节角度
            arm_right: 右臂关节角度
            gripper_left: 左夹爪位置
            gripper_right: 右夹爪位置
            task_prompt: 任务提示
            print_only: 是否仅打印结果（不计算 IK）
            show_joint_delta: 是否显示关节角度变化量（需要 compute_ik）
            gt_action: Ground truth action（用于比较，bag 模式）
            compare_gt: 是否比较预测与真实动作
            
        Returns:
            actions: 模型预测的动作序列
        """
        # 保存当前关节角度用于计算 delta
        prev_joint_left = self.current_joint_left.copy() if self.current_joint_left is not None else None
        prev_joint_right = self.current_joint_right.copy() if self.current_joint_right is not None else None
        prev_gripper_left = self.current_gripper_left
        prev_gripper_right = self.current_gripper_right
        
        # 计算状态向量
        state = self.compute_state_vector(arm_left, arm_right, gripper_left, gripper_right)
        print(f"\n[Controller] State vector (14 dims, format: xyz+euler+gripper):")
        print(f"    Left:  pos={state[:3]}, euler={state[3:6]}, gripper={state[6]:.4f}")
        print(f"    Right: pos={state[7:10]}, euler={state[10:13]}, gripper={state[13]:.4f}")
        
        # 设置语言指令
        if self.model.observation_window is None:
            self.model.set_language(task_prompt)
        
        # 准备图像数组 [head, right, left]
        img_arr = [head_rgb, right_rgb, left_rgb]
        
        # 更新观察窗口
        self.model.update_observation_window(img_arr, state)
        
        # 获取动作
        actions = self.model.get_action()[:self.model.pi0_step]
        actions = np.array(actions)
        
        print(f"\n[Controller] Received {len(actions)} actions from model:")
        for i, action in enumerate(actions):
            # 格式: [left(7): x,y,z,roll,pitch,yaw,gripper, right(7): x,y,z,roll,pitch,yaw,gripper]
            left_action = action[:7]
            right_action = action[7:14]
            print(f"  Action {i}:")
            print(f"    Left:  pos={left_action[:3]}, euler={left_action[3:6]}, gripper={left_action[6]:.4f}")
            print(f"    Right: pos={right_action[:3]}, euler={right_action[3:6]}, gripper={right_action[6]:.4f}")
        
        # 比较预测与真实动作 (bag 模式)
        if compare_gt and gt_action is not None:
            print(f"\n[Controller] === Prediction vs Ground Truth Comparison ===")
            pred_action = actions[0]  # 使用第一个预测动作比较
            
            # 计算误差
            left_pred = pred_action[:7]
            left_gt = gt_action[:7]
            right_pred = pred_action[7:14]
            right_gt = gt_action[7:14]
            
            # 位置误差 (米)
            left_pos_error = np.linalg.norm(left_pred[:3] - left_gt[:3])
            right_pos_error = np.linalg.norm(right_pred[:3] - right_gt[:3])
            
            # 姿态误差 (弧度)
            left_euler_error = np.linalg.norm(left_pred[3:6] - left_gt[3:6])
            right_euler_error = np.linalg.norm(right_pred[3:6] - right_gt[3:6])
            
            # 夹爪误差
            left_gripper_error = abs(left_pred[6] - left_gt[6])
            right_gripper_error = abs(right_pred[6] - right_gt[6])
            
            print(f"  Ground Truth Action:")
            print(f"    Left:  pos={left_gt[:3]}, euler={left_gt[3:6]}, gripper={left_gt[6]:.4f}")
            print(f"    Right: pos={right_gt[:3]}, euler={right_gt[3:6]}, gripper={right_gt[6]:.4f}")
            print(f"  Errors:")
            print(f"    Left:  pos_err={left_pos_error:.6f}m, euler_err={left_euler_error:.6f}rad, gripper_err={left_gripper_error:.4f}")
            print(f"    Right: pos_err={right_pos_error:.6f}m, euler_err={right_euler_error:.6f}rad, gripper_err={right_gripper_error:.4f}")
            print(f"    Total: pos_err={left_pos_error + right_pos_error:.6f}m, euler_err={left_euler_error + right_euler_error:.6f}rad")
        
        if not print_only:
            # 转换为关节角度并输出
            print("\n[Controller] Converting to joint angles:")
            print("  (gripper: 0-1 normalized, raw=0-100 for robot command)")
            for i, action in enumerate(actions):
                # action 格式: [left(7), right(7)] 共 14 维
                # 每个臂: [x, y, z, roll, pitch, yaw, gripper]
                action_dim = len(action)
                
                if action_dim >= 14:
                    # 双臂格式
                    left_joints, left_gripper, left_success = self.action_to_joint_angles(action, 'left')
                    right_joints, right_gripper, right_success = self.action_to_joint_angles(action, 'right')
                    
                    print(f"  Step {i}:")
                    if left_joints is not None:
                        # gripper: 显示归一化值和原始值
                        left_gripper_raw = left_gripper * 100.0
                        print(f"    Left  joints: {left_joints}, gripper: {left_gripper:.4f} (raw: {left_gripper_raw:.2f}), success: {left_success}")
                    if right_joints is not None:
                        right_gripper_raw = right_gripper * 100.0
                        print(f"    Right joints: {right_joints}, gripper: {right_gripper:.4f} (raw: {right_gripper_raw:.2f}), success: {right_success}")
                    
                    # 显示关节角度变化量 (仅对第一个 action)
                    if show_joint_delta and i == 0:
                        print(f"\n[Controller] === Joint Angle Delta (Action 0 vs Current State) ===")
                        if left_joints is not None and prev_joint_left is not None:
                            left_joint_delta = left_joints - prev_joint_left
                            left_gripper_delta = left_gripper - (prev_gripper_left if prev_gripper_left else 0)
                            print(f"    Left  joint delta:  {left_joint_delta}")
                            print(f"    Left  joint |delta|: {np.abs(left_joint_delta)}")
                            print(f"    Left  joint L2:     {np.linalg.norm(left_joint_delta):.6f} rad")
                            print(f"    Left  gripper delta: {left_gripper_delta:.4f}")
                        if right_joints is not None and prev_joint_right is not None:
                            right_joint_delta = right_joints - prev_joint_right
                            right_gripper_delta = right_gripper - (prev_gripper_right if prev_gripper_right else 0)
                            print(f"    Right joint delta:  {right_joint_delta}")
                            print(f"    Right joint |delta|: {np.abs(right_joint_delta)}")
                            print(f"    Right joint L2:     {np.linalg.norm(right_joint_delta):.6f} rad")
                            print(f"    Right gripper delta: {right_gripper_delta:.4f}")
                else:
                    # 单臂格式（不太可能，但保留兼容）
                    joints, gripper, success = self.action_to_joint_angles(action, 'left')
                    if joints is not None:
                        print(f"  Step {i}: joints: {joints}, gripper: {gripper:.4f}, success: {success}")
        
        return actions


def create_dummy_images():
    """创建虚拟图像用于测试"""
    head_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    left_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    right_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return head_rgb, left_rgb, right_rgb


class ZMQDataSubscriber:
    """通过 ZMQ 从 ros_bridge.py 接收数据（解决 Python 版本兼容问题）
    
    使用方法:
        1. 先用系统 Python 运行 ros_bridge.py
           /usr/bin/python3 ros_bridge.py
        
        2. 再运行 PI0 测试
           ./run_test_pi0.sh zmq
    """
    
    def __init__(self, host="localhost", port=5555, timeout_ms=5000):
        if not ZMQ_AVAILABLE:
            raise RuntimeError("zmq is not available. Install with: pip install pyzmq")
        
        print(f"[ZMQSubscriber] Connecting to {host}:{port}...")
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{host}:{port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")  # 订阅所有消息
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        
        # 数据存储
        self.data = None
        self.last_receive_time = 0
        
        print("[ZMQSubscriber] Connected! Waiting for data from ros_bridge.py...")
    
    def _decode_image(self, img_bytes):
        """解码 JPEG 图像"""
        if img_bytes is None:
            return None
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return None
    
    def receive(self):
        """接收一帧数据
        
        Returns:
            tuple: (head_rgb, left_rgb, right_rgb, arm_left, arm_right, gripper_left, gripper_right)
                   如果超时返回 None
        """
        try:
            import pickle
            data_bytes = self.socket.recv()
            data = pickle.loads(data_bytes)
            
            # 解码图像
            head_rgb = self._decode_image(data.get('head_rgb'))
            left_rgb = self._decode_image(data.get('left_rgb'))
            right_rgb = self._decode_image(data.get('right_rgb'))
            
            # 获取关节数据
            arm_left = np.array(data['arm_left']) if data.get('arm_left') else None
            arm_right = np.array(data['arm_right']) if data.get('arm_right') else None
            gripper_left = np.array(data['gripper_left']) if data.get('gripper_left') else None
            gripper_right = np.array(data['gripper_right']) if data.get('gripper_right') else None
            
            self.last_receive_time = time.time()
            
            return head_rgb, left_rgb, right_rgb, arm_left, arm_right, gripper_left, gripper_right
            
        except zmq.Again:
            print("[ZMQSubscriber] Timeout waiting for data. Is ros_bridge.py running?")
            return None
        except Exception as e:
            print(f"[ZMQSubscriber] Error receiving data: {e}")
            return None
    
    def close(self):
        """关闭连接"""
        self.socket.close()
        self.context.term()


class ZMQCommandPublisher:
    """通过 ZMQ 向 ros_bridge.py 发送控制命令
    
    ros_bridge.py 会将命令发布到 ROS topics:
        - /motion_target/target_joint_state_arm_left
        - /motion_target/target_joint_state_arm_right
        - /motion_control/position_control_gripper_left
        - /motion_control/position_control_gripper_right
    """
    
    # 初始位置配置
    # 左臂初始关节角度 (6个关节，发送时会追加无效夹爪值变成7维)
    INIT_LEFT_JOINTS = np.array([
        -0.16744680851063828, 2.0108510638297874, -0.6593617021276595,
        2.002127659574468, 0.39382978723404255, -1.7193617021276595
    ], dtype=np.float32)
    
    # 右臂初始关节角度 (6个关节，发送时会追加无效夹爪值变成7维)
    INIT_RIGHT_JOINTS = np.array([
        0.19234042553191488, 1.8925531914893616, -0.6874468085106383,
        -1.6057446808510638, -0.10148936170212766, 1.3085106382978724
    ], dtype=np.float32)
    
    # 夹爪初始位置 (0-100)
    INIT_GRIPPER_LEFT = 100.0
    INIT_GRIPPER_RIGHT = 100.0
    
    # 无效夹爪占位值（第7维）
    INVALID_GRIPPER_VALUE = -2.7
    
    def __init__(self, host="localhost", port=5556):
        if not ZMQ_AVAILABLE:
            raise RuntimeError("zmq is not available. Install with: pip install pyzmq")
        
        print(f"[ZMQCommandPub] Connecting to {host}:{port}...")
        
        import pickle
        self.pickle = pickle
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.connect(f"tcp://{host}:{port}")
        
        # 等待连接建立
        time.sleep(0.5)
        
        self.cmd_count = 0
        print("[ZMQCommandPub] Connected! Ready to send commands.")
    
    def send_command(self, left_joints=None, left_gripper_raw=None, 
                    right_joints=None, right_gripper_raw=None):
        """发送控制命令
        
        Args:
            left_joints: 左臂 6 个关节角度 (rad)，numpy array 或 list
            left_gripper_raw: 左夹爪位置 (0-100)
            right_joints: 右臂 6 个关节角度 (rad)
            right_gripper_raw: 右夹爪位置 (0-100)
            
        Note:
            发送给机器人的关节角度是 7 维（6关节 + 无效夹爪占位值-2.7）
            真正的夹爪控制通过单独的 gripper_left/gripper_right 字段发送
        """
        # 将 6 维关节角扩展为 7 维（追加无效夹爪占位值）
        def make_7dim(joints):
            if joints is None:
                return None
            joints_list = list(joints)
            if len(joints_list) == 6:
                joints_list.append(self.INVALID_GRIPPER_VALUE)
            return joints_list
        
        cmd = {
            'arm_left': make_7dim(left_joints),
            'arm_right': make_7dim(right_joints),
            'gripper_left': float(left_gripper_raw) if left_gripper_raw is not None else None,
            'gripper_right': float(right_gripper_raw) if right_gripper_raw is not None else None,
        }
        
        try:
            cmd_bytes = self.pickle.dumps(cmd)
            self.socket.send(cmd_bytes)
            self.cmd_count += 1
            return True
        except Exception as e:
            print(f"[ZMQCommandPub] Error sending command: {e}")
            return False
    
    def send_init_position(self, wait_for_confirm=True):
        """发送初始位置命令
        
        将机器人移动到预设的初始位置：
        - 左臂: INIT_LEFT_JOINTS (6个关节)
        - 右臂: INIT_RIGHT_JOINTS (6个关节)
        - 夹爪: 100 (完全打开)
        
        Args:
            wait_for_confirm: 是否等待用户确认
            
        Returns:
            success: 是否成功发送
        """
        # 生成 7 维关节角（6 关节 + 无效夹爪占位值）
        left_joints_7 = list(self.INIT_LEFT_JOINTS) + [self.INVALID_GRIPPER_VALUE]
        right_joints_7 = list(self.INIT_RIGHT_JOINTS) + [self.INVALID_GRIPPER_VALUE]
        
        print("\n" + "="*60)
        print("🤖 ROBOT INITIALIZATION - MOVE TO INITIAL POSITION")
        print("="*60)
        print(f"\nInitial Position Configuration:")
        print(f"  Left Arm Joints (6):  {list(self.INIT_LEFT_JOINTS)}")
        print(f"  Right Arm Joints (6): {list(self.INIT_RIGHT_JOINTS)}")
        print(f"  (Will send as 7-dim with invalid gripper={self.INVALID_GRIPPER_VALUE})")
        print(f"  Left Gripper:  {self.INIT_GRIPPER_LEFT:.1f} (fully open)")
        print(f"  Right Gripper: {self.INIT_GRIPPER_RIGHT:.1f} (fully open)")
        print(f"\n⚠️  WARNING: The robot will move to the initial position!")
        print(f"    Make sure the workspace is clear and safe.")
        
        if wait_for_confirm:
            print("\n" + "-"*60)
            while True:
                user_input = input("Type 'yes' to send init position command (or 'no' to abort): ").strip().lower()
                if user_input == 'yes':
                    print("-"*60)
                    break
                elif user_input == 'no':
                    print("\n❌ Aborted by user - init position NOT sent.")
                    return False
                else:
                    print("Please type 'yes' to continue or 'no' to abort.")
        
        # 发送初始位置命令
        success = self.send_command(
            left_joints=self.INIT_LEFT_JOINTS,
            left_gripper_raw=self.INIT_GRIPPER_LEFT,
            right_joints=self.INIT_RIGHT_JOINTS,
            right_gripper_raw=self.INIT_GRIPPER_RIGHT
        )
        
        if success:
            print("\n✅ Initial position command sent successfully!")
            print("   Waiting for robot to reach initial position...")
            time.sleep(2.0)  # 等待机器人移动到位
            print("   Robot should now be at initial position.")
        else:
            print("\n❌ Failed to send initial position command!")
        
        return success
    
    def close(self):
        """关闭连接"""
        self.socket.close()
        self.context.term()


class RosbagDataReader:
    """从 rosbag 文件读取数据（不需要 ROS 运行时）"""
    
    # Topic 名称 - 优先使用原始高频topic
    TOPICS_PRIMARY = {
        'head_rgb': '/hdas/camera_head/rgb/image_rect_color/compressed',
        'left_rgb': '/left/camera/color/image_raw/compressed',
        'right_rgb': '/right/camera/color/image_raw/compressed',
        'arm_left': '/hdas/feedback_arm_left',
        'arm_right': '/hdas/feedback_arm_right',
        'gripper_left': '/hdas/feedback_gripper_left',
        'gripper_right': '/hdas/feedback_gripper_right',
    }
    
    # 降频topic作为fallback
    TOPICS_FALLBACK = {
        'head_rgb': '/hdas/camera_head/rgb/image_rect_color/compressed',
        'left_rgb': '/left/camera/color/image_raw/compressed',
        'right_rgb': '/right/camera/color/image_raw/compressed',
        'arm_left': '/hdas/feedback_arm_left_low',
        'arm_right': '/hdas/feedback_arm_right_low',
        'gripper_left': '/hdas/feedback_gripper_left_low',
        'gripper_right': '/hdas/feedback_gripper_right_low',
    }
    
    def __init__(self, bag_path):
        """初始化 rosbag 读取器
        
        Args:
            bag_path: rosbag 文件路径
        """
        if not ROSBAG_AVAILABLE:
            raise RuntimeError("rosbag is not available. Install it with: pip install rosbag")
        
        self.bag_path = bag_path
        print(f"[RosbagReader] Opening bag file: {bag_path}")
        
        self.bag = rosbag.Bag(bag_path, 'r')
        
        # 检测bag中可用的topics并选择使用哪套
        available_topics = self.bag.get_type_and_topic_info()[1].keys()
        print(f"\n[BagReader] Detecting topics in bag...")
        
        # 检查是否有原始高频topics（关键是关节和夹爪数据）
        use_primary = all(topic in available_topics for topic in [
            self.TOPICS_PRIMARY['arm_left'],
            self.TOPICS_PRIMARY['arm_right'],
            self.TOPICS_PRIMARY['gripper_left'],
            self.TOPICS_PRIMARY['gripper_right']
        ])
        
        if use_primary:
            self.TOPICS = self.TOPICS_PRIMARY.copy()
            print(f"[BagReader] ✓ Using PRIMARY (high freq) topics for better performance")
        else:
            self.TOPICS = self.TOPICS_FALLBACK.copy()
            print(f"[BagReader] Using FALLBACK (low freq) topics")
        
        print(f"[BagReader] Selected topics:")
        for key, topic in self.TOPICS.items():
            status = "✓" if topic in available_topics else "✗"
            print(f"    {status} {key}: {topic}")
        
        # 获取所有消息并按时间排序
        self._load_data()
        
        print(f"[RosbagReader] Loaded {len(self.timestamps)} frames")
    
    def _decode_compressed_image(self, msg):
        """解码压缩图像"""
        np_arr = np.frombuffer(msg.data, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            if img_bgr.shape[1] != 640 or img_bgr.shape[0] != 480:
                img_bgr = cv2.resize(img_bgr, (640, 480), interpolation=cv2.INTER_LINEAR)
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return None
    
    def _load_data(self):
        """加载所有数据"""
        # 收集各个 topic 的消息
        raw_data = {key: [] for key in self.TOPICS.keys()}
        timestamps = {key: [] for key in self.TOPICS.keys()}
        
        topic_list = list(self.TOPICS.values())
        
        for topic, msg, t in self.bag.read_messages(topics=topic_list):
            ts = t.to_sec()
            
            for key, topic_name in self.TOPICS.items():
                if topic == topic_name:
                    timestamps[key].append(ts)
                    raw_data[key].append(msg)
                    break
        
        # 找到公共时间范围
        all_ts = []
        for key in timestamps:
            all_ts.extend(timestamps[key])
        
        if not all_ts:
            raise ValueError("No data found in bag file")
        
        t_start, t_end = max(min(ts) for ts in timestamps.values() if ts), \
                         min(max(ts) for ts in timestamps.values() if ts)
        
        # 创建统一的时间线（15Hz）
        TARGET_FPS = 15
        duration = t_end - t_start
        num_samples = int(duration * TARGET_FPS)
        
        if num_samples < 1:
            raise ValueError("Bag file too short")
        
        self.timestamps = np.linspace(t_start, t_end, num_samples)
        
        # 对齐数据
        self.data = {key: [] for key in self.TOPICS.keys()}
        
        for i, target_ts in enumerate(self.timestamps):
            for key in self.TOPICS.keys():
                if not timestamps[key]:
                    self.data[key].append(None)
                    continue
                
                # 找最近的时间戳
                ts_arr = np.array(timestamps[key])
                idx = np.argmin(np.abs(ts_arr - target_ts))
                self.data[key].append(raw_data[key][idx])
        
        # 解码图像
        print("[RosbagReader] Decoding images...")
        for key in ['head_rgb', 'left_rgb', 'right_rgb']:
            decoded = []
            for msg in self.data[key]:
                if msg is not None:
                    decoded.append(self._decode_compressed_image(msg))
                else:
                    decoded.append(None)
            self.data[key] = decoded
        
        # 解码关节角度 (JointState 消息使用 position 属性)
        for key in ['arm_left', 'arm_right', 'gripper_left', 'gripper_right']:
            decoded = []
            for msg in self.data[key]:
                if msg is not None:
                    # JointState 消息有 position 属性
                    if hasattr(msg, 'position'):
                        decoded.append(np.array(msg.position))
                    elif hasattr(msg, 'data'):
                        decoded.append(np.array(msg.data))
                    else:
                        # 尝试直接转换
                        decoded.append(np.array(msg))
                else:
                    decoded.append(None)
            self.data[key] = decoded
        
        self.current_idx = 0
    
    def __len__(self):
        return len(self.timestamps)
    
    def get_frame(self, idx):
        """获取指定帧的数据
        
        Returns:
            head_rgb, left_rgb, right_rgb, arm_left, arm_right, gripper_left, gripper_right
        """
        if idx >= len(self.timestamps):
            return None, None, None, None, None, None, None
        
        return (
            self.data['head_rgb'][idx],
            self.data['left_rgb'][idx],
            self.data['right_rgb'][idx],
            self.data['arm_left'][idx],
            self.data['arm_right'][idx],
            self.data['gripper_left'][idx],
            self.data['gripper_right'][idx],
        )
    
    def compute_gt_action(self, current_idx, ik_solver_left=None, ik_solver_right=None):
        """计算 Ground Truth Action (下一帧的 eepose)
        
        通过下一帧的关节角度计算 FK 得到下一帧的末端位姿作为 GT action
        
        Args:
            current_idx: 当前帧索引
            ik_solver_left: 左臂 IK 求解器（用于 FK）
            ik_solver_right: 右臂 IK 求解器（用于 FK）
            
        Returns:
            gt_action: Ground truth action (14,) [left(7), right(7)]
                      每个臂: [x, y, z, roll, pitch, yaw, gripper]
                      如果下一帧不存在返回 None
        """
        next_idx = current_idx + 1
        if next_idx >= len(self.timestamps):
            return None
        
        # 获取下一帧数据
        arm_left = self.data['arm_left'][next_idx]
        arm_right = self.data['arm_right'][next_idx]
        gripper_left = self.data['gripper_left'][next_idx]
        gripper_right = self.data['gripper_right'][next_idx]
        
        gt_action = np.zeros(14, dtype=np.float32)
        
        # 计算左臂 eepose
        if ik_solver_left is not None and arm_left is not None:
            joint_left = arm_left[:6] if len(arm_left) > 6 else arm_left
            pos_left, quat_left, _ = ik_solver_left.forward_kinematics(joint_left)
            r = R.from_quat([quat_left[1], quat_left[2], quat_left[3], quat_left[0]])
            euler_left = r.as_euler('xyz')
            
            gt_action[0:3] = pos_left
            gt_action[3:6] = euler_left
            gripper_val = gripper_left[0] / 100.0 if gripper_left is not None and len(gripper_left) > 0 else 0.0
            gt_action[6] = float(gripper_val)
        
        # 计算右臂 eepose
        if ik_solver_right is not None and arm_right is not None:
            joint_right = arm_right[:6] if len(arm_right) > 6 else arm_right
            pos_right, quat_right, _ = ik_solver_right.forward_kinematics(joint_right)
            r = R.from_quat([quat_right[1], quat_right[2], quat_right[3], quat_right[0]])
            euler_right = r.as_euler('xyz')
            
            gt_action[7:10] = pos_right
            gt_action[10:13] = euler_right
            gripper_val = gripper_right[0] / 100.0 if gripper_right is not None and len(gripper_right) > 0 else 0.0
            gt_action[13] = float(gripper_val)
        
        return gt_action
    
    def get_next_frame(self):
        """获取下一帧数据"""
        frame = self.get_frame(self.current_idx)
        self.current_idx += 1
        return frame
    
    def reset(self):
        """重置到开始"""
        self.current_idx = 0
    
    def close(self):
        """关闭 bag 文件"""
        self.bag.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PI0 Test with ROS Topics")
    
    # 模型参数
    parser.add_argument("--train_config_name", type=str, default="R1_FFT_pour_35_0130_5k",
                       help="Training config name")
    parser.add_argument("--checkpoint_path", type=str, 
                       default="/home/pine/yzj/RoboTwin/policy/pi0/checkpoint/10000",
                       help="Checkpoint path")
    parser.add_argument("--pi0_step", type=int, default=10,
                       help="Number of steps to predict")
    
    # 任务参数
    parser.add_argument("--task_prompt", type=str, default="pour",
                       help="Task prompt/instruction")
    parser.add_argument("--n_iterations", type=int, default=10,
                       help="Number of test iterations")
    
    # 运行模式
    parser.add_argument("--ros_mode", action="store_true",
                       help="Use ROS topics for observation (requires ROS)")
    parser.add_argument("--zmq_mode", action="store_true",
                       help="Use ZMQ bridge mode (run ros_bridge.py first)")
    parser.add_argument("--zmq_host", type=str, default="localhost",
                       help="ZMQ bridge host")
    parser.add_argument("--zmq_port", type=int, default=5555,
                       help="ZMQ bridge port")
    parser.add_argument("--bag_file", type=str, default=None,
                       help="Path to rosbag file for offline testing")
    parser.add_argument("--dummy_mode", action="store_true",
                       help="Run in dummy mode with random images")
    parser.add_argument("--print_only", action="store_true", default=True,
                       help="Only print action results, don't compute IK")
    parser.add_argument("--compute_ik", action="store_true",
                       help="Compute IK from eepose actions")
    
    # 分析功能
    parser.add_argument("--show_joint_delta", action="store_true",
                       help="Show joint angle delta after IK (requires --compute_ik)")
    parser.add_argument("--compare_gt_action", action="store_true",
                       help="Compare predicted action with ground truth (bag mode only)")
    
    # 控制机器人功能
    parser.add_argument("--publish_command", action="store_true",
                       help="Publish control commands to ROS via ros_bridge.py (zmq mode only)")
    parser.add_argument("--cmd_port", type=int, default=5556,
                       help="ZMQ command port for publishing control commands")
    parser.add_argument("--action_index", type=int, default=0,
                       help="Which action to start executing from (0 = first action)")
    parser.add_argument("--execute_steps", type=int, default=1,
                       help="Number of actions to execute per prediction (should be <= pi0_step)")
    parser.add_argument("--init_robot", action="store_true", default=True,
                       help="Initialize robot to starting position before control")
    parser.add_argument("--confirm_each_command", action="store_true",
                       help="Require ENTER confirmation before sending each batch of control commands")
    parser.add_argument("--auto_execute_threshold", type=float, default=None,
                       help="Auto execute if max delta_action < threshold (e.g., 0.05)")
    parser.add_argument("--auto_joint_threshold", type=float, default=None,
                       help="Auto execute if max delta_joint < threshold (e.g., 0.2)")
    parser.add_argument("--execution_delay", type=float, default=1.0,
                       help="Delay in seconds after sending commands to allow robot to execute (default: 1.0)")
    
    args = parser.parse_args()
    
    # 初始化模型
    print("\n" + "="*60)
    print("Initializing PI0 Model...")
    print("="*60)
    model = PI0Model(
        train_config_name=args.train_config_name,
        checkpoint_path=args.checkpoint_path,
        pi0_step=args.pi0_step
    )
    
    # 初始化 IK 求解器（bag/zmq 模式需要 FK 计算状态向量）
    ik_solver_left = None
    ik_solver_right = None
    is_bag_mode = args.bag_file is not None
    is_zmq_mode = args.zmq_mode
    need_ik = args.compute_ik or is_bag_mode or is_zmq_mode  # bag/zmq 模式总是需要 FK
    if need_ik and CUROBO_AVAILABLE:
        print("\nInitializing Curobo IK Solvers...")
        ik_solver_left = IKSolver(side='left')
        ik_solver_right = IKSolver(side='right')
    elif need_ik and not CUROBO_AVAILABLE:
        print("\n[Error] curobo not available, FK/IK will not work!")
        print("         State will be zeros, which may affect model output.")
        print("         Install curobo: https://github.com/NVlabs/curobo")
        return
    
    # 初始化控制器
    controller = PI0TestController(model, ik_solver_left, ik_solver_right)
    
    # 选择数据源
    if args.zmq_mode:
        # ZMQ 桥接模式（推荐用于实时测试）
        print(f"\n[Main] Running in ZMQ BRIDGE mode: {args.zmq_host}:{args.zmq_port}")
        print("[Main] Make sure ros_bridge.py is running with system Python!")
        print("[Main] Command: /usr/bin/python3 /home/pine/yzj/src/ros_bridge.py")
        
        if not ZMQ_AVAILABLE:
            print("[Error] zmq is not available. Install with: pip install pyzmq")
            return
        
        # 如果启用发布命令，需要计算 IK
        if args.publish_command and not args.compute_ik:
            print("[Main] --publish_command requires --compute_ik, enabling it automatically...")
            args.compute_ik = True
        
        zmq_sub = ZMQDataSubscriber(host=args.zmq_host, port=args.zmq_port)
        
        # 命令发布器
        cmd_pub = None
        if args.publish_command:
            print(f"\n[Main] ⚠️ CONTROL MODE ENABLED! Commands will be sent to robot!")
            print(f"[Main] Command port: {args.cmd_port}")
            print(f"[Main] Prediction steps: {args.pi0_step}, Execute steps: {args.execute_steps}")
            if args.confirm_each_command:
                print(f"[Main] Confirm mode: Will ask for confirmation before each batch")
            cmd_pub = ZMQCommandPublisher(host=args.zmq_host, port=args.cmd_port)
            
            # 发送初始位置
            if args.init_robot:
                init_success = cmd_pub.send_init_position(wait_for_confirm=True)
                if not init_success:
                    print("[Main] Failed to send initial position, aborting...")
                    zmq_sub.close()
                    cmd_pub.close()
                    return
                
                # 等待机器人移动到初始位置，然后用户输入 yes 确认
                print("\n" + "="*60)
                print("🤖 WAITING FOR ROBOT TO REACH INITIAL POSITION")
                print("="*60)
                print("\nPlease wait for the robot to move to the initial position.")
                print("Once the robot has stopped and is ready, type 'yes' to start PI0 control.\n")
                
                while True:
                    user_input = input("Type 'yes' to start PI0 control (or 'no' to abort): ").strip().lower()
                    if user_input == 'yes':
                        print("\n✅ Starting PI0 control loop...")
                        break
                    elif user_input == 'no':
                        print("\n❌ Aborted by user.")
                        zmq_sub.close()
                        cmd_pub.close()
                        return
                    else:
                        print("Please type 'yes' to continue or 'no' to abort.")
                
                print("="*60)
        
        try:
            for step in range(args.n_iterations):
                print(f"\n{'='*60}")
                print(f"Frame {step + 1}/{args.n_iterations}")
                print("="*60)
                
                # 从 ZMQ 接收数据
                result = zmq_sub.receive()
                if result is None:
                    print("[Main] No data received, retrying...")
                    continue
                
                head_rgb, left_rgb, right_rgb, arm_left, arm_right, gripper_left, gripper_right = result
                
                if head_rgb is None or left_rgb is None or right_rgb is None:
                    print("[Main] Missing image data, skipping...")
                    continue
                
                # 调试：验证 FK 计算
                if arm_left is not None and controller.ik_solver_left is not None:
                    fk_pos, _, fk_euler = controller.ik_solver_left.forward_kinematics(arm_left[:6])
                    print(f"\n[DEBUG] FK Verification for LEFT arm:")
                    print(f"    Current joints: {list(arm_left[:6])}")
                    print(f"    FK position:    {fk_pos}")
                    print(f"    FK euler(zyx):  {fk_euler}")
                
                if arm_right is not None and controller.ik_solver_right is not None:
                    fk_pos, _, fk_euler = controller.ik_solver_right.forward_kinematics(arm_right[:6])
                    print(f"[DEBUG] FK Verification for RIGHT arm:")
                    print(f"    Current joints: {list(arm_right[:6])}")
                    print(f"    FK position:    {fk_pos}")
                    print(f"    FK euler(zyx):  {fk_euler}")
                
                # 运行推理
                actions = controller.run_single_inference(
                    head_rgb, left_rgb, right_rgb,
                    arm_left, arm_right, gripper_left, gripper_right,
                    args.task_prompt,
                    print_only=not args.compute_ik,
                    show_joint_delta=args.show_joint_delta
                )
                
                # 发布控制命令 - 批量执行多步
                if args.publish_command and cmd_pub is not None and len(actions) >= args.execute_steps:
                    
                    # 计算所有要执行的动作的 IK
                    execute_steps = min(args.execute_steps, len(actions))
                    # ============================================
                    # 流式执行：逐个计算IK并立即发送
                    # ============================================
                    print(f"\n" + "="*60)
                    print(f"🚀 STREAMING EXECUTION - {execute_steps} actions")
                    print("="*60)
                    
                    # 显示当前状态
                    print(f"\n📍 STARTING STATE:")
                    if arm_left is not None:
                        print(f"    Left  arm joints:  {list(arm_left[:6])}")
                    if arm_right is not None:
                        print(f"    Right arm joints:  {list(arm_right[:6])}")
                    if gripper_left is not None:
                        gl_val = gripper_left[0] if hasattr(gripper_left, '__len__') else gripper_left
                        print(f"    Left  gripper:     {gl_val:.2f} (0-100)")
                    if gripper_right is not None:
                        gr_val = gripper_right[0] if hasattr(gripper_right, '__len__') else gripper_right
                        print(f"    Right gripper:     {gr_val:.2f} (0-100)")
                    
                    # 计算当前状态的 eepose (通过 FK)
                    current_state_action = None
                    if arm_left is not None and arm_right is not None:
                        current_state_full = controller.compute_state_vector(
                            arm_left, arm_right, gripper_left, gripper_right
                        )
                        current_state_action = current_state_full[:14]
                    
                    # 用于跟踪前一帧状态
                    prev_action = current_state_action
                    prev_left_joints = np.array(arm_left[:6]) if arm_left is not None else None
                    prev_right_joints = np.array(arm_right[:6]) if arm_right is not None else None
                    
                    # 用于 IK 失败时的回退值
                    last_good_left_joints = None
                    last_good_right_joints = None
                    last_good_left_gripper = None
                    last_good_right_gripper = None
                    
                    # 尝试从上次发送的命令获取回退值
                    if hasattr(cmd_pub, '_last_left_joints') and cmd_pub._last_left_joints is not None:
                        last_good_left_joints = cmd_pub._last_left_joints
                        last_good_left_gripper = getattr(cmd_pub, '_last_left_gripper', 50.0)
                    if hasattr(cmd_pub, '_last_right_joints') and cmd_pub._last_right_joints is not None:
                        last_good_right_joints = cmd_pub._last_right_joints
                        last_good_right_gripper = getattr(cmd_pub, '_last_right_gripper', 50.0)
                    
                    # 统计
                    ik_fail_count = 0
                    executed_count = 0
                    skipped_count = 0
                    
                    print(f"\n📋 PROCESSING ACTIONS:")
                    print("-"*60)
                    
                    # 逐个处理动作
                    for action_idx in range(execute_steps):
                        action = actions[action_idx]
                        
                        # IK 求解
                        left_joints, left_gripper_raw, right_joints, right_gripper_raw, success = \
                            controller.action_to_robot_command(action)
                        
                        print(f"[DEBUG] Action {action_idx}: left_joints type={type(left_joints)}, shape={np.array(left_joints).shape if left_joints is not None else 'None'}")
                        print(f"[DEBUG] Action {action_idx}: right_joints type={type(right_joints)}, shape={np.array(right_joints).shape if right_joints is not None else 'None'}")
                        
                        left_ok, right_ok = success[0], success[1]
                        left_valid, right_valid = True, True
                        
                        # 处理左手 IK 失败 - 用左手上一次成功的值代替
                        if not left_ok:
                            if last_good_left_joints is not None:
                                print(f"[IK] ⚠️ Action {action_idx}: Left IK failed, using left's previous value")
                                left_joints = last_good_left_joints
                                left_gripper_raw = last_good_left_gripper
                                ik_fail_count += 1
                            else:
                                print(f"[IK] ❌ Action {action_idx}: Left IK failed, no left fallback available!")
                                left_valid = False
                        
                        # 处理右手 IK 失败 - 用右手上一次成功的值代替
                        if not right_ok:
                            if last_good_right_joints is not None:
                                print(f"[IK] ⚠️ Action {action_idx}: Right IK failed, using right's previous value")
                                right_joints = last_good_right_joints
                                right_gripper_raw = last_good_right_gripper
                                ik_fail_count += 1
                            else:
                                print(f"[IK] ❌ Action {action_idx}: Right IK failed, no right fallback available!")
                                right_valid = False
                        
                        # 只有当两边都无效时才跳过
                        if not left_valid and not right_valid:
                            print(f"[IK] ❌ Action {action_idx}: Both arms failed with no fallback, skipping!")
                            skipped_count += 1
                            continue
                        
                        # 计算 delta
                        left_joints_arr = np.array(left_joints)
                        right_joints_arr = np.array(right_joints)
                        
                        if prev_action is not None:
                            delta_action = action - prev_action
                            delta_action_norm = np.linalg.norm(delta_action)
                        else:
                            delta_action_norm = 0.0
                        
                        if prev_left_joints is not None:
                            delta_left_norm = np.linalg.norm(left_joints_arr - prev_left_joints)
                        else:
                            delta_left_norm = 0.0
                        
                        if prev_right_joints is not None:
                            delta_right_norm = np.linalg.norm(right_joints_arr - prev_right_joints)
                        else:
                            delta_right_norm = 0.0
                        
                        max_joint_delta = max(delta_left_norm, delta_right_norm)
                        
                        # IK 状态标记
                        ik_status = ""
                        if not left_ok or not right_ok:
                            ik_status = " ⚠️"
                        
                        # 输出动作信息
                        print(f"  Action {action_idx}{ik_status}:")
                        print(f"    Δ action (L2):     {delta_action_norm:.6f}")
                        print(f"    Δ joints L (L2):   {delta_left_norm:.6f} rad")
                        print(f"    Δ joints R (L2):   {delta_right_norm:.6f} rad")
                        
                        # 判断是否执行
                        should_send = True
                        auto_execute = False
                        
                        if args.auto_execute_threshold is not None and args.auto_joint_threshold is not None:
                            if delta_action_norm < args.auto_execute_threshold and max_joint_delta < args.auto_joint_threshold:
                                auto_execute = True
                                print(f"    ✅ AUTO: within thresholds")
                            else:
                                print(f"    ⚠️  THRESHOLD: delta_action={delta_action_norm:.6f}, delta_joint={max_joint_delta:.6f}")
                                # 超过阈值时，如果是确认模式则询问
                                if args.confirm_each_command:
                                    user_input = input(f"    Execute action {action_idx}? (ENTER=yes, 'skip'=no): ").strip().lower()
                                    if user_input == 'skip':
                                        should_send = False
                        
                        # 立即发送命令
                        if should_send:
                            cmd_pub.send_command(
                                left_joints=left_joints,
                                left_gripper_raw=left_gripper_raw,
                                right_joints=right_joints,
                                right_gripper_raw=right_gripper_raw
                            )
                            executed_count += 1
                            status_msg = "AUTO SENT" if auto_execute else "SENT"
                            print(f"    🚀 {status_msg}")
                            
                            # 保存成功的值作为下一帧的回退（左右分开保存）
                            if left_ok:
                                last_good_left_joints = left_joints
                                last_good_left_gripper = left_gripper_raw
                                cmd_pub._last_left_joints = left_joints
                                cmd_pub._last_left_gripper = left_gripper_raw
                            if right_ok:
                                last_good_right_joints = right_joints
                                last_good_right_gripper = right_gripper_raw
                                cmd_pub._last_right_joints = right_joints
                                cmd_pub._last_right_gripper = right_gripper_raw
                            
                            # 更新前一帧
                            prev_action = action
                            prev_left_joints = left_joints_arr
                            prev_right_joints = right_joints_arr
                            
                            # 动作间短暂延迟（避免发送过快）
                            if action_idx < execute_steps - 1:
                                time.sleep(0.05)  # 50ms
                        else:
                            skipped_count += 1
                            print(f"    ⏭️ SKIPPED by user")
                    
                    # 执行完毕统计
                    print(f"\n" + "="*60)
                    print(f"✅ STREAMING EXECUTION COMPLETE")
                    print(f"    Executed: {executed_count}/{execute_steps}")
                    print(f"    Skipped:  {skipped_count}/{execute_steps}")
                    if ik_fail_count > 0:
                        print(f"    ⚠️ IK failures: {ik_fail_count} (used fallback)")
                    print("="*60)
                    
                    # 等待机器人执行完动作
                    if executed_count > 0 and args.execution_delay > 0:
                        print(f"[Main] ⏳ Waiting {args.execution_delay}s for robot to execute actions...")
                        time.sleep(args.execution_delay)
                        print(f"[Main] ✅ Execution delay complete, ready for next prediction")
                    
                    if executed_count == 0:
                        print(f"\n[Main] ❌ No commands executed - all actions skipped or failed")
        finally:
            zmq_sub.close()
            if cmd_pub is not None:
                cmd_pub.close()
    
    elif args.bag_file:
        # 从 rosbag 文件读取数据
        print(f"\n[Main] Running in BAG FILE mode: {args.bag_file}")
        if args.compare_gt_action:
            print("[Main] Ground truth comparison ENABLED")
        if args.show_joint_delta:
            print("[Main] Joint delta display ENABLED")
        
        if not ROSBAG_AVAILABLE:
            print("[Error] rosbag is not available. Install with: pip install rosbag")
            return
        
        bag_reader = RosbagDataReader(args.bag_file)
        n_frames = min(args.n_iterations, len(bag_reader) - 1)  # 保留最后一帧用于 GT
        
        # 统计累计误差
        total_errors = {
            'left_pos': [], 'right_pos': [],
            'left_euler': [], 'right_euler': [],
            'left_gripper': [], 'right_gripper': []
        }
        
        for step in range(n_frames):
            current_idx = step  # 当前帧索引
            
            print(f"\n{'='*60}")
            print(f"Frame {step + 1}/{n_frames}")
            print("="*60)
            
            # 获取 bag 数据
            head_rgb, left_rgb, right_rgb, arm_left, arm_right, gripper_left, gripper_right = bag_reader.get_frame(current_idx)
            
            if head_rgb is None:
                print("[Main] No head RGB image, skipping...")
                continue
            
            # 处理缺失的腕部图像
            if left_rgb is None:
                left_rgb = np.zeros_like(head_rgb)
            if right_rgb is None:
                right_rgb = np.zeros_like(head_rgb)
            
            # 计算 Ground Truth action（下一帧的 eepose）
            gt_action = None
            if args.compare_gt_action:
                gt_action = bag_reader.compute_gt_action(current_idx, ik_solver_left, ik_solver_right)
            
            # 运行推理
            actions = controller.run_single_inference(
                head_rgb, left_rgb, right_rgb,
                arm_left, arm_right, gripper_left, gripper_right,
                args.task_prompt,
                print_only=(not args.compute_ik),
                show_joint_delta=args.show_joint_delta,
                gt_action=gt_action,
                compare_gt=args.compare_gt_action
            )
            
            # 收集误差统计
            if args.compare_gt_action and gt_action is not None and len(actions) > 0:
                pred_action = actions[0]
                total_errors['left_pos'].append(np.linalg.norm(pred_action[:3] - gt_action[:3]))
                total_errors['right_pos'].append(np.linalg.norm(pred_action[7:10] - gt_action[7:10]))
                total_errors['left_euler'].append(np.linalg.norm(pred_action[3:6] - gt_action[3:6]))
                total_errors['right_euler'].append(np.linalg.norm(pred_action[10:13] - gt_action[10:13]))
                total_errors['left_gripper'].append(abs(pred_action[6] - gt_action[6]))
                total_errors['right_gripper'].append(abs(pred_action[13] - gt_action[13]))
        
        # 打印总体统计
        if args.compare_gt_action and total_errors['left_pos']:
            print(f"\n{'='*60}")
            print("Overall Error Statistics")
            print("="*60)
            print(f"  Left Position Error:  mean={np.mean(total_errors['left_pos']):.6f}m, std={np.std(total_errors['left_pos']):.6f}m")
            print(f"  Right Position Error: mean={np.mean(total_errors['right_pos']):.6f}m, std={np.std(total_errors['right_pos']):.6f}m")
            print(f"  Left Euler Error:     mean={np.mean(total_errors['left_euler']):.6f}rad, std={np.std(total_errors['left_euler']):.6f}rad")
            print(f"  Right Euler Error:    mean={np.mean(total_errors['right_euler']):.6f}rad, std={np.std(total_errors['right_euler']):.6f}rad")
            print(f"  Left Gripper Error:   mean={np.mean(total_errors['left_gripper']):.4f}, std={np.std(total_errors['left_gripper']):.4f}")
            print(f"  Right Gripper Error:  mean={np.mean(total_errors['right_gripper']):.4f}, std={np.std(total_errors['right_gripper']):.4f}")
            print(f"  Total Position Error: mean={np.mean(total_errors['left_pos']) + np.mean(total_errors['right_pos']):.6f}m")
        
        bag_reader.close()
    
    elif args.ros_mode:
        print("\n[Main] Running in ROS mode")
        ros_sub = ROSDataSubscriber()
        
        for step in range(args.n_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {step + 1}/{args.n_iterations}")
            print("="*60)
            
            # 获取 ROS 数据
            head_rgb, left_rgb, right_rgb = ros_sub.get_images()
            arm_left, arm_right, gripper_left, gripper_right = ros_sub.get_robot_state()
            
            if head_rgb is None:
                print("[Main] No head RGB image, skipping...")
                time.sleep(0.5)
                continue
            
            # 处理缺失的腕部图像
            if left_rgb is None:
                left_rgb = np.zeros_like(head_rgb)
            if right_rgb is None:
                right_rgb = np.zeros_like(head_rgb)
            
            # 运行推理
            actions = controller.run_single_inference(
                head_rgb, left_rgb, right_rgb,
                arm_left, arm_right, gripper_left, gripper_right,
                args.task_prompt,
                print_only=(not args.compute_ik)
            )
            
            time.sleep(0.5)
    
    elif args.dummy_mode:
        print("\n[Main] Running in DUMMY mode")
        
        # 虚拟关节角度
        arm_left = np.array([0.0, 0.5, -0.3, 0.2, 0.1, -0.2])
        arm_right = np.array([0.0, 0.5, -0.3, -0.2, 0.1, 0.2])
        gripper_left = np.array([0.5])
        gripper_right = np.array([0.5])
        
        for step in range(args.n_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {step + 1}/{args.n_iterations}")
            print("="*60)
            
            # 创建虚拟图像
            head_rgb, left_rgb, right_rgb = create_dummy_images()
            
            # 运行推理
            actions = controller.run_single_inference(
                head_rgb, left_rgb, right_rgb,
                arm_left, arm_right, gripper_left, gripper_right,
                args.task_prompt,
                print_only=(not args.compute_ik)
            )
            
            time.sleep(0.5)
    
    else:
        print("\n[Main] Please specify --bag_file, --ros_mode, or --dummy_mode")
        return
    
    print("\n[Main] Test completed!")


if __name__ == "__main__":
    main()
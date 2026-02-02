"""
PI0 测试代码 - 集成逆运动学（IK）
基于真机代码算法，输出末端位置并通过IK计算关节角度

Author: Based on deploy_pi0_R1.py
"""

import os
import sys
import numpy as np
import torch
import time
import traceback
from PIL import Image
from scipy.spatial.transform import Rotation as R
import kinpy as kp
from scipy.optimize import minimize
import argparse
import yaml
import cv2
import threading

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

# 添加必要的路径
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)
sys.path.append(os.path.join(parent_directory, "../RoboTwin"))
sys.path.append(os.path.join(parent_directory, "../RoboTwin/policy/pi0"))

# 导入 PI0 相关模块
from RoboTwin.policy.pi0.pi_model import PI0


class IKSolver:
    """逆运动学求解器 - 基于 h52eepose.py 的 FK 逻辑"""
    
    # URDF 路径
    URDF_PATH = "/home/pine/yzj/R1_urdf/galaxea_sim/assets/r1/robot.urdf"
    
    # 固定的 torso 值（与 h52eepose.py 保持一致）
    TORSO_FIXED = np.array([0.25, -0.4, -0.85, 0], dtype=np.float32)
    
    # 运动学链配置
    CHAIN_CONFIG = {
        'left': {
            'root_link': 'base_link',
            'end_link': 'left_gripper_link',
            'joint_names': [
                'torso_joint1', 'torso_joint2', 'torso_joint3', 'torso_joint4',
                'left_arm_joint1', 'left_arm_joint2', 'left_arm_joint3', 
                'left_arm_joint4', 'left_arm_joint5', 'left_arm_joint6',
            ]
        },
        'right': {
            'root_link': 'base_link',
            'end_link': 'right_gripper_link',
            'joint_names': [
                'torso_joint1', 'torso_joint2', 'torso_joint3', 'torso_joint4',
                'right_arm_joint1', 'right_arm_joint2', 'right_arm_joint3', 
                'right_arm_joint4', 'right_arm_joint5', 'right_arm_joint6',
            ]
        }
    }
    
    def __init__(self, side='left'):
        """初始化 IK 求解器
        
        Args:
            side: 'left' 或 'right'，指定左臂或右臂
        """
        self.side = side
        
        # 加载 URDF
        try:
            with open(self.URDF_PATH, 'rb') as f:
                urdf_data = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"URDF file not found at {self.URDF_PATH}")
        
        # 构建运动学链
        cfg = self.CHAIN_CONFIG[side]
        self.chain = kp.build_serial_chain_from_urdf(
            urdf_data, 
            cfg['end_link'], 
            cfg['root_link']
        )
        self.joint_names = cfg['joint_names']
        
        print(f"[IKSolver] Initialized for {side} arm with {len(self.joint_names)} joints")
    
    def forward_kinematics(self, joint_angles):
        """正运动学：从关节角度计算末端位置和姿态
        
        Args:
            joint_angles: 6个手臂关节角度（不包含 torso）
            
        Returns:
            pos: 3D 位置 [x, y, z]
            quat: 四元数 [w, x, y, z]
            euler: 欧拉角 [roll, pitch, yaw]
        """
        # 拼接 torso 和 arm 关节角度
        joint_angles_full = np.concatenate([self.TORSO_FIXED, joint_angles])
        
        # 构建关节字典
        th = dict(zip(self.joint_names, joint_angles_full))
        
        # 计算正运动学
        transform = self.chain.forward_kinematics(th)
        
        pos = transform.pos
        quat = transform.rot  # [w, x, y, z]
        
        # 计算欧拉角
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy uses [x,y,z,w]
        euler = r.as_euler('zyx')
        
        return pos, quat, euler
    
    def inverse_kinematics(self, target_pos, target_euler=None, initial_guess=None, 
                          max_iterations=100, tolerance=1e-4):
        """逆运动学：从末端位置（和可选的姿态）计算关节角度
        
        Args:
            target_pos: 目标位置 [x, y, z]
            target_euler: 目标欧拉角 [roll, pitch, yaw]，可选
            initial_guess: 初始关节角度猜测（6个），如果为 None 则使用零位
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
            
        Returns:
            joint_angles: 6个手臂关节角度
            success: 是否成功求解
        """
        # 初始猜测
        if initial_guess is None:
            x0 = np.zeros(6)
        else:
            x0 = np.array(initial_guess)
        
        # 目标位置
        target_pos = np.array(target_pos)
        
        # 定义优化目标函数
        def objective(joint_angles):
            pos, quat, euler = self.forward_kinematics(joint_angles)
            
            # 位置误差
            pos_error = np.linalg.norm(pos - target_pos)
            
            # 如果指定了目标姿态，则加入姿态误差
            if target_euler is not None:
                euler_error = np.linalg.norm(euler - np.array(target_euler))
                return pos_error + 0.1 * euler_error  # 姿态权重较小
            else:
                return pos_error
        
        # 关节限制（示例，可根据实际机器人调整）
        bounds = [(-np.pi, np.pi) for _ in range(6)]
        
        # 使用优化求解
        result = minimize(
            objective, 
            x0, 
            method='L-BFGS-B',  # 或 'SLSQP'
            bounds=bounds,
            options={'maxiter': max_iterations, 'ftol': tolerance}
        )
        
        success = result.success and result.fun < tolerance * 10
        
        if not success:
            print(f"[IKSolver] Warning: IK did not converge well. Error: {result.fun:.6f}")
        
        return result.x, success


class PI0TestController:
    """PI0 测试控制器 - 集成模型推理和 IK 计算"""
    
    def __init__(self, model_config, controller=None, side='left'):
        """初始化控制器
        
        Args:
            model_config: PI0 模型配置字典
            controller: 机器人控制器（如果有）
            side: 'left' 或 'right'
        """
        self.side = side
        self.controller = controller
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 初始化 PI0 模型
        print("[Controller] Loading PI0 model...")
        self.model = PI0(
            model_config['train_config_name'],
            model_config['model_name'],
            model_config['checkpoint_id'],
            model_config['pi0_step']
        )
        print("[Controller] PI0 model loaded successfully")
        
        # 初始化 IK 求解器
        self.ik_solver = IKSolver(side=side)
        
        # 初始关节角度（根据左右手设置）
        if side == 'left':
            # 左手初始关节角度
            self.current_joint_angles = np.array([
                -0.16744680851063828, 
                2.0108510638297874, 
                -0.6593617021276595, 
                2.002127659574468, 
                0.39382978723404255, 
                -1.7193617021276595
            ])
            # 左手初始夹爪值
            self.initial_gripper = 1.0
        else:  # right
            # 右手初始关节角度
            self.current_joint_angles = np.array([
                0.19234042553191488, 
                1.8925531914893616, 
                -0.6874468085106383, 
                -1.6057446808510638, 
                -0.10148936170212766, 
                1.3085106382978724
            ])
            # 右手初始夹爪值
            self.initial_gripper = 1.0
        
        # 固定的 torso 值
        self.torso_fixed = IKSolver.TORSO_FIXED
        
        # Set numpy print precision
        np.set_printoptions(precision=4, suppress=True)
    
    def reset_model(self):
        """重置模型的观察窗口"""
        if hasattr(self.model, 'reset_obsrvationwindows'):
            self.model.reset_obsrvationwindows()
        elif hasattr(self.model, 'observation_window'):
            self.model.observation_window = None
    
    def prepare_observation(self, main_image, wrist_image, current_state, prompt):
        """准备观察数据（模拟 RoboTwin 的格式）
        
        Args:
            main_image: 主相机图像（PIL Image）
            wrist_image: 腕部相机图像（PIL Image）
            current_state: 当前状态 [x, y, z, roll, pitch, yaw, gripper]
            prompt: 任务提示文本
            
        Returns:
            observation: 观察字典
        """
        # 转换为 numpy 数组
        main_rgb = np.asarray(main_image, dtype=np.uint8)
        wrist_rgb = np.asarray(wrist_image, dtype=np.uint8)
        
        # 构建观察字典（参考 deploy_policy.py 的格式）
        observation = {
            "observation": {
                "head_camera": {"rgb": main_rgb},
                "right_camera": {"rgb": wrist_rgb},  # 或 left_camera
                "left_camera": {"rgb": wrist_rgb},   # 占位
            },
            "joint_action": {
                "vector": current_state
            }
        }
        
        return observation
    
    def encode_obs(self, observation):
        """编码观察数据（参考 deploy_policy.py）"""
        input_rgb_arr = [
            observation["observation"]["head_camera"]["rgb"],
            observation["observation"]["right_camera"]["rgb"],
            observation["observation"]["left_camera"]["rgb"],
        ]
        input_state = observation["joint_action"]["vector"]
        
        return input_rgb_arr, input_state
    
    def get_action_from_model(self, observation, prompt):
        """从模型获取动作
        
        Args:
            observation: 观察字典
            prompt: 任务提示
            
        Returns:
            actions: 动作数组 [N, action_dim]
        """
        # 设置语言指令（如果是第一次）
        if self.model.observation_window is None:
            self.model.set_language(prompt)
        
        # 编码观察
        input_rgb_arr, input_state = self.encode_obs(observation)
        
        # 更新观察窗口
        self.model.update_observation_window(input_rgb_arr, input_state)
        
        # 获取动作
        actions = self.model.get_action()[:self.model.pi0_step]
        
        return np.array(actions)
    
    def action_to_joint_angles(self, action, use_ik=True):
        """将动作转换为关节角度
        
        Args:
            action: 模型输出的动作（假设是末端位置 [x, y, z, ...] 格式）
            use_ik: 是否使用 IK，False 则假设动作直接是关节角度
            
        Returns:
            joint_angles: 6个关节角度
            gripper: 夹爪位置
            success: IK 是否成功（如果使用 IK）
        """
        if use_ik:
            # 假设 action 格式: [x, y, z, roll, pitch, yaw, gripper] 或 [x, y, z, qw, qx, qy, qz, gripper]
            if len(action) >= 7:
                target_pos = action[:3]
                gripper = action[-1]
                
                # 判断是欧拉角还是四元数
                if len(action) == 7:
                    # [x, y, z, roll, pitch, yaw, gripper]
                    target_euler = action[3:6]
                elif len(action) == 8:
                    # [x, y, z, qw, qx, qy, qz, gripper] - 转换四元数到欧拉角
                    quat = action[3:7]  # [qw, qx, qy, qz]
                    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy: [x,y,z,w]
                    target_euler = r.as_euler('zyx')
                else:
                    target_euler = None
                
                # 使用 IK 求解
                joint_angles, success = self.ik_solver.inverse_kinematics(
                    target_pos, 
                    target_euler=target_euler,
                    initial_guess=self.current_joint_angles
                )
                
                return joint_angles, gripper, success
            else:
                raise ValueError(f"Action dimension {len(action)} not supported for IK")
        else:
            # 假设动作直接是关节角度
            joint_angles = action[:6]
            gripper = action[6] if len(action) > 6 else 0.0
            return joint_angles, gripper, True
    
    def execute_joint_action(self, joint_angles, gripper):
        """执行关节角度命令
        
        Args:
            joint_angles: 6个关节角度
            gripper: 夹爪位置
        """
        if self.controller is not None:
            # 如果有控制器，执行动作
            # 构建完整的命令 [torso(4) + arm(6) + gripper(1)]
            full_command = np.concatenate([
                self.torso_fixed,
                joint_angles,
                [gripper]
            ])
            
            # 调用控制器执行
            # 具体接口取决于你的控制器实现
            self.controller.execute_joints(full_command, self.side)
        else:
            # 仅打印（模拟执行）
            print(f"[Execute] Joint angles: {joint_angles}")
            print(f"[Execute] Gripper: {gripper}")
        
        # 更新当前状态
        self.current_joint_angles = joint_angles
    
    def run_test_loop(self, task_prompt, get_observation_fn, n_iterations=50, 
                     chunk_size=10, use_ik=True):
        """运行测试循环
        
        Args:
            task_prompt: 任务提示文本
            get_observation_fn: 获取观察的函数，返回 (main_image, wrist_image)
            n_iterations: 最大迭代次数
            chunk_size: 每次执行的动作数量
            use_ik: 是否使用 IK（True 表示模型输出末端位置，False 表示输出关节角度）
        """
        print(f"\n[Controller] Starting test loop for task: {task_prompt}")
        print(f"[Controller] Use IK: {use_ik}")
        print(f"[Controller] Initial joint angles: {self.current_joint_angles}")
        
        # 重置模型
        self.reset_model()
        
        # 初始状态（末端位置）
        init_pos, init_quat, init_euler = self.ik_solver.forward_kinematics(self.current_joint_angles)
        init_gripper = self.initial_gripper  # 使用设置的初始夹爪值
        
        print(f"[Controller] Initial end-effector position: {init_pos}")
        print(f"[Controller] Initial end-effector euler: {init_euler}")
        print(f"[Controller] Initial gripper: {init_gripper}")
        
        step = 0
        
        try:
            while step < n_iterations:
                print(f"\n--- Step {step + 1}/{n_iterations} ---")
                
                # 获取观察
                main_image, wrist_image = get_observation_fn()
                
                if main_image is None or wrist_image is None:
                    print("[Controller] Failed to get observation, skipping")
                    time.sleep(0.1)
                    continue
                
                # 计算当前末端状态（用于输入模型）
                current_pos, current_quat, current_euler = self.ik_solver.forward_kinematics(
                    self.current_joint_angles
                )
                current_state = np.concatenate([current_pos, current_euler, [init_gripper]])
                
                # 准备观察
                observation = self.prepare_observation(
                    main_image, wrist_image, current_state, task_prompt
                )
                
                # 从模型获取动作
                actions = self.get_action_from_model(observation, task_prompt)
                print(f"[Controller] Received {len(actions)} actions from model")
                
                # 截取要执行的动作
                actions_to_execute = actions[:chunk_size]
                
                # 执行每个动作
                for i, action in enumerate(actions_to_execute):
                    print(f"\n  Action {i+1}/{len(actions_to_execute)}: {action}")
                    
                    # 转换为关节角度
                    joint_angles, gripper, success = self.action_to_joint_angles(action, use_ik=use_ik)
                    
                    if not success:
                        print(f"  [Warning] IK failed for action {i+1}, using best effort")
                    
                    print(f"  Joint angles: {joint_angles}")
                    print(f"  Gripper: {gripper}")
                    
                    # 执行
                    self.execute_joint_action(joint_angles, gripper)
                    
                    # 更新模型观察（如果需要）
                    # 这里简化处理，只在 chunk 结束后更新
                    time.sleep(0.05)  # 小延迟
                
                step += 1
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n[Controller] Test loop interrupted by user")
        except Exception as e:
            print(f"\n[Controller] Error occurred: {e}")
            traceback.print_exc()
        finally:
            print("[Controller] Test loop ended")


class ROSDataSubscriber:
    """ROS数据订阅器 - 订阅相机和机器人状态topics"""
    
    def __init__(self, side='left'):
        """初始化ROS订阅器
        
        Args:
            side: 'left' 或 'right'，指定使用哪只手的腕部相机
        """
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS is not available. Cannot use ROS mode.")
        
        self.side = side
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
        rospy.init_node('pi0_test_node', anonymous=True, disable_signals=True)
        
        # 订阅相机topics
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
        
        # 订阅机器人状态topics
        print("[ROSSubscriber] Subscribing to robot state topics...")
        rospy.Subscriber("/hdas/feedback_arm_left", 
                        Float64MultiArray, self._arm_left_callback)
        rospy.Subscriber("/hdas/feedback_arm_right", 
                        Float64MultiArray, self._arm_right_callback)
        rospy.Subscriber("/hdas/feedback_gripper_left", 
                        Float64MultiArray, self._gripper_left_callback)
        rospy.Subscriber("/hdas/feedback_gripper_right", 
                        Float64MultiArray, self._gripper_right_callback)
        
        # 等待数据
        print("[ROSSubscriber] Waiting for initial data...")
        self._wait_for_data()
        print("[ROSSubscriber] Ready!")
    
    def _head_rgb_callback(self, msg):
        """头部RGB相机回调"""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                # Resize到640x480
                if img_bgr.shape[1] != 640 or img_bgr.shape[0] != 480:
                    img_bgr = cv2.resize(img_bgr, (640, 480), interpolation=cv2.INTER_LINEAR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.head_rgb = Image.fromarray(img_rgb)
        except Exception as e:
            print(f"[ROSSubscriber] Error decoding head RGB: {e}")
    
    def _head_depth_callback(self, msg):
        """头部深度相机回调"""
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # Resize到640x480（深度图用最近邻插值）
            if depth_img.shape[1] != 640 or depth_img.shape[0] != 480:
                depth_img = cv2.resize(depth_img, (640, 480), interpolation=cv2.INTER_NEAREST)
            with self.lock:
                self.head_depth = depth_img
        except Exception as e:
            print(f"[ROSSubscriber] Error decoding head depth: {e}")
    
    def _left_rgb_callback(self, msg):
        """左腕RGB相机回调"""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                if img_bgr.shape[1] != 640 or img_bgr.shape[0] != 480:
                    img_bgr = cv2.resize(img_bgr, (640, 480), interpolation=cv2.INTER_LINEAR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.left_wrist_rgb = Image.fromarray(img_rgb)
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
        """右腕RGB相机回调"""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                if img_bgr.shape[1] != 640 or img_bgr.shape[0] != 480:
                    img_bgr = cv2.resize(img_bgr, (640, 480), interpolation=cv2.INTER_LINEAR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.right_wrist_rgb = Image.fromarray(img_rgb)
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
        raise RuntimeError("Timeout waiting for ROS data")
    
    def get_observation(self):
        """获取当前观察数据
        
        Returns:
            main_image: 头部RGB图像（PIL Image）
            wrist_image: 腕部RGB图像（PIL Image，根据side选择）
        """
        with self.lock:
            main_image = self.head_rgb
            if self.side == 'left':
                wrist_image = self.left_wrist_rgb
            else:
                wrist_image = self.right_wrist_rgb
        
        return main_image, wrist_image
    
    def get_current_joint_state(self):
        """获取当前关节状态
        
        Returns:
            arm_pos: 手臂关节位置（6个）
            gripper_pos: 夹爪位置
        """
        with self.lock:
            if self.side == 'left':
                arm_pos = self.arm_left_pos[:6] if self.arm_left_pos is not None else None
                gripper_pos = self.gripper_left_pos[0] if self.gripper_left_pos is not None else None
            else:
                arm_pos = self.arm_right_pos[:6] if self.arm_right_pos is not None else None
                gripper_pos = self.gripper_right_pos[0] if self.gripper_right_pos is not None else None
        
        return arm_pos, gripper_pos


def create_dummy_observation():
    """创建虚拟观察（用于测试）"""
    # 创建虚拟图像
    main_image = Image.new('RGB', (640, 480), color=(100, 100, 100))
    wrist_image = Image.new('RGB', (320, 240), color=(50, 50, 50))
    return main_image, wrist_image


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PI0 Test with IK")
    parser.add_argument("--config", type=str, default="policy/pi0/deploy_policy.yml",
                       help="Path to config file")
    parser.add_argument("--train_config_name", type=str, required=True,
                       help="Training config name")
    parser.add_argument("--model_name", type=str, required=True,
                       help="Model name")
    parser.add_argument("--checkpoint_id", type=str, default="latest",
                       help="Checkpoint ID")
    parser.add_argument("--pi0_step", type=int, default=10,
                       help="Number of steps to predict")
    parser.add_argument("--task_prompt", type=str, default="Pick up the object",
                       help="Task prompt/instruction")
    parser.add_argument("--n_iterations", type=int, default=50,
                       help="Number of test iterations")
    parser.add_argument("--chunk_size", type=int, default=10,
                       help="Chunk size for action execution")
    parser.add_argument("--use_ik", action="store_true", default=True,
                       help="Use IK to convert end-effector positions to joint angles")
    parser.add_argument("--side", type=str, default="left", choices=["left", "right"],
                       help="Which arm to control")
    parser.add_argument("--dummy_mode", action="store_true",
                       help="Run in dummy mode without real camera/robot")
    parser.add_argument("--ros_mode", action="store_true",
                       help="Use ROS topics for observation (requires ROS and running topics)")
    
    args = parser.parse_args()
    
    # 模型配置
    model_config = {
        'train_config_name': args.train_config_name,
        'model_name': args.model_name,
        'checkpoint_id': args.checkpoint_id,
        'pi0_step': args.pi0_step
    }
    
    # 初始化控制器
    controller = PI0TestController(
        model_config=model_config,
        controller=None,  # 如果有真实控制器，在这里传入
        side=args.side
    )
    
    # 定义观察获取函数
    if args.dummy_mode:
        print("[Main] Running in DUMMY mode")
        get_observation_fn = create_dummy_observation
    elif args.ros_mode:
        # ROS模式：从ROS topics获取数据
        print("[Main] Running in ROS mode")
        if not ROS_AVAILABLE:
            print("[Error] ROS is not available. Please install rospy and cv_bridge.")
            return
        
        ros_subscriber = ROSDataSubscriber(side=args.side)
        get_observation_fn = ros_subscriber.get_observation
    else:
        # RealSense相机模式
        print("[Main] Running in RealSense mode")
        try:
            from pyrealsense_image import initialize_camera, get_L515_image, get_D435_image
            
            pipelines, align_objects = initialize_camera()
            
            def get_real_observation():
                main_image = get_L515_image(pipelines)
                wrist_image = get_D435_image(pipelines)
                return main_image, wrist_image
            
            get_observation_fn = get_real_observation
        except ImportError:
            print("[Error] pyrealsense_image not available. Use --dummy_mode or --ros_mode")
            return
    
    # 运行测试循环
    controller.run_test_loop(
        task_prompt=args.task_prompt,
        get_observation_fn=get_observation_fn,
        n_iterations=args.n_iterations,
        chunk_size=args.chunk_size,
        use_ik=args.use_ik
    )
    
    print("[Main] Test completed")


if __name__ == "__main__":
    main()

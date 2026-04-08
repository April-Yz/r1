"""
PI0/PI0.5 Robot Controller for R1 Dual-Arm - Server-Client Mode

Uses:
  - serve_policy.py as the inference server (WebSocket)
  - ros_bridge.py for sensor data (ZMQ sub) and robot control (ZMQ pub)
  - IK solver (curobo) for FK/IK computation

Usage:
  # Terminal 1: Start the ROS bridge (system Python)
  /usr/bin/python3 /home/pine/yzj/src/ros_bridge.py

  # Terminal 2: Start the policy server (openpi venv)
  cd /home/pine/yzj/openpi
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 uv run scripts/serve_policy.py \\
      policy:checkpoint \\
      --policy.config pi05_zaijia_0215 \\
      --policy.dir /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/pi05_pour_50_0215/29999 \\
      --default_prompt "pour"

  # Terminal 3: Run this controller
  python deploy_pi0_1030.py --task_prompt "pour" --n_iterations 20

Based on: test_pi0_ros.py, deploy_pi0_R1.py
"""

import time
import argparse
import numpy as np
import os
import sys
import signal
import threading

from scipy.spatial.transform import Rotation as R

# WebSocket client for serve_policy.py
try:
    from openpi_client import websocket_client_policy
    WEBSOCKET_AVAILABLE = True
except ImportError:
    print("[Warning] openpi_client not available. Install with: pip install openpi-client")
    WEBSOCKET_AVAILABLE = False

# ZMQ for ros_bridge.py communication
try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    print("[Warning] zmq not available. Install with: pip install pyzmq")
    ZMQ_AVAILABLE = False

# Curobo for FK/IK
try:
    from urdfik import URDFInverseKinematics
    CUROBO_AVAILABLE = True
except ImportError:
    print("[Warning] curobo/urdfik not available. IK functionality disabled.")
    CUROBO_AVAILABLE = False

# OpenCV for image processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("[Warning] cv2 not available.")
    CV2_AVAILABLE = False

# Optional video writer
try:
    import imageio
    _HAS_IMAGEIO = True
except Exception:
    imageio = None
    _HAS_IMAGEIO = False

np.set_printoptions(precision=4, suppress=True)


# ==============================================================================
# IK Solver
# ==============================================================================

class IKSolver:
    """逆运动学求解器 - 基于 Curobo"""

    URDF_PATH = "/home/pine/yzj/R1_urdf/galaxea_sim/assets/r1/robot.urdf"
    TORSO_FIXED = np.array([0.25, -0.4, -0.85, 0], dtype=np.float32)

    def __init__(self, side='left'):
        if not CUROBO_AVAILABLE:
            raise RuntimeError("curobo/urdfik not available")
        self.side = side
        ee_link = 'left_gripper_link' if side == 'left' else 'right_gripper_link'
        self.solver = URDFInverseKinematics(
            urdf_file=self.URDF_PATH,
            base_link='base_link',
            ee_link=ee_link
        )
        print(f"[IKSolver] Initialized for {side} arm (ee_link={ee_link})")

    def forward_kinematics(self, joint_angles):
        """FK: 6 joint angles -> (pos, quat_wxyz, euler_zyx)"""
        full_joints = np.concatenate([self.TORSO_FIXED, joint_angles])
        pos, quat, euler = self.solver.forward_kinematics(full_joints)
        return pos, quat, euler

    def inverse_kinematics(self, target_pos, target_euler=None, initial_guess=None):
        """IK: target pos + euler -> 6 joint angles
        
        Args:
            target_pos: [x, y, z]
            target_euler: [roll, pitch, yaw] in zyx order
            initial_guess: 6 joint angles (optional)
            
        Returns:
            joint_angles: (6,) ndarray
            success: bool
        """
        if target_euler is None:
            return np.zeros(6), False
        try:
            r = R.from_euler('zyx', target_euler)
            quat_xyzw = r.as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

            current_full = np.concatenate([self.TORSO_FIXED, initial_guess]) if initial_guess is not None else None

            result = self.solver.solve_ik(
                target_position=target_pos,
                target_orientation=quat_wxyz,
                current_joints=current_full
            )

            if result is not None and result.success.cpu().numpy().all():
                solution = result.solution.cpu().numpy().flatten()
                if len(solution) < 4:
                    return np.zeros(6), False
                arm_joints = solution[4:]  # skip torso
                if len(arm_joints) < 6:
                    return np.zeros(6), False
                return arm_joints[:6], True
            else:
                return np.zeros(6), False
        except Exception as e:
            print(f"[IKSolver] Error: {e}")
            return np.zeros(6), False


# ==============================================================================
# ZMQ Data Subscriber (from ros_bridge.py)
# ==============================================================================

class ZMQDataSubscriber:
    """通过 ZMQ 从 ros_bridge.py 接收传感器数据"""

    def __init__(self, host="localhost", port=5555, timeout_ms=5000):
        if not ZMQ_AVAILABLE:
            raise RuntimeError("zmq not available")

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        try:
            self.socket.setsockopt(zmq.CONFLATE, 1)
        except:
            pass
        self.socket.connect(f"tcp://{host}:{port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)

        self.latest_data = None
        self.data_lock = threading.Lock()
        self.running = True
        self.last_receive_time = 0
        self.receive_thread = threading.Thread(target=self._run_receive_loop, daemon=True)
        self.receive_thread.start()
        print(f"[ZMQSub] Connected to {host}:{port}")

    def _run_receive_loop(self):
        import pickle
        while self.running:
            try:
                data_bytes = self.socket.recv()
                data = pickle.loads(data_bytes)

                head_rgb = self._decode_image(data.get('head_rgb'))
                left_rgb = self._decode_image(data.get('left_rgb'))
                right_rgb = self._decode_image(data.get('right_rgb'))
                arm_left = np.array(data['arm_left']) if data.get('arm_left') is not None else None
                arm_right = np.array(data['arm_right']) if data.get('arm_right') is not None else None
                gripper_left = np.array(data['gripper_left']) if data.get('gripper_left') is not None else None
                gripper_right = np.array(data['gripper_right']) if data.get('gripper_right') is not None else None

                with self.data_lock:
                    self.latest_data = (head_rgb, left_rgb, right_rgb,
                                        arm_left, arm_right, gripper_left, gripper_right)
                self.last_receive_time = time.time()
            except zmq.Again:
                continue
            except Exception as e:
                if self.running:
                    print(f"[ZMQSub] Error: {e}")
                time.sleep(0.1)

    def _decode_image(self, img_bytes):
        if img_bytes is None or not CV2_AVAILABLE:
            return None
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            if img_bgr.shape[1] != 640 or img_bgr.shape[0] != 480:
                img_bgr = cv2.resize(img_bgr, (640, 480))
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return None

    def receive(self):
        with self.data_lock:
            return self.latest_data

    def close(self):
        self.running = False
        time.sleep(0.1)
        self.socket.close()
        self.context.term()


# ==============================================================================
# ZMQ Command Publisher (to ros_bridge.py)
# ==============================================================================

class ZMQCommandPublisher:
    """通过 ZMQ 向 ros_bridge.py 发送控制命令"""

    # 抬手位置（安全中间位置）
    LIFT_ARM_LEFT_JOINTS = np.array([
        0.9693617021276596, 0.365531914893617, -0.24425531914893617,
        2.7825531914893618, -0.02021276595744681, 0.17212765957446807
    ], dtype=np.float32)
    LIFT_ARM_RIGHT_JOINTS = np.array([
        -1.0529787234042554, 0.451063829787234, -0.38340425531914896,
        -0.15723404255319148, 0.05127659574468085, -0.02872340425531915
    ], dtype=np.float32)

    # 不同任务的初始关节角度
    # 初始化位置参数
    # 1: pour（胸前，默认）
    # 2: pnp_apple_star（竖直向下）
    TASK_INIT_POSITIONS = {
        1: {  # 胸前 pour
            "left": np.array([
                -0.16744680851063828, 2.0108510638297874, -0.6593617021276595,
                2.002127659574468, 0.39382978723404255, -1.7193617021276595
            ], dtype=np.float32),
            "right": np.array([
                0.19234042553191488, 1.8925531914893616, -0.6874468085106383,
                -1.6057446808510638, -0.10148936170212766, 1.3085106382978724
            ], dtype=np.float32),
            "gripper_left": 100.0,
            "gripper_right": 100.0,
        },
        2: {  # 竖直向下 pnp_apple_star
            "left": np.array([
                -0.6742553191489362, 2.656595744680851, -1.3061702127659574,
                -0.0325531914893617, 1.4363829787234041, -1.2231914893617022,
                -2.753191489361702
            ], dtype=np.float32),
            "right": np.array([
                0.6561702127659574, 2.3187234042553193, -1.1055319148936171,
                0.19127659574468084, -1.4397872340425533, 0.9957446808510638,
                -2.468936170212766
            ], dtype=np.float32),
            "gripper_left": 100.0,
            "gripper_right": 100.0,
        },
    }

    # 默认使用 pour（胸前）的初始位置
    INIT_LEFT_JOINTS = TASK_INIT_POSITIONS[1]["left"]
    INIT_RIGHT_JOINTS = TASK_INIT_POSITIONS[1]["right"]
    INIT_GRIPPER_LEFT = 100.0
    INIT_GRIPPER_RIGHT = 100.0
    INVALID_GRIPPER_VALUE = -2.7

    def __init__(self, host="localhost", port=5556):
        if not ZMQ_AVAILABLE:
            raise RuntimeError("zmq not available")
        import pickle
        self.pickle = pickle

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.connect(f"tcp://{host}:{port}")
        time.sleep(0.5)
        self.cmd_count = 0
        self._last_left_joints = None
        self._last_left_gripper = None
        self._last_right_joints = None
        self._last_right_gripper = None
        print(f"[ZMQCmd] Connected to {host}:{port}")

    def send_command(self, left_joints=None, left_gripper_raw=None,
                     right_joints=None, right_gripper_raw=None):
        """发送控制命令
        
        Args:
            left_joints: 左臂 6 个关节角度 (rad)
            left_gripper_raw: 左夹爪位置 (0-100)
            right_joints: 右臂 6 个关节角度 (rad)
            right_gripper_raw: 右夹爪位置 (0-100)
        """
        def make_7dim(joints):
            if joints is None:
                return None
            j = list(joints)
            if len(j) == 6:
                j.append(self.INVALID_GRIPPER_VALUE)
            return j

        cmd = {
            'arm_left': make_7dim(left_joints),
            'arm_right': make_7dim(right_joints),
            'gripper_left': float(left_gripper_raw) if left_gripper_raw is not None else None,
            'gripper_right': float(right_gripper_raw) if right_gripper_raw is not None else None,
        }
        try:
            self.socket.send(self.pickle.dumps(cmd))
            self.cmd_count += 1
            return True
        except Exception as e:
            print(f"[ZMQCmd] Error: {e}")
            return False

    def send_lift_arm_position(self, wait_for_confirm=True):
        """发送抬手位置（安全位置）"""
        print("\n" + "=" * 60)
        print("STEP 1: LIFT ARMS - MOVE TO SAFE POSITION")
        print("=" * 60)
        if wait_for_confirm:
            while True:
                inp = input("Type 'yes' to LIFT ARMS, 'skip' to skip, 'no' to abort: ").strip().lower()
                if inp == 'yes':
                    break
                elif inp == 'skip':
                    return "skipped"
                elif inp == 'no':
                    return False
        success = self.send_command(
            self.LIFT_ARM_LEFT_JOINTS, self.INIT_GRIPPER_LEFT,
            self.LIFT_ARM_RIGHT_JOINTS, self.INIT_GRIPPER_RIGHT
        )
        if success:
            print("Lift arm command sent. Waiting 3s...")
            time.sleep(3.0)
        return success

    def send_init_position(self, init_type=1, wait_for_confirm=True):
        """发送初始位置（根据参数选择）
        init_type: 1=胸前pour，2=竖直向下pnp，默认1
        """
        if init_type in self.TASK_INIT_POSITIONS:
            init_cfg = self.TASK_INIT_POSITIONS[init_type]
            left_joints = init_cfg["left"]
            right_joints = init_cfg["right"]
            left_gripper = init_cfg.get("gripper_left", 100.0)
            right_gripper = init_cfg.get("gripper_right", 100.0)
            if init_type == 1:
                print("\n  Using INIT_TYPE 1: pour (胸前)")
            elif init_type == 2:
                print("\n  Using INIT_TYPE 2: pnp_apple_star (竖直向下)")
        else:
            left_joints = self.INIT_LEFT_JOINTS
            right_joints = self.INIT_RIGHT_JOINTS
            left_gripper = self.INIT_GRIPPER_LEFT
            right_gripper = self.INIT_GRIPPER_RIGHT
            print(f"\n  Using default init position (invalid init_type={init_type})")

        print("\n" + "=" * 60)
        print("STEP 2: MOVE TO INITIAL POSITION")
        print("=" * 60)
        if wait_for_confirm:
            while True:
                inp = input("Type 'yes' to send INIT position, 'no' to abort: ").strip().lower()
                if inp == 'yes':
                    break
                elif inp == 'no':
                    return False
        success = self.send_command(
            left_joints, left_gripper,
            right_joints, right_gripper
        )
        if success:
            print("Init position command sent. Waiting 2s...")
            time.sleep(2.0)
        return success

    @staticmethod
    def _match_task_key(task_prompt):
        """根据 task_prompt 匹配 TASK_INIT_POSITIONS 的 key"""
        task_lower = task_prompt.lower().strip()
        # 精确匹配
        for key in ZMQCommandPublisher.TASK_INIT_POSITIONS:
            if key == task_lower:
                return key
        # 关键词匹配
        keyword_map = {
            "pour": "pour",
            "apple": "pnp_apple_star",
            "starfruit": "pnp_apple_star",
            "star": "pnp_apple_star",
        }
        for keyword, key in keyword_map.items():
            if keyword in task_lower:
                return key
        return None

    def close(self):
        self.socket.close()
        self.context.term()


# ==============================================================================
# R1 Robot Controller (Server-Client Mode)
# ==============================================================================

class PI0RobotController:
    """R1 双臂 PI0 控制器 - 通过 WebSocket 连接 serve_policy.py

    数据流:
        ros_bridge.py (ZMQ) -> 本控制器 -> serve_policy.py (WebSocket) -> 本控制器 -> ros_bridge.py (ZMQ)
        [传感器数据]         [FK->state]  [模型推理]                     [IK->joints]  [执行命令]
    """

    # 锁定朝向的固定欧拉角 (zyx 顺序)
    LOCKED_EULER_LEFT = np.array([-1.8158525116572453, 1.4325311534153802, 1.6442961973990653], dtype=np.float64)
    LOCKED_EULER_RIGHT = np.array([2.856346276854692, 1.3826080031756822, -2.8279581076651064], dtype=np.float64)

    def __init__(self, websocket_host="localhost", websocket_port=8000,
                 zmq_host="localhost", zmq_data_port=5555, zmq_cmd_port=5556,
                 lock_euler=False, save_video=True, video_dir="./videos"):
        """
        Args:
            websocket_host: serve_policy.py 的主机地址
            websocket_port: serve_policy.py 的端口
            zmq_host: ros_bridge.py 的主机地址
            zmq_data_port: ZMQ 数据端口 (传感器数据)
            zmq_cmd_port: ZMQ 命令端口 (控制命令)
            lock_euler: 是否锁定末端朝向
            save_video: 是否保存视频
            video_dir: 视频保存目录
        """
        # WebSocket client -> serve_policy.py
        print(f"\n[Controller] Connecting to policy server at {websocket_host}:{websocket_port}...")
        self.client = websocket_client_policy.WebsocketClientPolicy(websocket_host, websocket_port)
        print(f"[Controller] WebSocket client connected!")

        # ZMQ data subscriber <- ros_bridge.py
        print(f"[Controller] Connecting to ZMQ data at {zmq_host}:{zmq_data_port}...")
        self.zmq_sub = ZMQDataSubscriber(host=zmq_host, port=zmq_data_port)

        # ZMQ command publisher -> ros_bridge.py
        print(f"[Controller] Connecting to ZMQ cmd at {zmq_host}:{zmq_cmd_port}...")
        self.cmd_pub = ZMQCommandPublisher(host=zmq_host, port=zmq_cmd_port)

        # IK solvers for FK/IK computation
        print("[Controller] Initializing IK solvers...")
        self.ik_solver_left = IKSolver(side='left')
        self.ik_solver_right = IKSolver(side='right')

        # Lock euler mode
        self.lock_euler = lock_euler
        if self.lock_euler:
            print(f"[Controller] LOCK EULER MODE ENABLED")
            print(f"    Left  fixed euler: {self.LOCKED_EULER_LEFT}")
            print(f"    Right fixed euler: {self.LOCKED_EULER_RIGHT}")

        # Current state tracking (for IK initial guess)
        self.current_joint_left = None
        self.current_joint_right = None
        self.current_gripper_left = None
        self.current_gripper_right = None

        # Video saving
        self.save_video = save_video
        self.video_dir = video_dir
        self.video_frames = []

        print("[Controller] Initialization complete!\n")

    def compute_state_vector(self, arm_left, arm_right, gripper_left, gripper_right):
        """通过 FK 计算 14 维 eepose 状态向量

        State format (14 dims, 与 norm_stats 一致):
            [0:3]   left_ee_x, left_ee_y, left_ee_z
            [3:6]   left_ee_roll, left_ee_pitch, left_ee_yaw  (zyx euler)
            [6]     left_gripper (0-1 normalized)
            [7:10]  right_ee_x, right_ee_y, right_ee_z
            [10:13] right_ee_roll, right_ee_pitch, right_ee_yaw (zyx euler)
            [13]    right_gripper (0-1 normalized)
        """
        state = np.zeros(14, dtype=np.float32)

        # Left arm FK
        if arm_left is not None:
            joint_left = arm_left[:6]
            pos, quat, _ = self.ik_solver_left.forward_kinematics(joint_left)
            # kinpy quat: [w,x,y,z] -> scipy: [x,y,z,w]
            r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            euler = r.as_euler('zyx')
            state[0:3] = pos
            state[3:6] = euler
            gripper_raw = gripper_left[0] if gripper_left is not None and len(gripper_left) > 0 else 0.0
            state[6] = float(gripper_raw / 100.0)
            self.current_joint_left = joint_left
            self.current_gripper_left = state[6]

        # Right arm FK
        if arm_right is not None:
            joint_right = arm_right[:6]
            pos, quat, _ = self.ik_solver_right.forward_kinematics(joint_right)
            r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            euler = r.as_euler('zyx')
            state[7:10] = pos
            state[10:13] = euler
            gripper_raw = gripper_right[0] if gripper_right is not None and len(gripper_right) > 0 else 0.0
            state[13] = float(gripper_raw / 100.0)
            self.current_joint_right = joint_right
            self.current_gripper_right = state[13]

        return state

    def prepare_inference_data(self, head_rgb, left_rgb, right_rgb, state, prompt):
        """准备发送给 serve_policy.py 的推理数据

        serve_policy.py 的 create_trained_policy 中，repack_transforms 参数默认为空，
        config 中的 repack_transforms 仅用于训练数据加载，推理时不会应用。
        因此客户端需要直接发送 AlohaInputs 期望的嵌套格式。

        AlohaInputs 期望的输入格式:
            images:
                cam_high:        (C, H, W) uint8
                cam_left_wrist:  (C, H, W) uint8
                cam_right_wrist: (C, H, W) uint8
            state: (14,) float32  (与 norm_stats 维度一致)
            prompt: str
        """
        return {
            "images": {
                "cam_high": np.transpose(head_rgb, (2, 0, 1)).astype(np.uint8),
                "cam_left_wrist": np.transpose(left_rgb, (2, 0, 1)).astype(np.uint8),
                "cam_right_wrist": np.transpose(right_rgb, (2, 0, 1)).astype(np.uint8),
            },
            "state": state,
            "prompt": prompt,
        }

    def action_to_joint_angles(self, action, side='left'):
        """将 eepose action 转换为关节角度

        Action format: [x, y, z, roll, pitch, yaw, gripper] (7D per arm)
        Full action: [left(7), right(7)] = 14D
        """
        if side == 'left':
            solver = self.ik_solver_left
            initial = self.current_joint_left
            a = action[0:7]
        else:
            solver = self.ik_solver_right
            initial = self.current_joint_right
            a = action[7:14]

        target_pos = a[:3]
        target_euler = a[3:6]
        gripper = a[6]

        # 锁定朝向模式
        if self.lock_euler:
            if side == 'left':
                target_euler = self.LOCKED_EULER_LEFT.copy()
            else:
                target_euler = self.LOCKED_EULER_RIGHT.copy()

        joints, success = solver.inverse_kinematics(
            target_pos, target_euler=target_euler, initial_guess=initial
        )
        return joints, gripper, success

    def action_to_robot_command(self, action):
        """将 14D action 转换为机器人命令

        Returns:
            left_joints, left_gripper_raw, right_joints, right_gripper_raw, (left_ok, right_ok)
        """
        left_joints, left_gripper, left_ok = self.action_to_joint_angles(action, 'left')
        right_joints, right_gripper, right_ok = self.action_to_joint_angles(action, 'right')

        left_gripper_raw = left_gripper * 100.0 if left_gripper is not None else None
        right_gripper_raw = right_gripper * 100.0 if right_gripper is not None else None

        return left_joints, left_gripper_raw, right_joints, right_gripper_raw, (left_ok, right_ok)

    def initialize_robot(self, init_type=1, wait_for_confirm=True):
        """机器人初始化流程: 抬手 -> 初始位置 -> 确认
        init_type: 1=胸前pour，2=竖直向下pnp，默认1
        """
        # Step 1: 抬手
        lift_result = self.cmd_pub.send_lift_arm_position(wait_for_confirm=wait_for_confirm)
        if lift_result is False:
            return False

        # Step 2: 初始位置（根据参数选择）
        init_ok = self.cmd_pub.send_init_position(init_type=init_type, wait_for_confirm=wait_for_confirm)
        if not init_ok:
            return False

        # 确认
        print("\n" + "=" * 60)
        print("ROBOT INITIALIZATION COMPLETE")
        print("=" * 60)
        if wait_for_confirm:
            while True:
                inp = input("Type 'yes' to start PI0 control, 'no' to abort: ").strip().lower()
                if inp == 'yes':
                    break
                elif inp == 'no':
                    return False
        return True

    def save_video_file(self, task_name):
        """保存录制的视频帧"""
        if not self.save_video or len(self.video_frames) == 0:
            return
        os.makedirs(self.video_dir, exist_ok=True)

        # 找一个不冲突的文件名
        i = 0
        while True:
            path = os.path.join(self.video_dir, f"{task_name}_{i}.mp4")
            if not os.path.exists(path):
                break
            i += 1

        print(f"[Video] Saving {len(self.video_frames)} frames to {path}...")
        try:
            if _HAS_IMAGEIO:
                writer = imageio.get_writer(path, fps=15)
                for fr in self.video_frames:
                    writer.append_data(fr.astype(np.uint8))
                writer.close()
                print(f"[Video] Saved successfully!")
            else:
                png_dir = path.replace('.mp4', '_frames')
                os.makedirs(png_dir, exist_ok=True)
                for idx, fr in enumerate(self.video_frames):
                    from PIL import Image
                    Image.fromarray(fr.astype(np.uint8)).save(
                        os.path.join(png_dir, f'frame_{idx:05d}.png'))
                print(f"[Video] Saved frames to {png_dir}")
        except Exception as e:
            print(f"[Video] Failed: {e}")

    def check_joint_arrival(self, target_left_joints, target_right_joints, tolerance=0.001):
        """检查机器人关节是否到达目标位置

        从 ZMQ 读取当前关节角度, 与 IK 求解的目标关节角度比较.

        Args:
            target_left_joints: 左臂目标关节角度 (6D)
            target_right_joints: 右臂目标关节角度 (6D)
            tolerance: 关节角度容差 (rad)

        Returns:
            arrived: 是否到位
            left_error: 左臂最大关节误差
            right_error: 右臂最大关节误差
            left_errors: 左臂各关节误差 (6D)
            right_errors: 右臂各关节误差 (6D)
        """
        result = self.zmq_sub.receive()
        if result is None:
            return False, float('inf'), float('inf'), None, None

        _, _, _, arm_left, arm_right, _, _ = result

        left_error = 0.0
        right_error = 0.0
        left_errors = None
        right_errors = None
        left_ok = True
        right_ok = True

        if target_left_joints is not None and arm_left is not None:
            left_errors = np.abs(np.array(target_left_joints[:6]) - np.array(arm_left[:6]))
            left_error = float(np.max(left_errors))
            left_ok = left_error <= tolerance

        if target_right_joints is not None and arm_right is not None:
            right_errors = np.abs(np.array(target_right_joints[:6]) - np.array(arm_right[:6]))
            right_error = float(np.max(right_errors))
            right_ok = right_error <= tolerance

        return left_ok and right_ok, left_error, right_error, left_errors, right_errors

    def run_control_loop(self, task_prompt="pour", n_iterations=20,
                         chunk_size=10, action_index=0, execute_steps=10,
                         execution_delay=0.5, init_robot=True,
                         confirm_each=False, action_as_obs=False,
                         repeat_actions=1, ensemble_size=1,
                         joint_tolerance=0.01, arrival_timeout=1.0):
        """主控制循环

        Args:
            task_prompt: 任务提示 (需与 serve_policy.py 的 --default_prompt 匹配)
            n_iterations: 推理迭代次数
            chunk_size: 每次推理后执行的最大动作数
            action_index: 从第几个 action 开始执行 (0 = 第一个)
            execute_steps: 每次推理执行多少步
            execution_delay: 执行完动作后等待多少秒再进行下一次推理
            init_robot: 是否先初始化机器人位置
            confirm_each: 是否每步确认
            action_as_obs: 是否用上次执行的最后一个 action 作为 state obs
                           (第一次仍从 ZMQ 读取, 图片始终从 ZMQ 读取)
            repeat_actions: 每个动作重复发送的次数 (防止执行精度不足)
            ensemble_size: Action ensemble 大小, 将连续 N 步合并为一步发送
                           (仅在 action_as_obs=True 时生效, 1 = 不合并)
            joint_tolerance: 关节到位检查容差 (rad), 与 ensemble 配合使用
                             (0.01 rad ≈ 0.57°, 建议范围 0.005~0.05)
            arrival_timeout: 等待关节到位的超时时间 (秒), 超时后继续执行
        """
        # 初始化机器人
        if init_robot:
            if not self.initialize_robot(task=task_prompt):
                print("Initialization aborted.")
                return
            time.sleep(0.5)

        print(f"\n{'='*60}")
        print(f"PI0 R1 CONTROL LOOP")
        print(f"  Task: {task_prompt}")
        print(f"  Iterations: {n_iterations}")
        print(f"  Execute steps per inference: {execute_steps}")
        print(f"  Action index start: {action_index}")
        print(f"  Execution delay: {execution_delay}s")
        print(f"  Action as obs: {action_as_obs}")
        print(f"  Repeat actions: {repeat_actions}x")
        if action_as_obs and ensemble_size > 1:
            print(f"  Ensemble size: {ensemble_size}")
            print(f"  Joint tolerance: {joint_tolerance} rad")
            print(f"  Arrival timeout: {arrival_timeout}s")
        print(f"{'='*60}\n")

        # IK 失败时的回退值
        last_good = {
            'left_joints': None, 'left_gripper': None,
            'right_joints': None, 'right_gripper': None,
        }

        # action_as_obs 模式: 记录上次执行的最后一个 action 作为下次的 state
        last_executed_action = None  # 14D eepose: [left(7), right(7)]

        def signal_handler(signum, frame):
            print("\n\nCtrl+C detected!")
            raise KeyboardInterrupt
        signal.signal(signal.SIGINT, signal_handler)

        try:
            for step in range(n_iterations):
                inference_num = step + 1
                print(f"\n{'='*60}")
                print(f"INFERENCE #{inference_num}/{n_iterations}")
                print(f"{'='*60}")

                # 1. 从 ZMQ 获取传感器数据
                result = self.zmq_sub.receive()
                if result is None:
                    print("[Loop] No data, retrying...")
                    time.sleep(0.1)
                    continue

                staleness = time.time() - self.zmq_sub.last_receive_time
                if staleness > 1.0:
                    print(f"[Loop] Warning: stale data ({staleness:.2f}s old)")

                head_rgb, left_rgb, right_rgb, arm_left, arm_right, gripper_left, gripper_right = result

                if head_rgb is None or left_rgb is None or right_rgb is None:
                    print("[Loop] Missing image data, skipping...")
                    continue

                # 收集视频帧
                if self.save_video:
                    self.video_frames.append(head_rgb.copy())

                # 2. 计算状态向量
                if action_as_obs and last_executed_action is not None:
                    # action_as_obs 模式: 用上次执行的最后一个 action 作为 state
                    state = last_executed_action.astype(np.float32)
                    print(f"[State] (action_as_obs) Using last executed action as state")
                else:
                    # 正常模式: 通过 FK 从关节角度计算 eepose
                    state = self.compute_state_vector(arm_left, arm_right, gripper_left, gripper_right)
                print(f"[State] Left:  pos={state[:3]}, euler={state[3:6]}, gripper={state[6]:.4f}")
                print(f"[State] Right: pos={state[7:10]}, euler={state[10:13]}, gripper={state[13]:.4f}")

                # 3. 组装推理数据并发送到 serve_policy.py
                element = self.prepare_inference_data(head_rgb, left_rgb, right_rgb, state, task_prompt)

                inf_start = time.time()
                result = self.client.infer(element)
                inf_time = time.time() - inf_start
                print(f"[Inference] Time: {inf_time:.3f}s")

                # 4. 处理返回的 actions
                all_actions = np.asarray(result["actions"])
                print(f"[Actions] Received {len(all_actions)} actions from server")

                # 打印前几个 action
                for i, act in enumerate(all_actions[:min(3, len(all_actions))]):
                    left_a = act[:7]
                    right_a = act[7:14]
                    print(f"  Action {i}: L=[{left_a[0]:.4f},{left_a[1]:.4f},{left_a[2]:.4f}] "
                          f"gr={left_a[6]:.3f} | R=[{right_a[0]:.4f},{right_a[1]:.4f},{right_a[2]:.4f}] "
                          f"gr={right_a[6]:.3f}")

                # 5. 计算执行范围
                start_idx = action_index
                end_idx = min(start_idx + execute_steps, len(all_actions))

                if end_idx <= start_idx:
                    print(f"[Loop] No actions to execute")
                    continue

                executed = 0
                ik_fails = 0

                # 6. 流式执行每个动作
                for i in range(start_idx, end_idx):
                    action = all_actions[i]
                    left_eepose = action[:7]
                    right_eepose = action[7:14]

                    print(f"\n  Step {i - start_idx + 1}/{end_idx - start_idx} (Action {i}):")
                    print(f"    L: pos=[{left_eepose[0]:.4f},{left_eepose[1]:.4f},{left_eepose[2]:.4f}] "
                          f"euler=[{left_eepose[3]:.4f},{left_eepose[4]:.4f},{left_eepose[5]:.4f}] "
                          f"gripper={left_eepose[6]:.4f}")
                    print(f"    R: pos=[{right_eepose[0]:.4f},{right_eepose[1]:.4f},{right_eepose[2]:.4f}] "
                          f"euler=[{right_eepose[3]:.4f},{right_eepose[4]:.4f},{right_eepose[5]:.4f}] "
                          f"gripper={right_eepose[6]:.4f}")

                    # IK 求解
                    left_joints, left_gr_raw, right_joints, right_gr_raw, (l_ok, r_ok) = \
                        self.action_to_robot_command(action)

                    # IK 失败回退
                    if not l_ok:
                        ik_fails += 1
                        if last_good['left_joints'] is not None:
                            print(f"    [IK] Left failed, using fallback")
                            left_joints = last_good['left_joints']
                            left_gr_raw = last_good['left_gripper']
                        else:
                            print(f"    [IK] Left failed, no fallback!")

                    if not r_ok:
                        ik_fails += 1
                        if last_good['right_joints'] is not None:
                            print(f"    [IK] Right failed, using fallback")
                            right_joints = last_good['right_joints']
                            right_gr_raw = last_good['right_gripper']
                        else:
                            print(f"    [IK] Right failed, no fallback!")

                    if not l_ok and last_good['left_joints'] is None \
                       and not r_ok and last_good['right_joints'] is None:
                        print(f"    [IK] Both failed with no fallback, skipping!")
                        continue

                    # Ensemble 模式: 跳过中间步, 只在 ensemble 边界发送最终 action
                    if action_as_obs and ensemble_size > 1:
                        step_in_chunk = i - start_idx
                        is_ensemble_end = (step_in_chunk + 1) % ensemble_size == 0 or i == end_idx - 1
                        if not is_ensemble_end:
                            print(f"    [Ensemble] Accumulated ({step_in_chunk % ensemble_size + 1}/{ensemble_size}), skip send")
                            if l_ok:
                                last_good['left_joints'] = left_joints
                                last_good['left_gripper'] = left_gr_raw
                            if r_ok:
                                last_good['right_joints'] = right_joints
                                last_good['right_gripper'] = right_gr_raw
                            last_executed_action = action[:14].copy()
                            continue

                    # 确认模式
                    if confirm_each:
                        inp = input("    Execute? (ENTER=yes, 'skip'=no): ").strip().lower()
                        if inp == 'skip':
                            continue

                    # 发送命令 (重复 repeat_actions 次)
                    for rep in range(repeat_actions):
                        self.cmd_pub.send_command(left_joints, left_gr_raw, right_joints, right_gr_raw)
                        if repeat_actions > 1 and rep < repeat_actions - 1:
                            time.sleep(0.05)
                    executed += 1
                    if repeat_actions > 1:
                        print(f"    SENT x{repeat_actions}!")
                    else:
                        print(f"    SENT!")

                    # 保存成功值
                    if l_ok:
                        last_good['left_joints'] = left_joints
                        last_good['left_gripper'] = left_gr_raw
                    if r_ok:
                        last_good['right_joints'] = right_joints
                        last_good['right_gripper'] = right_gr_raw

                    # 记录最后执行的 action (用于 action_as_obs 模式)
                    last_executed_action = action[:14].copy()

                    # Ensemble 模式: 发送后轮询等待关节到位
                    ensemble_waited = False
                    if action_as_obs and ensemble_size > 1:
                        step_in_chunk = i - start_idx
                        grp_start = (step_in_chunk // ensemble_size) * ensemble_size
                        grp_size = step_in_chunk - grp_start + 1
                        wait_start = time.time()
                        poll_interval = 0.03
                        arrived = False
                        l_err = r_err = float('inf')
                        l_errs = r_errs = None
                        poll_count = 0
                        while True:
                            time.sleep(poll_interval)
                            poll_count += 1
                            arrived, l_err, r_err, l_errs, r_errs = self.check_joint_arrival(
                                left_joints, right_joints, tolerance=joint_tolerance)
                            elapsed = time.time() - wait_start
                            if arrived:
                                print(f"    [Ensemble] Joints ARRIVED ({grp_size} steps merged) "
                                      f"L_max_err={l_err:.6f} R_max_err={r_err:.6f} "
                                      f"waited={elapsed:.2f}s polls={poll_count}")
                                ensemble_waited = True
                                break
                            if elapsed >= arrival_timeout:
                                print(f"    [Ensemble] TIMEOUT ({grp_size} steps merged, {elapsed:.2f}s) "
                                      f"L_max_err={l_err:.6f} R_max_err={r_err:.6f} tol={joint_tolerance}")
                                if l_errs is not None:
                                    print(f"      L joint errors: {l_errs}")
                                if r_errs is not None:
                                    print(f"      R joint errors: {r_errs}")
                                ensemble_waited = True
                                break
                            # 逐渐增大轮询间隔, 最大 0.1s
                            poll_interval = min(poll_interval * 1.3, 0.1)

                    # 动作间短暂延迟 (ensemble 已等过则跳过)
                    if i < end_idx - 1 and not ensemble_waited:
                        time.sleep(0.05)

                print(f"\n  Inference #{inference_num}: Executed {executed}/{end_idx - start_idx} steps"
                      + (f", IK fails: {ik_fails}" if ik_fails > 0 else ""))

                # 等待机器人执行 (ensemble 模式已在每组等过到位, 跳过额外等待)
                if executed > 0 and execution_delay > 0:
                    if action_as_obs and ensemble_size > 1:
                        pass  # ensemble 已轮询等待到位, 无需额外 delay
                    else:
                        print(f"  Waiting {execution_delay}s for robot execution...")
                        time.sleep(execution_delay)

        except KeyboardInterrupt:
            print("\nControl loop interrupted")
        except Exception as e:
            import traceback
            print(f"\nError: {e}")
            traceback.print_exc()
        finally:
            print("\nControl loop ended")
            self.save_video_file(task_prompt.replace(' ', '_'))
            self.zmq_sub.close()
            self.cmd_pub.close()


def main():
    parser = argparse.ArgumentParser(description="PI0 R1 Dual-Arm Controller (Server-Client Mode)")
    parser.add_argument("--init_type", type=int, default=1,
                            help="Init position type: 1=胸前pour(默认), 2=竖直向下pnp")
    # 连接参数
    parser.add_argument("--websocket_host", type=str, default="localhost",
                        help="serve_policy.py host")
    parser.add_argument("--websocket_port", type=int, default=8000,
                        help="serve_policy.py port")
    parser.add_argument("--zmq_host", type=str, default="localhost",
                        help="ros_bridge.py host")
    parser.add_argument("--zmq_data_port", type=int, default=5555,
                        help="ZMQ data port (sensor data)")
    parser.add_argument("--zmq_cmd_port", type=int, default=5556,
                        help="ZMQ command port (robot control)")

    # 任务参数
    parser.add_argument("--task_prompt", type=str, default="pour",
                        help="Task prompt (should match --default_prompt of serve_policy.py)")
    parser.add_argument("--n_iterations", type=int, default=20,
                        help="Number of inference iterations")

    # 执行参数
    parser.add_argument("--chunk_size", type=int, default=10,
                        help="Max actions per inference")
    parser.add_argument("--action_index", type=int, default=0,
                        help="Start executing from this action index")
    parser.add_argument("--execute_steps", type=int, default=10,
                        help="Number of actions to execute per inference")
    parser.add_argument("--execution_delay", type=float, default=0.5,
                        help="Delay after action execution (seconds)")

    # 控制选项
    parser.add_argument("--no_init", action="store_true",
                        help="Skip robot initialization")
    parser.add_argument("--lock_euler", action="store_true",
                        help="Lock end-effector orientation to fixed euler angles")
    parser.add_argument("--confirm_each", action="store_true",
                        help="Confirm before each action")
    parser.add_argument("--action_as_obs", action="store_true",
                        help="Use last executed action as state obs (images still from ZMQ)")
    parser.add_argument("--repeat_actions", type=int, default=1,
                        help="Repeat each action command N times for better execution accuracy")
    parser.add_argument("--ensemble_size", type=int, default=1,
                        help="Action ensemble: merge N steps into 1 send, check joint arrival (with --action_as_obs)")
    parser.add_argument("--joint_tolerance", type=float, default=0.01,
                        help="Joint arrival tolerance in rad, 0.01≈0.57° (used with --ensemble_size)")
    parser.add_argument("--arrival_timeout", type=float, default=1.0,
                        help="Timeout in seconds waiting for joints to arrive (used with --ensemble_size)")

    # 视频
    parser.add_argument("--no_video", action="store_true", help="Disable video recording")
    parser.add_argument("--video_dir", type=str, default="./videos",
                        help="Video save directory")

    args = parser.parse_args()

    controller = PI0RobotController(
        websocket_host=args.websocket_host,
        websocket_port=args.websocket_port,
        zmq_host=args.zmq_host,
        zmq_data_port=args.zmq_data_port,
        zmq_cmd_port=args.zmq_cmd_port,
        lock_euler=args.lock_euler,
        save_video=not args.no_video,
        video_dir=args.video_dir,
    )

    # 传递init_type参数给初始化流程
    if not args.no_init:
        controller.initialize_robot(init_type=args.init_type, wait_for_confirm=True)

    controller.run_control_loop(
        task_prompt=args.task_prompt,
        n_iterations=args.n_iterations,
        chunk_size=args.chunk_size,
        action_index=args.action_index,
        execute_steps=args.execute_steps,
        execution_delay=args.execution_delay,
        init_robot=False,  # 已在上面初始化
        confirm_each=args.confirm_each,
        action_as_obs=args.action_as_obs,
        repeat_actions=args.repeat_actions,
        ensemble_size=args.ensemble_size,
        joint_tolerance=args.joint_tolerance,
        arrival_timeout=args.arrival_timeout,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ROS 数据桥接节点 - 使用系统 Python 3.8 运行
双向桥接：
  1. 将 ROS topics 数据通过 ZMQ 发送给 PI0 模型（Python 3.11）
  2. 接收 PI0 命令并发布到 ROS topics 控制机器人

使用方法:
    # 在终端1运行此脚本（使用系统 Python）
    /usr/bin/python3 ros_bridge.py

    # 在终端2运行 PI0 测试（使用 openpi 虚拟环境）
    ./run_test_pi0.sh zmq
    
    # 如果需要发布控制命令:
    ./run_test_pi0.sh zmq_control
"""

import sys
import os
import time
import json
import numpy as np
import threading
import pickle

# ROS 设置
sys.path.insert(0, '/opt/ros/noetic/lib/python3/dist-packages')
os.environ.setdefault('ROS_MASTER_URI', 'http://192.168.123.15:11311')
os.environ.setdefault('ROS_IP', '192.168.123.15')

import rospy
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float32
import cv2

# ZMQ 设置
try:
    import zmq
except ImportError:
    print("请安装 zmq: pip install pyzmq")
    sys.exit(1)


class ROSBridge:
    """ROS 数据桥接器，双向通信"""
    
    # 优先读取原始高频topic，如果不存在则fallback到降频topic
    READ_TOPICS_PRIMARY = {
        'head_rgb': '/hdas/camera_head/rgb/image_rect_color/compressed',
        'left_rgb': '/left/camera/color/image_raw/compressed',
        'right_rgb': '/right/camera/color/image_raw/compressed',
        'arm_left': '/hdas/feedback_arm_left',
        'arm_right': '/hdas/feedback_arm_right',
        'gripper_left': '/hdas/feedback_gripper_left',
        'gripper_right': '/hdas/feedback_gripper_right',
    }
    
    # 降频topic作为fallback（当录bag时使用throttle节点时）
    READ_TOPICS_FALLBACK = {
        'head_rgb': '/hdas/camera_head/rgb/image_rect_color/compressed',
        'left_rgb': '/left/camera/color/image_raw/compressed',
        'right_rgb': '/right/camera/color/image_raw/compressed',
        'arm_left': '/hdas/feedback_arm_left_low',
        'arm_right': '/hdas/feedback_arm_right_low',
        'gripper_left': '/hdas/feedback_gripper_left_low',
        'gripper_right': '/hdas/feedback_gripper_right_low',
    }
    
    # 发送控制命令的 topics (不带 _low，这些是真正控制机器人的)
    CONTROL_TOPICS = {
        'arm_left': '/motion_target/target_joint_state_arm_left',
        'arm_right': '/motion_target/target_joint_state_arm_right',
        'gripper_left': '/motion_control/position_control_gripper_left',
        'gripper_right': '/motion_control/position_control_gripper_right',
    }
    
    def __init__(self, data_port=5555, cmd_port=5556, use_fallback=False):
        # ZMQ 设置 - 数据发布
        self.context = zmq.Context()
        self.data_socket = self.context.socket(zmq.PUB)
        self.data_socket.bind(f"tcp://*:{data_port}")
        print(f"[ROSBridge] ZMQ data publisher started on port {data_port}")
        
        # ZMQ 设置 - 命令接收
        self.cmd_socket = self.context.socket(zmq.SUB)
        self.cmd_socket.bind(f"tcp://*:{cmd_port}")
        self.cmd_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.cmd_socket.setsockopt(zmq.RCVTIMEO, 10)  # 10ms 超时
        print(f"[ROSBridge] ZMQ command receiver started on port {cmd_port}")
        
        # 初始化 ROS
        print("[ROSBridge] Initializing ROS node...")
        rospy.init_node('ros_bridge_node', anonymous=True)
        
        # 自动检测并选择topic
        print("[ROSBridge] Auto-detecting available topics...")
        self.READ_TOPICS = self._select_topics(force_fallback=use_fallback)
        
        # 数据存储
        self.data = {key: None for key in self.READ_TOPICS.keys()}
        self.last_update = {key: 0 for key in self.READ_TOPICS.keys()}
        
        # 命令计数
        self.cmd_count = 0
        
        # 订阅读取话题
        print(f"[ROSBridge] Subscribing to selected topics...")
        rospy.Subscriber(self.READ_TOPICS['head_rgb'], CompressedImage, 
                        lambda msg: self._image_callback(msg, 'head_rgb'))
        rospy.Subscriber(self.READ_TOPICS['left_rgb'], CompressedImage,
                        lambda msg: self._image_callback(msg, 'left_rgb'))
        rospy.Subscriber(self.READ_TOPICS['right_rgb'], CompressedImage,
                        lambda msg: self._image_callback(msg, 'right_rgb'))
        rospy.Subscriber(self.READ_TOPICS['arm_left'], JointState,
                        lambda msg: self._joint_callback(msg, 'arm_left'))
        rospy.Subscriber(self.READ_TOPICS['arm_right'], JointState,
                        lambda msg: self._joint_callback(msg, 'arm_right'))
        rospy.Subscriber(self.READ_TOPICS['gripper_left'], JointState,
                        lambda msg: self._joint_callback(msg, 'gripper_left'))
        rospy.Subscriber(self.READ_TOPICS['gripper_right'], JointState,
                        lambda msg: self._joint_callback(msg, 'gripper_right'))
        
        # 创建控制发布器
        print("[ROSBridge] Creating control publishers...")
        self.arm_left_pub = rospy.Publisher(
            self.CONTROL_TOPICS['arm_left'], JointState, queue_size=1)
        self.arm_right_pub = rospy.Publisher(
            self.CONTROL_TOPICS['arm_right'], JointState, queue_size=1)
        # self.gripper_left_pub = rospy.Publisher(
        #     self.CONTROL_TOPICS['gripper_left'], JointState, queue_size=1)
        # self.gripper_right_pub = rospy.Publisher(
        #     self.CONTROL_TOPICS['gripper_right'], JointState, queue_size=1)
        self.gripper_left_pub = rospy.Publisher(
            self.CONTROL_TOPICS['gripper_left'], Float32, queue_size=1)
        self.gripper_right_pub = rospy.Publisher(
            self.CONTROL_TOPICS['gripper_right'], Float32, queue_size=1)
        
        print("[ROSBridge] Ready! Waiting for data...")

    
    def _select_topics(self, force_fallback=False):
        """自动检测并选择可用的topics
        
        Args:
            force_fallback: 如果为True，强制使用fallback topics
            
        Returns:
            dict: 选中的topic配置
        """
        import time
        
        # 等待ROS master启动
        timeout = 5.0
        start_time = time.time()
        while not rospy.is_shutdown() and time.time() - start_time < timeout:
            try:
                rospy.get_published_topics()
                break
            except:
                time.sleep(0.1)
        
        # 获取当前所有可用的topics
        try:
            available_topics = [topic for topic, _ in rospy.get_published_topics()]
        except Exception as e:
            print(f"[ROSBridge] Warning: Could not get published topics: {e}")
            print(f"[ROSBridge] Falling back to PRIMARY topics")
            return self.READ_TOPICS_PRIMARY.copy()
        
        print(f"[ROSBridge] Found {len(available_topics)} published topics")
        
        if force_fallback:
            print(f"[ROSBridge] Force using FALLBACK (low freq) topics")
            return self.READ_TOPICS_FALLBACK.copy()
        
        # 检测每个数据类型的topic可用性
        selected_topics = {}
        warnings = []
        
        for key in self.READ_TOPICS_PRIMARY.keys():
            primary_topic = self.READ_TOPICS_PRIMARY[key]
            fallback_topic = self.READ_TOPICS_FALLBACK[key]
            
            # 优先使用primary（高频）
            if primary_topic in available_topics:
                selected_topics[key] = primary_topic
                print(f"[ROSBridge] ✓ {key:15s} -> {primary_topic} (HIGH FREQ)")
            # 如果primary不可用，使用fallback（低频）
            elif fallback_topic in available_topics and fallback_topic != primary_topic:
                selected_topics[key] = fallback_topic
                warnings.append(key)
                print(f"[ROSBridge] ⚠ {key:15s} -> {fallback_topic} (LOW FREQ - FALLBACK)")
            # 两个都不可用，使用primary并警告
            else:
                selected_topics[key] = primary_topic
                print(f"[ROSBridge] ✗ {key:15s} -> {primary_topic} (NOT FOUND - WILL WAIT)")
        
        # 如果有使用fallback的topic，发出警告
        if warnings:
            print(f"\n{'='*70}")
            print(f"⚠️  WARNING: Using LOW FREQUENCY topics for: {', '.join(warnings)}")
            print(f"⚠️  This may cause:")
            print(f"    - Low state update rate")
            print(f"    - Robot movement lag/oscillation")
            print(f"    - Poor control performance")
            print(f"")
            print(f"💡 Solution:")
            print(f"    For REAL-TIME INFERENCE: Don't run record_15hz.launch")
            print(f"    For RECORDING DATA: Run record_15hz.launch (creates _low topics)")
            print(f"{'='*70}\n")
        else:
            print(f"\n{'='*70}")
            print(f"✅ Using HIGH FREQUENCY topics (optimal for real-time control)")
            print(f"{'='*70}\n")
        
        return selected_topics

    
    def _image_callback(self, msg, key):
        """处理图像消息"""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None:
                # 调整尺寸
                if img.shape[1] != 640 or img.shape[0] != 480:
                    img = cv2.resize(img, (640, 480))
                # BGR -> RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.data[key] = img
                self.last_update[key] = time.time()
        except Exception as e:
            print(f"[ROSBridge] Error decoding {key}: {e}")
    
    def _joint_callback(self, msg, key):
        """处理关节消息"""
        try:
            self.data[key] = np.array(msg.position)
            self.last_update[key] = time.time()
        except Exception as e:
            print(f"[ROSBridge] Error processing {key}: {e}")
    
    def _check_data_ready(self):
        """检查所有数据是否准备好"""
        now = time.time()
        timeout = 1.0  # 1秒超时
        for key, t in self.last_update.items():
            if now - t > timeout:
                return False
        return True
    
    def _pack_data(self):
        """打包数据用于发送"""
        # 将图像编码为 JPEG
        def encode_img(img):
            if img is None:
                return None
            _, encoded = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 
                                      [cv2.IMWRITE_JPEG_QUALITY, 90])
            return encoded.tobytes()
        
        packed = {
            'timestamp': time.time(),
            'head_rgb': encode_img(self.data['head_rgb']),
            'left_rgb': encode_img(self.data['left_rgb']),
            'right_rgb': encode_img(self.data['right_rgb']),
            'arm_left': self.data['arm_left'].tolist() if self.data['arm_left'] is not None else None,
            'arm_right': self.data['arm_right'].tolist() if self.data['arm_right'] is not None else None,
            'gripper_left': self.data['gripper_left'].tolist() if self.data['gripper_left'] is not None else None,
            'gripper_right': self.data['gripper_right'].tolist() if self.data['gripper_right'] is not None else None,
        }
        return packed
    
    def _publish_control_command(self, cmd):
        """发布控制命令到 ROS topics
        
        命令格式:
        {
            'arm_left': [j1, j2, j3, j4, j5, j6, gripper_invalid],  # 7 维: 6关节 + 无效夹爪占位(-2.7)
            'arm_right': [j1, j2, j3, j4, j5, j6, gripper_invalid], # 7 维: 6关节 + 无效夹爪占位(-2.7)
            'gripper_left': float,   # 0-100，真正的夹爪控制
            'gripper_right': float,  # 0-100，真正的夹爪控制
        }
        """
        try:
            # 发布左臂关节角度
            if cmd.get('arm_left') is not None:
                msg = JointState()
                msg.header.stamp = rospy.Time.now()
                msg.velocity = [0.5] * len(cmd['arm_left'])  # 设置速度
                msg.position = cmd['arm_left']
                self.arm_left_pub.publish(msg)
            
            # 发布右臂关节角度
            if cmd.get('arm_right') is not None:
                msg = JointState()
                msg.header.stamp = rospy.Time.now()
                msg.velocity = [0.5] * len(cmd['arm_right'])  # 设置速度    
                msg.position = cmd['arm_right']
                self.arm_right_pub.publish(msg)
            
            # 发布左夹爪
            if cmd.get('gripper_left') is not None:
                # msg = JointState()
                msg = Float32()
                # msg.header.stamp = rospy.Time.now()
                msg.data = cmd['gripper_left']  # 0-100
                print(f"[ROSBridge] Left gripper command: {msg.data}")
                self.gripper_left_pub.publish(msg)
            
            # 发布右夹爪
            if cmd.get('gripper_right') is not None:
                msg = Float32()
                # msg.header.stamp = rospy.Time.now()
                msg.data = cmd['gripper_right']  # 0-100
                print(f"[ROSBridge] Right gripper command: {msg.data}")
                self.gripper_right_pub.publish(msg)
            
            self.cmd_count += 1
            return True
        except Exception as e:
            print(f"[ROSBridge] Error publishing command: {e}")
            return False
    
    def _receive_commands(self):
        """接收并执行控制命令（非阻塞）"""
        try:
            while True:
                try:
                    cmd_bytes = self.cmd_socket.recv(zmq.NOBLOCK)
                    cmd = pickle.loads(cmd_bytes)
                    
                    if self._publish_control_command(cmd):
                        if self.cmd_count % 10 == 1:
                            print(f"[ROSBridge] Published command #{self.cmd_count}")
                except zmq.Again:
                    # 没有更多命令
                    break
        except Exception as e:
            print(f"[ROSBridge] Error receiving command: {e}")
    
    def run(self, rate_hz=15):
        """主循环"""
        rate = rospy.Rate(rate_hz)
        frame_count = 0
        
        print(f"[ROSBridge] Publishing at {rate_hz} Hz")
        print(f"[ROSBridge] Control topics:")
        print(f"    Arm Left:     {self.CONTROL_TOPICS['arm_left']}")
        print(f"    Arm Right:    {self.CONTROL_TOPICS['arm_right']}")
        print(f"    Gripper Left: {self.CONTROL_TOPICS['gripper_left']}")
        print(f"    Gripper Right:{self.CONTROL_TOPICS['gripper_right']}")
        
        while not rospy.is_shutdown():
            # 1. 接收并执行控制命令
            self._receive_commands()
            
            # 2. 发布传感器数据
            if self._check_data_ready():
                packed = self._pack_data()
                
                # 使用 pickle 序列化
                data_bytes = pickle.dumps(packed)
                
                self.data_socket.send(data_bytes)
                frame_count += 1
                
                if frame_count % 30 == 0:
                    print(f"[ROSBridge] Published {frame_count} frames, {self.cmd_count} commands")
            else:
                if frame_count == 0:
                    missing = [k for k, t in self.last_update.items() if time.time() - t > 1.0]
                    if missing:
                        print(f"[ROSBridge] Waiting for: {missing}")
            
            rate.sleep()
        
        self.data_socket.close()
        self.cmd_socket.close()
        self.context.term()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_port", type=int, default=5555, help="ZMQ data port")
    parser.add_argument("--cmd_port", type=int, default=5556, help="ZMQ command port")
    parser.add_argument("--rate", type=int, default=15, help="Publishing rate (Hz)")
    parser.add_argument("--use_fallback", action="store_true", 
                       help="Use fallback (low freq) topics instead of primary")
    args = parser.parse_args()
    
    try:
        bridge = ROSBridge(data_port=args.data_port, cmd_port=args.cmd_port, 
                          use_fallback=args.use_fallback)
        bridge.run(rate_hz=args.rate)
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print("\n[ROSBridge] Shutting down...")


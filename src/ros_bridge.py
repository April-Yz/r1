#!/usr/bin/env python3
"""
ROS 数据桥接节点 - 使用系统 Python 3.8 运行
将 ROS topics 数据通过 ZMQ 发送给 PI0 模型（Python 3.11）

使用方法:
    # 在终端1运行此脚本（使用系统 Python）
    /usr/bin/python3 ros_bridge.py

    # 在终端2运行 PI0 测试（使用 openpi 虚拟环境）
    ./run_test_pi0.sh zmq
"""

import sys
import os
import time
import json
import numpy as np

# ROS 设置
sys.path.insert(0, '/opt/ros/noetic/lib/python3/dist-packages')
os.environ.setdefault('ROS_MASTER_URI', 'http://192.168.123.15:11311')
os.environ.setdefault('ROS_IP', '192.168.123.15')

import rospy
from sensor_msgs.msg import CompressedImage, JointState
import cv2

# ZMQ 设置
try:
    import zmq
except ImportError:
    print("请安装 zmq: pip install pyzmq")
    sys.exit(1)


class ROSBridge:
    """ROS 数据桥接器，通过 ZMQ 发送数据"""
    
    TOPICS = {
        'head_rgb': '/hdas/camera_head/rgb/image_rect_color/compressed',
        'left_rgb': '/left/camera/color/image_raw/compressed',
        'right_rgb': '/right/camera/color/image_raw/compressed',
        'arm_left': '/hdas/feedback_arm_left_low',
        'arm_right': '/hdas/feedback_arm_right_low',
        'gripper_left': '/hdas/feedback_gripper_left_low',
        'gripper_right': '/hdas/feedback_gripper_right_low',
    }
    
    def __init__(self, port=5555):
        # ZMQ 设置
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        print(f"[ROSBridge] ZMQ publisher started on port {port}")
        
        # 数据存储
        self.data = {key: None for key in self.TOPICS.keys()}
        self.last_update = {key: 0 for key in self.TOPICS.keys()}
        
        # 初始化 ROS
        print("[ROSBridge] Initializing ROS node...")
        rospy.init_node('ros_bridge_node', anonymous=True)
        
        # 订阅话题
        print("[ROSBridge] Subscribing to topics...")
        rospy.Subscriber(self.TOPICS['head_rgb'], CompressedImage, 
                        lambda msg: self._image_callback(msg, 'head_rgb'))
        rospy.Subscriber(self.TOPICS['left_rgb'], CompressedImage,
                        lambda msg: self._image_callback(msg, 'left_rgb'))
        rospy.Subscriber(self.TOPICS['right_rgb'], CompressedImage,
                        lambda msg: self._image_callback(msg, 'right_rgb'))
        rospy.Subscriber(self.TOPICS['arm_left'], JointState,
                        lambda msg: self._joint_callback(msg, 'arm_left'))
        rospy.Subscriber(self.TOPICS['arm_right'], JointState,
                        lambda msg: self._joint_callback(msg, 'arm_right'))
        rospy.Subscriber(self.TOPICS['gripper_left'], JointState,
                        lambda msg: self._joint_callback(msg, 'gripper_left'))
        rospy.Subscriber(self.TOPICS['gripper_right'], JointState,
                        lambda msg: self._joint_callback(msg, 'gripper_right'))
        
        print("[ROSBridge] Ready! Waiting for data...")
    
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
    
    def run(self, rate_hz=15):
        """主循环"""
        rate = rospy.Rate(rate_hz)
        frame_count = 0
        
        print(f"[ROSBridge] Publishing at {rate_hz} Hz")
        
        while not rospy.is_shutdown():
            if self._check_data_ready():
                packed = self._pack_data()
                
                # 使用 msgpack 或 pickle 序列化
                import pickle
                data_bytes = pickle.dumps(packed)
                
                self.socket.send(data_bytes)
                frame_count += 1
                
                if frame_count % 30 == 0:
                    print(f"[ROSBridge] Published {frame_count} frames")
            else:
                if frame_count == 0:
                    missing = [k for k, t in self.last_update.items() if time.time() - t > 1.0]
                    if missing:
                        print(f"[ROSBridge] Waiting for: {missing}")
            
            rate.sleep()
        
        self.socket.close()
        self.context.term()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555, help="ZMQ port")
    parser.add_argument("--rate", type=int, default=15, help="Publishing rate (Hz)")
    args = parser.parse_args()
    
    try:
        bridge = ROSBridge(port=args.port)
        bridge.run(rate_hz=args.rate)
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print("\n[ROSBridge] Shutting down...")

# python rosbag2pi.py --bag_dir /home/pine/yzj/pour --output_dir ./training_data --task_name pour
import os
import glob
import argparse
import json
import numpy as np
import cv2
import h5py
import rosbag
from scipy.spatial.transform import Rotation
from cv_bridge import CvBridge

# ================= 配置部分 =================
# 定义 Topic 映射
TOPICS = {
    # 图像 Topics
    'cam_high': '/hdas/camera_head/rgb/image_rect_color/compressed',
    'cam_left_wrist': '/left/camera/color/image_raw/compressed',
    'cam_right_wrist': '/right/camera/color/image_raw/compressed',
    
    # 状态 Topics (假设这些 Topic 包含 Pose 信息)
    'left_arm_state': '/hdas/feedback_arm_left',
    'right_arm_state': '/hdas/feedback_arm_right',
    'left_gripper_state': '/hdas/feedback_gripper_left',
    'right_gripper_state': '/hdas/feedback_gripper_right'
}

RESIZE_DIMS = (640, 480) # 或者是 (320, 240)，根据你的需求调整

# ================= 辅助函数 =================

def quaternion_to_euler(quat, order='xyz'):
    """
    将四元数 [x, y, z, w] 或 [w, x, y, z] 转换为欧拉角
    注意：ROS 标准是 [x, y, z, w], scipy 也是 [x, y, z, w]
    但是你的参考代码里有一个转换逻辑，这里我们假设输入是 ROS 消息的标准 [x, y, z, w]
    """
    # 确保归一化
    quat = quat / np.linalg.norm(quat)
    r = Rotation.from_quat(quat)
    return r.as_euler(order, degrees=False)

def images_encoding(imgs):
    """ 将图像列表压缩为 JPEG 字节流并 Padding """
    encode_data = []
    max_len = 0
    
    for img in imgs:
        # cv2.imencode 默认输入 BGR。如果你读取的是 RGB，需注意。
        # 此处假设 imgs 已经是 BGR (OpenCV 默认)
        success, encoded_image = cv2.imencode(".jpg", img)
        if success:
            jpeg_data = encoded_image.tobytes()
            encode_data.append(jpeg_data)
            max_len = max(max_len, len(jpeg_data))
    
    padded_data = []
    for data in encode_data:
        padded_data.append(data.ljust(max_len, b"\0"))
        
    return padded_data, max_len

def get_closest_msg(target_time, topic_data_list):
    """
    在 topic_data_list [(time, msg), ...] 中寻找离 target_time 最近的消息
    """
    if not topic_data_list:
        return None
    
    # 简单遍历寻找最近时间戳 (由于是顺序的，也可以用二分查找优化，但数据量不大直接遍历即可)
    # 为了效率，我们假设数据是大致对齐的，直接找绝对差值最小的
    times = [t.to_sec() for t, m in topic_data_list]
    target_sec = target_time.to_sec()
    idx = (np.abs(np.array(times) - target_sec)).argmin()
    
    # 可以设置一个阈值，比如相差超过 0.1秒 认为丢失
    if abs(times[idx] - target_sec) > 0.2:
        # print(f"Warning: Time sync diff large: {abs(times[idx] - target_sec):.4f}s")
        pass
        
    return topic_data_list[idx][1] # 返回 msg

def extract_pose_from_msg(msg):
    """
    从 ROS 消息中提取位置和欧拉角。
    警告：这取决于你的 '/hdas/feedback_arm_...' 消息的具体定义。
    假设它是 geometry_msgs/PoseStamped 或类似结构。
    """
    pos = np.zeros(3)
    quat = np.array([0, 0, 0, 1.0]) # x, y, z, w

    try:
        # 尝试常见的属性路径
        if hasattr(msg, 'pose'):
            p = msg.pose
            if hasattr(p, 'pose'): # Handle PoseStamped
                p = p.pose
            
            pos = np.array([p.position.x, p.position.y, p.position.z])
            quat = np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
        
        elif hasattr(msg, 'transform'): # Handle TransformStamped
            t = msg.transform
            pos = np.array([t.translation.x, t.translation.y, t.translation.z])
            quat = np.array([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w])
        
        # === 如果你的消息是自定义的，请在这里添加逻辑 ===
        # 例如: 
        # pos = np.array([msg.x, msg.y, msg.z])
        # quat = np.array([msg.qx, msg.qy, msg.qz, msg.qw])
            
    except Exception as e:
        print(f"Error extracting pose: {e}")
    
    euler = quaternion_to_euler(quat)
    return np.concatenate([pos, euler]) # [x, y, z, r, p, y]

def extract_gripper_from_msg(msg):
    """
    提取夹爪状态。假设是 float (0-1) 或 bool
    """
    val = 0.0
    try:
        # 假设 msg.data 存储数值
        if hasattr(msg, 'data'):
            val = float(msg.data)
        # 或者是 width
        elif hasattr(msg, 'width'):
            val = float(msg.width)
        elif hasattr(msg, 'position'): # JointState
            val = msg.position[0]
    except:
        pass
    return val

def process_bag(bag_path, output_root, task_name="pour", episode_idx=0):
    print(f"Processing bag: {bag_path}")
    bag = rosbag.Bag(bag_path)
    
    # 1. 读取所有数据到内存 (按 Topic 分类)
    # 这是一个简单策略，如果 Bag 很大 (几GB)，建议改为流式处理
    data_buffer = {k: [] for k in TOPICS.values()}
    
    desired_topics = list(TOPICS.values())
    
    for topic, msg, t in bag.read_messages(topics=desired_topics):
        data_buffer[topic].append((t, msg))
        
    bag.close()
    
    # 检查主视角是否有数据
    main_topic = TOPICS['cam_high']
    if not data_buffer[main_topic]:
        print("Error: No data found for main camera.")
        return False

    # 2. 对齐数据 (以 Head Camera 为基准)
    aligned_data = []
    
    main_msgs = data_buffer[main_topic]
    total_frames = len(main_msgs)
    print(f"Total frames (Head Camera): {total_frames}")

    for t, main_img_msg in main_msgs:
        frame_data = {}
        
        # 获取各相机图像
        frame_data['cam_high'] = main_img_msg
        frame_data['cam_left_wrist'] = get_closest_msg(t, data_buffer[TOPICS['cam_left_wrist']])
        frame_data['cam_right_wrist'] = get_closest_msg(t, data_buffer[TOPICS['cam_right_wrist']])
        
        # 获取状态
        frame_data['left_arm'] = get_closest_msg(t, data_buffer[TOPICS['left_arm_state']])
        frame_data['right_arm'] = get_closest_msg(t, data_buffer[TOPICS['right_arm_state']])
        frame_data['left_gripper'] = get_closest_msg(t, data_buffer[TOPICS['left_gripper_state']])
        frame_data['right_gripper'] = get_closest_msg(t, data_buffer[TOPICS['right_gripper_state']])
        
        # 简单的完整性检查
        if frame_data['left_arm'] is None or frame_data['cam_left_wrist'] is None:
            continue # 跳过这一帧
            
        aligned_data.append(frame_data)

    # 3. 解析并转换数据
    states = []
    imgs_high = []
    imgs_left = []
    imgs_right = []
    
    bridge = CvBridge()
    
    print("Converting messages to numpy arrays...")
    for item in aligned_data:
        # --- 处理图像 ---
        try:
            # CompressedImage -> cv2 image (BGR)
            # ROS compressed is usually bgr8 or rgb8 encoded in jpeg
            i_h = bridge.compressed_imgmsg_to_cv2(item['cam_high'])
            i_l = bridge.compressed_imgmsg_to_cv2(item['cam_left_wrist'])
            i_r = bridge.compressed_imgmsg_to_cv2(item['cam_right_wrist'])
            
            # Resize
            i_h = cv2.resize(i_h, RESIZE_DIMS)
            i_l = cv2.resize(i_l, RESIZE_DIMS)
            i_r = cv2.resize(i_r, RESIZE_DIMS)
            
            # 注意：如果训练代码期望 RGB，这里需要转换
            # i_h = cv2.cvtColor(i_h, cv2.COLOR_BGR2RGB)
            
            imgs_high.append(i_h)
            imgs_left.append(i_l)
            imgs_right.append(i_r)
        except Exception as e:
            print(f"Image decode error: {e}")
            continue

        # --- 处理状态 ---
        # 目标格式: [Left Pose(6), Left Gripper(1), Right Pose(6), Right Gripper(1)]
        l_pose = extract_pose_from_msg(item['left_arm'])    # shape (6,)
        r_pose = extract_pose_from_msg(item['right_arm'])   # shape (6,)
        l_grip = extract_gripper_from_msg(item['left_gripper'])
        r_grip = extract_gripper_from_msg(item['right_gripper'])
        
        # 拼接
        state_vec = np.concatenate([l_pose, [l_grip], r_pose, [r_grip]])
        states.append(state_vec)

    # 转为 Numpy
    states = np.array(states, dtype=np.float32)
    
    # 4. 构建输出 (Action / Observation Split)
    # Observation: 0 ~ T-1
    # Action: 1 ~ T
    if len(states) < 2:
        print("Not enough data to create episode.")
        return False
        
    state_out = states[:-1]
    action_out = states[1:]
    
    # 图像也要切片对应 Observation
    imgs_high = imgs_high[:-1]
    imgs_left = imgs_left[:-1]
    imgs_right = imgs_right[:-1]

    # 5. 保存 HDF5
    episode_dir = os.path.join(output_root, f"episode_{episode_idx}")
    os.makedirs(episode_dir, exist_ok=True)
    
    # 保存 json
    with open(os.path.join(episode_dir, "instructions.json"), "w") as f:
        json.dump({"instructions": [task_name]}, f, indent=2)
        
    hdf5_path = os.path.join(episode_dir, f"episode_{episode_idx}.hdf5")
    
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("action", data=action_out)
        
        obs = f.create_group("observations")
        obs.create_dataset("ee_pose", data=state_out) # 使用 ee_pose 字段
        
        # 辅助维度信息 (可选，参照你给的代码)
        obs.create_dataset("left_arm_dim", data=np.full(len(state_out), 6)) 
        obs.create_dataset("right_arm_dim", data=np.full(len(state_out), 6))
        
        img_grp = obs.create_group("images")
        
        print("Encoding Images...")
        enc_high, len_h = images_encoding(imgs_high)
        img_grp.create_dataset("cam_high", data=enc_high, dtype=f"S{len_h}")
        
        enc_left, len_l = images_encoding(imgs_left)
        img_grp.create_dataset("cam_left_wrist", data=enc_left, dtype=f"S{len_l}")
        
        enc_right, len_r = images_encoding(imgs_right)
        img_grp.create_dataset("cam_right_wrist", data=enc_right, dtype=f"S{len_r}")
        
    print(f"Saved: {hdf5_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag_dir", type=str, required=True, help="Path to folder containing .bag files")
    parser.add_argument("--output_dir", type=str, default="processed_data_rosbag", help="Output root")
    parser.add_argument("--task_name", type=str, default="pour")
    args = parser.parse_args()
    
    bag_files = glob.glob(os.path.join(args.bag_dir, "*.bag"))
    bag_files.sort()
    
    if not bag_files:
        print(f"No bag files found in {args.bag_dir}")
        exit()
        
    count = 0
    for idx, bag_file in enumerate(bag_files):
        success = process_bag(bag_file, args.output_dir, args.task_name, idx)
        if success:
            count += 1
            
    print(f"All done. Processed {count} episodes.")
#!/usr/bin/env python3
"""
High Performance ROS bag to HDF5 Converter (Resized to 640x480)
Optimizations:
1. Resizes all images to 640x480 (Save space).
2. Uses INTER_NEAREST for depth resizing (Preserve accuracy).
3. Parallel Processing & Deferred Decoding.
resize 640*480

python3 bag2h5x_yzj_speed.py /home/pine/yzj/pour/r1_data_20260127_154637.bag
python3 bag2h5x_yzj_speed.py /home/pine/yzj/pour --batch

python3 bag2h5x_yzj_speed.py /home/pine/yzj/pour --batch --output /home/pine/yzj/h5out
python3 bag2h5x_yzj_speed.py /media/pine/新加卷/R1/pour --batch --workers 4 --output /home/pine/yzj/h5out

High Performance ROS bag to HDF5 Converter (Batch Support + Resized 640x480)
Usage:
    Single file: python3 bag2h5x.py input.bag --output output.h5
    Batch dir:   python3 bag2h5x.py /input/dir --batch --workers 8 --output /output/dir
"""

import rosbag
import h5py
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import bisect

# --- 目标分辨率设置 ---
TARGET_WIDTH = 640
TARGET_HEIGHT = 480

# --- 全局解码函数 (必须在主作用域) ---

def decode_depth_chunk(args):
    """解码深度图并 Resize 到 640x480"""
    raw_bytes_list, width, height, encoding = args
    decoded_list = []
    
    # 确定数据类型
    if '16UC1' in encoding or 'mono16' in encoding:
        dtype = np.uint16
        n_channels = 1
    elif '32FC1' in encoding:
        dtype = np.float32
        n_channels = 1
    elif '8UC1' in encoding or 'mono8' in encoding:
        dtype = np.uint8
        n_channels = 1
    else:
        dtype = np.uint16
        n_channels = 1

    for data in raw_bytes_list:
        if data is None:
            decoded_list.append(np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=dtype))
            continue
            
        img = np.frombuffer(data, dtype=dtype)
        try:
            # 1. 还原原始形状
            if n_channels == 1:
                img = img.reshape(height, width)
            else:
                img = img.reshape(height, width, n_channels)
            
            # 2. Resize (关键：深度图必须用最近邻插值)
            if width != TARGET_WIDTH or height != TARGET_HEIGHT:
                img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST)
                
            decoded_list.append(img)
        except ValueError:
            decoded_list.append(np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=dtype))
            
    return decoded_list

def decode_rgb_chunk(bytes_list):
    """解码 RGB 并 Resize 到 640x480"""
    decoded_list = []
    for data in bytes_list:
        if data is None:
            decoded_list.append(np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)) 
            continue
            
        np_arr = np.frombuffer(data, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img_bgr is not None:
            # 1. Resize (RGB图用线性插值)
            if img_bgr.shape[1] != TARGET_WIDTH or img_bgr.shape[0] != TARGET_HEIGHT:
                img_bgr = cv2.resize(img_bgr, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
            
            # 2. 转 RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            decoded_list.append(img_rgb)
        else:
            decoded_list.append(None)
    return decoded_list

# --- 辅助函数 ---

def get_nearest_index(timestamps, target_time):
    idx = bisect.bisect_left(timestamps, target_time)
    if idx == 0: return 0
    if idx == len(timestamps): return len(timestamps) - 1
    before = timestamps[idx - 1]
    after = timestamps[idx]
    if after - target_time < target_time - before: return idx
    else: return idx - 1

# --- 核心转换逻辑 ---

def bag_to_h5(bag_path, output_path, num_workers=4):
    bag_path = Path(bag_path)
    print(f"\n>>> Processing: {bag_path.name}")
    print(f"    Output: {output_path}")
    
    start_time = time.time()

    # 1. READ PHASE
    raw_data = {
        'camera_head_rgb': [], 'camera_head_depth': [],
        'camera_left_rgb': [], 'camera_left_depth': [],
        'camera_right_rgb': [], 'camera_right_depth': [],
    }
    meta = {
        'head_w': 640, 'head_h': 480, 'head_enc': '16UC1',
        'left_w': 640, 'left_h': 480, 'left_enc': '16UC1',
        'right_w': 640, 'right_h': 480, 'right_enc': '16UC1'
    }
    num_data = {
        'arm_left_pos': [], 'arm_right_pos': [],
        'gripper_left_pos': [], 'gripper_right_pos': [],
        'action_arm_left_pos': [], 'action_arm_right_pos': [],
        'action_gripper_left_pos_cmd': [], 'action_gripper_right_pos_cmd': []
    }
    timestamps = {k: [] for k in list(raw_data.keys()) + list(num_data.keys())}
    
    try:
        bag = rosbag.Bag(str(bag_path))
        info = bag.get_type_and_topic_info()[1]
        total_msgs = sum(info[topic].message_count for topic in info.keys() if topic in info)
        
        print("    [1/4] Reading bag structure...")
        with tqdm(total=total_msgs, leave=False) as pbar:
            for topic, msg, t in bag.read_messages():
                t_sec = t.to_sec()
                
                # Images
                if topic == '/hdas/camera_head/rgb/image_rect_color/compressed':
                    raw_data['camera_head_rgb'].append(msg.data)
                    timestamps['camera_head_rgb'].append(t_sec)
                elif topic == '/hdas/camera_head/depth/depth_registered':
                    raw_data['camera_head_depth'].append(msg.data)
                    timestamps['camera_head_depth'].append(t_sec)
                    meta['head_enc'] = msg.encoding; meta['head_h'] = msg.height; meta['head_w'] = msg.width
                elif topic == '/left/camera/color/image_raw/compressed':
                    raw_data['camera_left_rgb'].append(msg.data)
                    timestamps['camera_left_rgb'].append(t_sec)
                elif topic == '/left/camera/depth/image_rect_raw':
                    raw_data['camera_left_depth'].append(msg.data)
                    timestamps['camera_left_depth'].append(t_sec)
                    meta['left_enc'] = msg.encoding; meta['left_h'] = msg.height; meta['left_w'] = msg.width
                elif topic == '/right/camera/color/image_raw/compressed':
                    raw_data['camera_right_rgb'].append(msg.data)
                    timestamps['camera_right_rgb'].append(t_sec)
                elif topic == '/right/camera/depth/image_rect_raw':
                    raw_data['camera_right_depth'].append(msg.data)
                    timestamps['camera_right_depth'].append(t_sec)
                    meta['right_enc'] = msg.encoding; meta['right_h'] = msg.height; meta['right_w'] = msg.width

                # Numerical
                elif topic == '/hdas/feedback_arm_left':
                    num_data['arm_left_pos'].append(msg.position); timestamps['arm_left_pos'].append(t_sec)
                elif topic == '/hdas/feedback_arm_right':
                    num_data['arm_right_pos'].append(msg.position); timestamps['arm_right_pos'].append(t_sec)
                elif topic == '/hdas/feedback_gripper_left':
                    num_data['gripper_left_pos'].append(msg.position); timestamps['gripper_left_pos'].append(t_sec)
                elif topic == '/hdas/feedback_gripper_right':
                    num_data['gripper_right_pos'].append(msg.position); timestamps['gripper_right_pos'].append(t_sec)
                elif topic == '/motion_target/target_joint_state_arm_left':
                    num_data['action_arm_left_pos'].append(msg.position); timestamps['action_arm_left_pos'].append(t_sec)
                elif topic == '/motion_target/target_joint_state_arm_right':
                    num_data['action_arm_right_pos'].append(msg.position); timestamps['action_arm_right_pos'].append(t_sec)
                elif topic == '/motion_control/position_control_gripper_left':
                    num_data['action_gripper_left_pos_cmd'].append(msg.data); timestamps['action_gripper_left_pos_cmd'].append(t_sec)
                elif topic == '/motion_control/position_control_gripper_right':
                    num_data['action_gripper_right_pos_cmd'].append(msg.data); timestamps['action_gripper_right_pos_cmd'].append(t_sec)
                
                pbar.update(1)
        bag.close()
    except Exception as e:
        print(f"Error reading bag: {e}")
        return False

    # 2. ALIGNMENT
    print("    [2/4] Aligning timestamps (15Hz)...")
    all_ts = []
    for k in timestamps:
        if timestamps[k]: all_ts.extend(timestamps[k])
    
    if not all_ts:
        print("    Error: No timestamps found.")
        return False

    TARGET_FPS = 15
    t_start, t_end = min(all_ts), max(all_ts)
    duration = t_end - t_start
    num_samples = int(duration * TARGET_FPS)
    if num_samples < 1: num_samples = 1
    
    uniform_timeline = np.linspace(t_start, t_end, num_samples)
    aligned_data = {'timestamps': uniform_timeline}

    # Interpolate Numerical
    def interp_numerical(key):
        ts = np.array(timestamps[key])
        vals = np.array(num_data[key])
        if len(ts) == 0: return None
        if len(ts) == 1: return np.repeat(vals[0][np.newaxis, :], num_samples, axis=0)
        
        if vals.ndim == 1:
            f = interp1d(ts, vals, kind='nearest', fill_value="extrapolate")
        else:
            f = interp1d(ts, vals, axis=0, kind='linear', bounds_error=False, fill_value=(vals[0], vals[-1]))
        return f(uniform_timeline)

    aligned_data['obs_arm_left_pos'] = interp_numerical('arm_left_pos')
    aligned_data['obs_arm_right_pos'] = interp_numerical('arm_right_pos')
    aligned_data['obs_gripper_left_pos'] = interp_numerical('gripper_left_pos')
    aligned_data['obs_gripper_right_pos'] = interp_numerical('gripper_right_pos')
    aligned_data['action_arm_left_pos'] = interp_numerical('action_arm_left_pos')
    aligned_data['action_arm_right_pos'] = interp_numerical('action_arm_right_pos')

    if len(num_data['action_gripper_left_pos_cmd']) > 0:
        aligned_data['action_gripper_left_cmd'] = np.array(num_data['action_gripper_left_pos_cmd'])
        aligned_data['action_gripper_left_cmd_timestamps'] = np.array(timestamps['action_gripper_left_pos_cmd'])
    if len(num_data['action_gripper_right_pos_cmd']) > 0:
        aligned_data['action_gripper_right_cmd'] = np.array(num_data['action_gripper_right_pos_cmd'])
        aligned_data['action_gripper_right_cmd_timestamps'] = np.array(timestamps['action_gripper_right_pos_cmd'])

    def compute_gripper_action(obs):
        if obs is None: return None
        action = np.zeros_like(obs)
        action[:-1] = obs[1:]
        action[-1] = obs[-1]
        action = np.where(action > 50, action, action * 0.9)
        return action

    aligned_data['action_gripper_left_pos'] = compute_gripper_action(aligned_data['obs_gripper_left_pos'])
    aligned_data['action_gripper_right_pos'] = compute_gripper_action(aligned_data['obs_gripper_right_pos'])

    # 3. DECODING
    print(f"    [3/4] Decoding & Resizing images ({num_workers} workers)...")
    
    def get_selected_bytes(key):
        ts_list = timestamps[key]
        data_list = raw_data[key]
        if not ts_list: return []
        return [data_list[get_nearest_index(ts_list, t)] for t in uniform_timeline]

    rgb_keys = ['camera_head_rgb', 'camera_left_rgb', 'camera_right_rgb']
    depth_keys = ['camera_head_depth', 'camera_left_depth', 'camera_right_depth']
    
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = {}

    for key in rgb_keys:
        bytes_list = get_selected_bytes(key)
        if not bytes_list: continue
        chunk_size = len(bytes_list) // num_workers + 1
        for i in range(0, len(bytes_list), chunk_size):
            chunk = bytes_list[i:i + chunk_size]
            futures[executor.submit(decode_rgb_chunk, chunk)] = (key, i)

    for key in depth_keys:
        bytes_list = get_selected_bytes(key)
        if not bytes_list: continue
        prefix = key.split('_')[1]
        args = (bytes_list, meta[f'{prefix}_w'], meta[f'{prefix}_h'], meta[f'{prefix}_enc'])
        
        # 深度图解码参数切片 (args需要特殊处理，因为args[0]是list)
        chunk_size = len(bytes_list) // num_workers + 1
        for i in range(0, len(bytes_list), chunk_size):
            chunk_bytes = bytes_list[i:i + chunk_size]
            chunk_args = (chunk_bytes, args[1], args[2], args[3])
            futures[executor.submit(decode_depth_chunk, chunk_args)] = (key, i)

    # Initialize lists
    for key in rgb_keys + depth_keys:
        if raw_data[key]: aligned_data['obs_' + key] = [None] * num_samples

    with tqdm(total=len(futures), leave=False) as pbar:
        for future in as_completed(futures):
            key, start_idx = futures[future]
            try:
                decoded_chunk = future.result()
                final_key = 'obs_' + key
                for j, img in enumerate(decoded_chunk):
                    if start_idx + j < num_samples:
                        aligned_data[final_key][start_idx + j] = img
            except Exception as e:
                print(f"    Error decoding {key}: {e}")
            pbar.update(1)
    
    executor.shutdown()

    # To Numpy
    for key in rgb_keys + depth_keys:
        obs_key = 'obs_' + key
        if obs_key in aligned_data and aligned_data[obs_key] is not None:
             valid = [img for img in aligned_data[obs_key] if img is not None]
             if valid: aligned_data[obs_key] = np.array(valid)
             else: del aligned_data[obs_key]

    # 4. SAVE
    print(f"    [4/4] Saving to HDF5...")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('timestamps', data=aligned_data['timestamps'])
        obs_group = f.create_group('obs')
        action_group = f.create_group('action')

        def save_ds(group, name, data, compress=False):
            if data is None or len(data) == 0: return
            if compress: group.create_dataset(name, data=data, compression='gzip', compression_opts=4)
            else: group.create_dataset(name, data=data)

        for cam in ['head', 'left', 'right']:
            save_ds(obs_group, f'camera_{cam}/rgb', aligned_data.get(f'obs_camera_{cam}_rgb'), True)
            save_ds(obs_group, f'camera_{cam}/depth', aligned_data.get(f'obs_camera_{cam}_depth'), True)

        for side in ['left', 'right']:
            save_ds(obs_group, f'arm_{side}/joint_pos', aligned_data.get(f'obs_arm_{side}_pos'))
            save_ds(obs_group, f'gripper_{side}/joint_pos', aligned_data.get(f'obs_gripper_{side}_pos'))
            save_ds(action_group, f'arm_{side}/joint_pos', aligned_data.get(f'action_arm_{side}_pos'))
            save_ds(action_group, f'gripper_{side}/joint_pos', aligned_data.get(f'action_gripper_{side}_pos'))
            
            if f'action_gripper_{side}_cmd' in aligned_data:
                save_ds(action_group, f'gripper_{side}/commanded_pos', aligned_data[f'action_gripper_{side}_cmd'])
                save_ds(action_group, f'gripper_{side}/commanded_pos_timestamps', aligned_data[f'action_gripper_{side}_cmd_timestamps'])

        f.attrs['source_bag'] = str(bag_path.name)
        f.attrs['fps'] = TARGET_FPS
        f.attrs['duration'] = duration

    elapsed = time.time() - start_time
    print(f"    Done! Time: {elapsed/60:.2f} min, Size: {output_path.stat().st_size / 1024**3:.2f} GB")
    return True

# --- 批量处理逻辑 ---

def batch_process(input_path, output_path, workers):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有 bag 文件
    bag_files = sorted(list(input_path.glob("*.bag")))
    print(f"Found {len(bag_files)} bag files in {input_path}")
    
    success_count = 0
    for i, bag_file in enumerate(bag_files):
        h5_name = bag_file.with_suffix('.h5').name
        h5_path = output_path / h5_name
        
        print(f"\n[{i+1}/{len(bag_files)}] Starting conversion...")
        if h5_path.exists():
            print(f"    Skipping {h5_name} (Already exists)")
            continue
            
        if bag_to_h5(bag_file, h5_path, workers):
            success_count += 1
            
    print(f"\nBatch processing complete! Converted {success_count}/{len(bag_files)} files.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch convert ROS bags to resized HDF5')
    parser.add_argument('input', type=str, help='Input directory or file')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output directory or file')
    parser.add_argument('--workers', '-w', type=int, default=4, help='Number of CPU workers per file')
    parser.add_argument('--batch', '-b', action='store_true', help='Enable batch directory processing')
    
    args = parser.parse_args()
    input_path = Path(args.input)
    
    # 自动判断是否为文件夹
    is_dir = input_path.is_dir() or args.batch
    
    if is_dir:
        batch_process(args.input, args.output, args.workers)
    else:
        # 单文件模式
        output_file = Path(args.output)
        if output_file.is_dir():
            output_file = output_file / input_path.with_suffix('.h5').name
        bag_to_h5(input_path, output_file, args.workers)
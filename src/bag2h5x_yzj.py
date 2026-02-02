#!/usr/bin/env python3
"""
Convert ROS bag files to HDF5 format
Extracts RGB images, depth images, and joint states from multiple cameras and robot arms
python3 bag2h5x_yzj.py /home/pine/yzj/pour/r1_data_20260127_154637.bag --output /home/pine/yzj/h5out/r1_data_20260127_154637.h5

"""

import rosbag
import h5py
import numpy as np
import cv2  # Replaced cv_bridge with direct OpenCV usage
from pathlib import Path
from tqdm import tqdm
import argparse
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

def raw_depth_to_cv2(msg):
    """
    纯 Python 实现：将 ROS 深度图 (Raw) 转为 Numpy 数组
    替代 bridge.imgmsg_to_cv2，避免 Boost 版本冲突
    """
    # 深度图通常是 16UC1 (16-bit unsigned int, 单位通常是毫米)
    if '16UC1' in msg.encoding or 'mono16' in msg.encoding:
        dtype = np.uint16
        n_channels = 1
    elif '32FC1' in msg.encoding:
        dtype = np.float32
        n_channels = 1
    elif '8UC1' in msg.encoding or 'mono8' in msg.encoding:
        dtype = np.uint8
        n_channels = 1
    else:
        # 默认尝试 16位，如果格式很特殊可能会需要调整
        dtype = np.uint16
        n_channels = 1

    # 从字节流解析
    img = np.frombuffer(msg.data, dtype=dtype)
    
    # 重塑形状
    try:
        if n_channels == 1:
            img = img.reshape(msg.height, msg.width)
        else:
            img = img.reshape(msg.height, msg.width, n_channels)
    except ValueError as e:
        print(f"Error reshaping image: {e}. Msg shape: {msg.height}x{msg.width}, Data len: {len(msg.data)}")
        return None
        
    return img

def compressed_rgb_to_cv2(msg):
    """
    纯 Python 实现：将 ROS 压缩图像 (Compressed) 转为 Numpy 数组 (RGB)
    替代 bridge.compressed_imgmsg_to_cv2
    """
    try:
        np_arr = np.frombuffer(msg.data, np.uint8)
        # OpenCV 默认读入是 BGR，需要转成 RGB
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception as e:
        print(f"Error decoding compressed image: {e}")
        return None

def bag_to_h5(bag_path, output_path=None):
    """
    Convert a ROS bag file to HDF5 format
    """
    bag_path = Path(bag_path)
    start_time = time.time()
    
    if output_path is None:
        output_path = bag_path.with_suffix('.h5')
    else:
        output_path = Path(output_path)
    
    print(f"Converting: {bag_path}")
    print(f"Output: {output_path}")
    
    # Data storage
    data = {
        'camera_head_rgb': [],
        'camera_head_depth': [],
        'camera_left_rgb': [],
        'camera_left_depth': [],
        'camera_right_rgb': [],
        'camera_right_depth': [],
        # Observation (feedback)
        'arm_left_pos': [],
        'arm_right_pos': [],
        'gripper_left_pos': [],
        'gripper_right_pos': [],
        'arm_left_timestamps': [],
        'arm_right_timestamps': [],
        'gripper_left_timestamps': [],
        'gripper_right_timestamps': [],
        # Action (motion target)
        'action_arm_left_pos': [],
        'action_arm_right_pos': [],
        'action_arm_left_timestamps': [],
        'action_arm_right_timestamps': [],
        # Gripper action (motion control)
        'action_gripper_left_pos_cmd': [],
        'action_gripper_right_pos_cmd': [],
        'action_gripper_left_timestamps_cmd': [],
        'action_gripper_right_timestamps_cmd': [],
        # Camera timestamps
        'camera_head_rgb_timestamps': [],
        'camera_head_depth_timestamps': [],
        'camera_left_rgb_timestamps': [],
        'camera_left_depth_timestamps': [],
        'camera_right_rgb_timestamps': [],
        'camera_right_depth_timestamps': [],
    }
    
    # Open bag file
    bag = rosbag.Bag(str(bag_path))
    
    # Get total message count
    info = bag.get_type_and_topic_info()[1]
    total_msgs = sum(info[topic].message_count for topic in info.keys() if topic in info)
    
    print(f"Total messages: {total_msgs}")
    
    # Read messages
    with tqdm(total=total_msgs, desc="Reading bag") as pbar:
        for topic, msg, t in bag.read_messages():
            
            # Camera head RGB
            if topic == '/hdas/camera_head/rgb/image_rect_color/compressed':
                img = compressed_rgb_to_cv2(msg) # Replaced bridge
                if img is not None:
                    data['camera_head_rgb'].append(img)
                    data['camera_head_rgb_timestamps'].append(t.to_sec())
                
            # Camera head depth
            elif topic == '/hdas/camera_head/depth/depth_registered':
                depth = raw_depth_to_cv2(msg) # Replaced bridge
                if depth is not None:
                    data['camera_head_depth'].append(depth)
                    data['camera_head_depth_timestamps'].append(t.to_sec())
                
            # Left camera RGB
            elif topic == '/left/camera/color/image_raw/compressed':
                img = compressed_rgb_to_cv2(msg) # Replaced bridge
                if img is not None:
                    data['camera_left_rgb'].append(img)
                    data['camera_left_rgb_timestamps'].append(t.to_sec())
                
            # Left camera depth
            elif topic == '/left/camera/depth/image_rect_raw':
                depth = raw_depth_to_cv2(msg) # Replaced bridge
                if depth is not None:
                    data['camera_left_depth'].append(depth)
                    data['camera_left_depth_timestamps'].append(t.to_sec())
                
            # Right camera RGB
            elif topic == '/right/camera/color/image_raw/compressed':
                img = compressed_rgb_to_cv2(msg) # Replaced bridge
                if img is not None:
                    data['camera_right_rgb'].append(img)
                    data['camera_right_rgb_timestamps'].append(t.to_sec())
                
            # Right camera depth
            elif topic == '/right/camera/depth/image_rect_raw':
                depth = raw_depth_to_cv2(msg) # Replaced bridge
                if depth is not None:
                    data['camera_right_depth'].append(depth)
                    data['camera_right_depth_timestamps'].append(t.to_sec())
                
            # Left arm joint positions (observation)
            elif topic == '/hdas/feedback_arm_left':
                data['arm_left_pos'].append(msg.position)
                data['arm_left_timestamps'].append(t.to_sec())
                
            # Right arm joint positions (observation)
            elif topic == '/hdas/feedback_arm_right':
                data['arm_right_pos'].append(msg.position)
                data['arm_right_timestamps'].append(t.to_sec())
           
            # Left gripper joint positions (observation)
            elif topic == '/hdas/feedback_gripper_left':
                data['gripper_left_pos'].append(msg.position)
                data['gripper_left_timestamps'].append(t.to_sec())
                
            # Right gripper joint positions (observation)
            elif topic == '/hdas/feedback_gripper_right':
                data['gripper_right_pos'].append(msg.position)
                data['gripper_right_timestamps'].append(t.to_sec())
           
            # Left arm motion target (action)
            elif topic == '/motion_target/target_joint_state_arm_left':
                data['action_arm_left_pos'].append(msg.position)
                data['action_arm_left_timestamps'].append(t.to_sec())
                
            # Right arm motion target (action)
            elif topic == '/motion_target/target_joint_state_arm_right':
                data['action_arm_right_pos'].append(msg.position)
                data['action_arm_right_timestamps'].append(t.to_sec())
           
            # Gripper left action (motion control)
            elif topic == '/motion_control/position_control_gripper_left':
                data['action_gripper_left_pos_cmd'].append(msg.data)
                data['action_gripper_left_timestamps_cmd'].append(t.to_sec())
            # Gripper right action (motion control)
            elif topic == '/motion_control/position_control_gripper_right':
                data['action_gripper_right_pos_cmd'].append(msg.data)
                data['action_gripper_right_timestamps_cmd'].append(t.to_sec())

            pbar.update(1)
    
    bag.close()
    
    # Convert to numpy arrays
    print("Converting to numpy arrays...")
    for key in data:
        if len(data[key]) > 0:
            # Handle list of arrays (images) differently than list of lists (joints)
            try:
                data[key] = np.array(data[key])
                print(f"  {key}: {data[key].shape}")
            except Exception as e:
                print(f"  Warning: Could not convert {key} to numpy array directly. {e}")

    # Align to 15Hz timeline
    print("\nAligning data to 15Hz timeline...")
    TARGET_FPS = 15
    
    # Find the time range
    all_timestamps = []
    # Collect all timestamps from all sources
    for key in data:
        if 'timestamps' in key and len(data[key]) > 0:
            all_timestamps.extend(data[key])
    
    if len(all_timestamps) == 0:
        print("Error: No timestamps found!")
        return
    
    t_start = min(all_timestamps)
    t_end = max(all_timestamps)
    duration = t_end - t_start
    
    # Create uniform timeline at 15Hz
    num_samples = int(duration * TARGET_FPS)
    if num_samples <= 0:
        print("Warning: Duration too short, creating at least 1 frame.")
        num_samples = 1
        
    uniform_timeline = np.linspace(t_start, t_end, num_samples)
    
    print(f"  Duration: {duration:.2f}s")
    print(f"  Resampled to: {num_samples} frames @ {TARGET_FPS}Hz")
    
    # Resample data - organized into obs and action
    aligned_data = {
        'timestamps': uniform_timeline,
        # Observations
        'obs_arm_left_pos': None,
        'obs_arm_right_pos': None,
        'obs_gripper_left_pos': None,
        'obs_gripper_right_pos': None,
        'obs_camera_head_rgb': [],
        'obs_camera_head_depth': [],
        'obs_camera_left_rgb': [],
        'obs_camera_left_depth': [],
        'obs_camera_right_rgb': [],
        'obs_camera_right_depth': [],
        # Actions (motion targets)
        'action_arm_left_pos': None,
        'action_arm_right_pos': None,
        'action_gripper_left_pos': None,
        'action_gripper_right_pos': None,
    }

    # --- FIX: Moved Command saving here (after aligned_data is created) ---
    if len(data['action_gripper_left_pos_cmd']) > 0:
        aligned_data['action_gripper_left_cmd'] = data['action_gripper_left_pos_cmd']
        aligned_data['action_gripper_left_cmd_timestamps'] = data['action_gripper_left_timestamps_cmd']
    if len(data['action_gripper_right_pos_cmd']) > 0:
        aligned_data['action_gripper_right_cmd'] = data['action_gripper_right_pos_cmd']
        aligned_data['action_gripper_right_cmd_timestamps'] = data['action_gripper_right_timestamps_cmd']
    # ---------------------------------------------------------------------
    
    # Interpolate observation joint positions
    def safe_interp(timestamps, values, target_time):
        if len(timestamps) == 0: return None
        if len(timestamps) == 1:
            # If only one point, repeat it
            return np.repeat(values[0][np.newaxis, :], len(target_time), axis=0)
            
        f = interp1d(timestamps, values, axis=0, kind='linear', 
                     bounds_error=False, fill_value=(values[0], values[-1]))
        return f(target_time)

    if len(data['arm_left_pos']) > 0:
        print("  Interpolating left arm joint positions (obs)...")
        aligned_data['obs_arm_left_pos'] = safe_interp(data['arm_left_timestamps'], data['arm_left_pos'], uniform_timeline)
    
    if len(data['arm_right_pos']) > 0:
        print("  Interpolating right arm joint positions (obs)...")
        aligned_data['obs_arm_right_pos'] = safe_interp(data['arm_right_timestamps'], data['arm_right_pos'], uniform_timeline)
    
    if len(data['gripper_left_pos']) > 0:
        print("  Interpolating left gripper joint positions (obs)...")
        aligned_data['obs_gripper_left_pos'] = safe_interp(data['gripper_left_timestamps'], data['gripper_left_pos'], uniform_timeline)
    
    if len(data['gripper_right_pos']) > 0:
        print("  Interpolating right gripper joint positions (obs)...")
        aligned_data['obs_gripper_right_pos'] = safe_interp(data['gripper_right_timestamps'], data['gripper_right_pos'], uniform_timeline)
    
    # Interpolate action joint positions (motion targets)
    if len(data['action_arm_left_pos']) > 0:
        print("  Interpolating left arm joint positions (action)...")
        aligned_data['action_arm_left_pos'] = safe_interp(data['action_arm_left_timestamps'], data['action_arm_left_pos'], uniform_timeline)
    
    if len(data['action_arm_right_pos']) > 0:
        print("  Interpolating right arm joint positions (action)...")
        aligned_data['action_arm_right_pos'] = safe_interp(data['action_arm_right_timestamps'], data['action_arm_right_pos'], uniform_timeline)
    
    # Compute gripper action logic
    def compute_gripper_action(obs_data):
        """Compute gripper action from obs: action[t] = mapped(obs[t+1])"""
        if obs_data is None:
            return None
        action_data = np.zeros_like(obs_data)
        # action[t] = obs[t+1] for t < T-1
        action_data[:-1] = obs_data[1:]
        # action[T-1] = obs[T-1] (last frame)
        action_data[-1] = obs_data[-1]
        # Apply mapping: >50 unchanged, <=50 multiply by 0.9
        action_data = np.where(action_data > 50, action_data, action_data * 0.9)
        return action_data
    
    if aligned_data['obs_gripper_left_pos'] is not None:
        print("  Computing left gripper action (obs[t+1] with mapping)...")
        aligned_data['action_gripper_left_pos'] = compute_gripper_action(aligned_data['obs_gripper_left_pos'])
    
    if aligned_data['obs_gripper_right_pos'] is not None:
        print("  Computing right gripper action (obs[t+1] with mapping)...")
        aligned_data['action_gripper_right_pos'] = compute_gripper_action(aligned_data['obs_gripper_right_pos'])
    
    # Resample images (nearest neighbor for frames)
    def resample_images(images, timestamps, target_timeline):
        """Select nearest image for each target timestamp"""
        if len(images) == 0: return []
        timestamps = np.array(timestamps)
        resampled = []
        for t_target in tqdm(target_timeline, desc="  Resampling images", leave=False):
            # Find closest timestamp index
            idx = np.argmin(np.abs(timestamps - t_target))
            resampled.append(images[idx])
        return np.array(resampled)
    
    if len(data['camera_head_rgb']) > 0:
        print("  Resampling camera head RGB...")
        aligned_data['obs_camera_head_rgb'] = resample_images(
            data['camera_head_rgb'], data['camera_head_rgb_timestamps'], uniform_timeline)
    
    if len(data['camera_head_depth']) > 0:
        print("  Resampling camera head depth...")
        aligned_data['obs_camera_head_depth'] = resample_images(
            data['camera_head_depth'], data['camera_head_depth_timestamps'], uniform_timeline)
    
    if len(data['camera_left_rgb']) > 0:
        print("  Resampling left camera RGB...")
        aligned_data['obs_camera_left_rgb'] = resample_images(
            data['camera_left_rgb'], data['camera_left_rgb_timestamps'], uniform_timeline)
    
    if len(data['camera_left_depth']) > 0:
        print("  Resampling left camera depth...")
        aligned_data['obs_camera_left_depth'] = resample_images(
            data['camera_left_depth'], data['camera_left_depth_timestamps'], uniform_timeline)
    
    if len(data['camera_right_rgb']) > 0:
        print("  Resampling right camera RGB...")
        aligned_data['obs_camera_right_rgb'] = resample_images(
            data['camera_right_rgb'], data['camera_right_rgb_timestamps'], uniform_timeline)
    
    if len(data['camera_right_depth']) > 0:
        print("  Resampling right camera depth...")
        aligned_data['obs_camera_right_depth'] = resample_images(
            data['camera_right_depth'], data['camera_right_depth_timestamps'], uniform_timeline)
    
    print("\nAligned data shapes:")
    for key in aligned_data:
        val = aligned_data[key]
        if val is not None and isinstance(val, np.ndarray) and len(val) > 0:
            print(f"  {key}: {val.shape}")
        elif isinstance(val, list) and len(val) > 0:
            print(f"  {key}: List with {len(val)} elements")
    
    # Save to HDF5
    print(f"\nSaving to {output_path}...")
    
    with h5py.File(output_path, 'w') as f:
        # Save timestamps
        f.create_dataset('timestamps', data=aligned_data['timestamps'])
        
        # Create obs and action groups
        obs_group = f.create_group('obs')
        action_group = f.create_group('action')
        
        # Helper function to save large datasets with progress
        def save_with_progress(group, name, data, compression=None, compression_opts=None):
            if data is None or len(data) == 0: return
            
            # Ensure data is numpy array
            if isinstance(data, list): data = np.array(data)
            
            shape = data.shape
            
            # Create dataset
            if compression:
                dataset = group.create_dataset(name, shape=shape, dtype=data.dtype,
                                             compression=compression, compression_opts=compression_opts)
            else:
                dataset = group.create_dataset(name, shape=shape, dtype=data.dtype)
            
            # Write in chunks with progress
            chunk_size = 10  # frames per chunk
            num_chunks = (len(data) + chunk_size - 1) // chunk_size
            
            with tqdm(total=num_chunks, desc=f"  Saving {name}", leave=False) as pbar:
                for i in range(0, len(data), chunk_size):
                    end = min(i + chunk_size, len(data))
                    dataset[i:end] = data[i:end]
                    pbar.update(1)
        
        # Save observation images
        print("Saving obs/camera_head data...")
        save_with_progress(obs_group, 'camera_head/rgb', aligned_data['obs_camera_head_rgb'], compression='gzip', compression_opts=4)
        save_with_progress(obs_group, 'camera_head/depth', aligned_data['obs_camera_head_depth'], compression='gzip', compression_opts=4)
        
        print("Saving obs/camera_left data...")
        save_with_progress(obs_group, 'camera_left/rgb', aligned_data['obs_camera_left_rgb'], compression='gzip', compression_opts=4)
        save_with_progress(obs_group, 'camera_left/depth', aligned_data['obs_camera_left_depth'], compression='gzip', compression_opts=4)
        
        print("Saving obs/camera_right data...")
        save_with_progress(obs_group, 'camera_right/rgb', aligned_data['obs_camera_right_rgb'], compression='gzip', compression_opts=4)
        save_with_progress(obs_group, 'camera_right/depth', aligned_data['obs_camera_right_depth'], compression='gzip', compression_opts=4)
        
        # Save observation joint positions
        print("Saving obs joint data...")
        if aligned_data['obs_arm_left_pos'] is not None:
            obs_group.create_dataset('arm_left/joint_pos', data=aligned_data['obs_arm_left_pos'])
        if aligned_data['obs_arm_right_pos'] is not None:
            obs_group.create_dataset('arm_right/joint_pos', data=aligned_data['obs_arm_right_pos'])
        if aligned_data['obs_gripper_left_pos'] is not None:
            obs_group.create_dataset('gripper_left/joint_pos', data=aligned_data['obs_gripper_left_pos'])
        if aligned_data['obs_gripper_right_pos'] is not None:
            obs_group.create_dataset('gripper_right/joint_pos', data=aligned_data['obs_gripper_right_pos'])
        
        # Save action joint positions
        print("Saving action joint data...")
        if aligned_data['action_arm_left_pos'] is not None:
            action_group.create_dataset('arm_left/joint_pos', data=aligned_data['action_arm_left_pos'])
        if aligned_data['action_arm_right_pos'] is not None:
            action_group.create_dataset('arm_right/joint_pos', data=aligned_data['action_arm_right_pos'])
        if aligned_data['action_gripper_left_pos'] is not None:
            action_group.create_dataset('gripper_left/joint_pos', data=aligned_data['action_gripper_left_pos'])
        if aligned_data['action_gripper_right_pos'] is not None:
            action_group.create_dataset('gripper_right/joint_pos', data=aligned_data['action_gripper_right_pos'])
            
        # Save original command actions
        if 'action_gripper_left_cmd' in aligned_data:
            action_group.create_dataset('gripper_left/commanded_pos', data=aligned_data['action_gripper_left_cmd'])
        if 'action_gripper_left_cmd_timestamps' in aligned_data:
            action_group.create_dataset('gripper_left/commanded_pos_timestamps', data=aligned_data['action_gripper_left_cmd_timestamps'])
        if 'action_gripper_right_cmd' in aligned_data:
            action_group.create_dataset('gripper_right/commanded_pos', data=aligned_data['action_gripper_right_cmd'])
        if 'action_gripper_right_cmd_timestamps' in aligned_data:
            action_group.create_dataset('gripper_right/commanded_pos_timestamps', data=aligned_data['action_gripper_right_cmd_timestamps'])
        
        # Save metadata
        f.attrs['source_bag'] = str(bag_path.name)
        f.attrs['num_frames'] = num_samples
        f.attrs['fps'] = TARGET_FPS
        f.attrs['duration'] = duration
    
    elapsed = time.time() - start_time
    print(f"✓ Conversion complete: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024**3:.2f} GB")
    print(f"  Time elapsed: {elapsed/60:.1f} minutes")

def _convert_single_file(args):
    """Wrapper function for parallel processing"""
    bag_file, output_file = args
    try:
        bag_to_h5(bag_file, output_file)
        return (bag_file.name, True, None)
    except Exception as e:
        return (bag_file.name, False, str(e))

def batch_convert(bag_dir, output_dir=None, num_workers=1):
    """Convert all .bag files in a directory"""
    bag_dir = Path(bag_dir)
    
    if output_dir is None:
        output_dir = bag_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    bag_files = sorted(bag_dir.glob('*.bag'))
    print(f"Found {len(bag_files)} bag files in {bag_dir}")
    
    tasks = []
    for bag_file in bag_files:
        output_file = output_dir / bag_file.with_suffix('.h5').name
        if output_file.exists():
            print(f"Skipping {bag_file.name} (already converted)")
            continue
        tasks.append((bag_file, output_file))
    
    if len(tasks) == 0:
        print("All files already converted!")
        return
    
    print(f"Converting {len(tasks)} files with {num_workers} worker(s)...\n")
    
    batch_start_time = time.time()
    completed = 0
    
    if num_workers == 1:
        for i, (bag_file, output_file) in enumerate(tasks):
            try:
                bag_to_h5(bag_file, output_file)
                completed += 1
                print(f"\n[Progress: {completed}/{len(tasks)}]\n")
            except Exception as e:
                print(f"✗ Error converting {bag_file.name}: {e}")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_convert_single_file, task): task for task in tasks}
            for future in as_completed(futures):
                filename, success, error = future.result()
                completed += 1
                if success:
                    print(f"\n✓ Completed {filename} [{completed}/{len(tasks)}]")
                else:
                    print(f"\n✗ Error converting {filename}: {error}")
    
    print(f"\n{'='*60}")
    print(f"Batch conversion complete!")
    print(f"Converted: {completed}/{len(tasks)} files")
    print(f"Total time: {(time.time() - batch_start_time)/60:.1f} minutes")
    print(f"{'='*60}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert ROS bag files to HDF5')
    parser.add_argument('input', type=str, help='Input .bag file or directory')
    parser.add_argument('--output', '-o', type=str, help='Output .h5 file or directory')
    parser.add_argument('--batch', '-b', action='store_true', help='Batch process directory')
    parser.add_argument('--workers', '-w', type=int, default=1, 
                       help='Number of parallel workers for batch processing (default: 1)')
    
    args = parser.parse_args()
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        max_workers = multiprocessing.cpu_count()
        num_workers = min(args.workers, max_workers)
        batch_convert(input_path, args.output, num_workers)
    else:
        bag_to_h5(input_path, args.output)
#!/usr/bin/env python3
"""
Convert ROS bag files to HDF5 format
Extracts RGB images, depth images, and joint states from multiple cameras and robot arms
"""

import rosbag
import h5py
import numpy as np
from cv_bridge import CvBridge
from pathlib import Path
from tqdm import tqdm
import argparse
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time


def bag_to_h5(bag_path, output_path=None):
    """
    Convert a ROS bag file to HDF5 format
    
    Args:
        bag_path: Path to the .bag file
        output_path: Output .h5 file path (default: same name as bag)
    """
    bag_path = Path(bag_path)
    start_time = time.time()
    
    if output_path is None:
        output_path = bag_path.with_suffix('.h5')
    else:
        output_path = Path(output_path)
    
    print(f"Converting: {bag_path}")
    print(f"Output: {output_path}")
    
    bridge = CvBridge()
    
    # Data storage
    data = {
        'camera_head_rgb': [],
        'camera_head_depth': [],
        'camera_left_rgb': [],
        'camera_left_depth': [],
        'camera_right_rgb': [],
        'camera_right_depth': [],
        'arm_left_pos': [],
        'arm_right_pos': [],
        'arm_left_timestamps': [],
        'arm_right_timestamps': [],
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
    total_msgs = sum(info[topic].message_count for topic in info.keys())
    
    print(f"Total messages: {total_msgs}")
    
    # Read messages
    with tqdm(total=total_msgs, desc="Reading bag") as pbar:
        for topic, msg, t in bag.read_messages():
            
            # Camera head RGB
            if topic == '/hdas/camera_head/rgb/image_rect_color/compressed':
                img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8')
                data['camera_head_rgb'].append(img)
                data['camera_head_rgb_timestamps'].append(t.to_sec())
                
            # Camera head depth
            elif topic == '/hdas/camera_head/depth/depth_registered':
                depth = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                data['camera_head_depth'].append(depth)
                data['camera_head_depth_timestamps'].append(t.to_sec())
                
            # Left camera RGB
            elif topic == '/left/camera/color/image_raw/compressed':
                img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8')
                data['camera_left_rgb'].append(img)
                data['camera_left_rgb_timestamps'].append(t.to_sec())
                
            # Left camera depth
            elif topic == '/left/camera/depth/image_rect_raw':
                depth = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                data['camera_left_depth'].append(depth)
                data['camera_left_depth_timestamps'].append(t.to_sec())
                
            # Right camera RGB
            elif topic == '/right/camera/color/image_raw/compressed':
                img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8')
                data['camera_right_rgb'].append(img)
                data['camera_right_rgb_timestamps'].append(t.to_sec())
                
            # Right camera depth
            elif topic == '/right/camera/depth/image_rect_raw':
                depth = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                data['camera_right_depth'].append(depth)
                data['camera_right_depth_timestamps'].append(t.to_sec())
                
            # Left arm joint positions
            elif topic == '/hdas/feedback_arm_left':
                data['arm_left_pos'].append(msg.position)
                data['arm_left_timestamps'].append(t.to_sec())
                
            # Right arm joint positions
            elif topic == '/hdas/feedback_arm_right':
                data['arm_right_pos'].append(msg.position)
                data['arm_right_timestamps'].append(t.to_sec())
            
            pbar.update(1)
    
    bag.close()
    
    # Convert to numpy arrays
    print("Converting to numpy arrays...")
    for key in data:
        if len(data[key]) > 0:
            data[key] = np.array(data[key])
            print(f"  {key}: {data[key].shape}")
    
    # Align to 15Hz timeline
    print("\nAligning data to 15Hz timeline...")
    TARGET_FPS = 15
    
    # Find the time range
    all_timestamps = []
    if len(data['camera_head_rgb_timestamps']) > 0:
        all_timestamps.extend(data['camera_head_rgb_timestamps'])
    if len(data['camera_head_depth_timestamps']) > 0:
        all_timestamps.extend(data['camera_head_depth_timestamps'])
    if len(data['camera_left_rgb_timestamps']) > 0:
        all_timestamps.extend(data['camera_left_rgb_timestamps'])
    if len(data['camera_left_depth_timestamps']) > 0:
        all_timestamps.extend(data['camera_left_depth_timestamps'])
    if len(data['camera_right_rgb_timestamps']) > 0:
        all_timestamps.extend(data['camera_right_rgb_timestamps'])
    if len(data['camera_right_depth_timestamps']) > 0:
        all_timestamps.extend(data['camera_right_depth_timestamps'])
    if len(data['arm_left_timestamps']) > 0:
        all_timestamps.extend(data['arm_left_timestamps'])
    if len(data['arm_right_timestamps']) > 0:
        all_timestamps.extend(data['arm_right_timestamps'])
    
    if len(all_timestamps) == 0:
        print("Error: No timestamps found!")
        return
    
    t_start = min(all_timestamps)
    t_end = max(all_timestamps)
    duration = t_end - t_start
    
    # Create uniform timeline at 15Hz
    num_samples = int(duration * TARGET_FPS)
    uniform_timeline = np.linspace(t_start, t_end, num_samples)
    
    print(f"  Duration: {duration:.2f}s")
    print(f"  Resampled to: {num_samples} frames @ {TARGET_FPS}Hz")
    
    # Resample data
    aligned_data = {
        'timestamps': uniform_timeline,
        'arm_left_pos': None,
        'arm_right_pos': None,
        'camera_head_rgb': [],
        'camera_head_depth': [],
        'camera_left_rgb': [],
        'camera_left_depth': [],
        'camera_right_rgb': [],
        'camera_right_depth': [],
    }
    
    # Interpolate joint positions
    if len(data['arm_left_pos']) > 0:
        print("  Interpolating left arm joint positions...")
        interp_funcs = [interp1d(data['arm_left_timestamps'], data['arm_left_pos'][:, i], 
                                kind='linear', bounds_error=False, fill_value='extrapolate')
                       for i in range(data['arm_left_pos'].shape[1])]
        aligned_data['arm_left_pos'] = np.array([f(uniform_timeline) for f in interp_funcs]).T
    
    if len(data['arm_right_pos']) > 0:
        print("  Interpolating right arm joint positions...")
        interp_funcs = [interp1d(data['arm_right_timestamps'], data['arm_right_pos'][:, i], 
                                kind='linear', bounds_error=False, fill_value='extrapolate')
                       for i in range(data['arm_right_pos'].shape[1])]
        aligned_data['arm_right_pos'] = np.array([f(uniform_timeline) for f in interp_funcs]).T
    
    # Resample images (nearest neighbor for frames)
    def resample_images(images, timestamps, target_timeline):
        """Select nearest image for each target timestamp"""
        resampled = []
        for t_target in tqdm(target_timeline, desc="  Resampling images", leave=False):
            idx = np.argmin(np.abs(timestamps - t_target))
            resampled.append(images[idx])
        return np.array(resampled)
    
    if len(data['camera_head_rgb']) > 0:
        print("  Resampling camera head RGB...")
        aligned_data['camera_head_rgb'] = resample_images(
            data['camera_head_rgb'], data['camera_head_rgb_timestamps'], uniform_timeline)
    
    if len(data['camera_head_depth']) > 0:
        print("  Resampling camera head depth...")
        aligned_data['camera_head_depth'] = resample_images(
            data['camera_head_depth'], data['camera_head_depth_timestamps'], uniform_timeline)
    
    if len(data['camera_left_rgb']) > 0:
        print("  Resampling left camera RGB...")
        aligned_data['camera_left_rgb'] = resample_images(
            data['camera_left_rgb'], data['camera_left_rgb_timestamps'], uniform_timeline)
    
    if len(data['camera_left_depth']) > 0:
        print("  Resampling left camera depth...")
        aligned_data['camera_left_depth'] = resample_images(
            data['camera_left_depth'], data['camera_left_depth_timestamps'], uniform_timeline)
    
    if len(data['camera_right_rgb']) > 0:
        print("  Resampling right camera RGB...")
        aligned_data['camera_right_rgb'] = resample_images(
            data['camera_right_rgb'], data['camera_right_rgb_timestamps'], uniform_timeline)
    
    if len(data['camera_right_depth']) > 0:
        print("  Resampling right camera depth...")
        aligned_data['camera_right_depth'] = resample_images(
            data['camera_right_depth'], data['camera_right_depth_timestamps'], uniform_timeline)
    
    print("\nAligned data shapes:")
    for key in aligned_data:
        if aligned_data[key] is not None and len(aligned_data[key]) > 0:
            if isinstance(aligned_data[key], np.ndarray):
                print(f"  {key}: {aligned_data[key].shape}")
    
    # Save to HDF5
    print(f"\nSaving to {output_path}...")
    
    with h5py.File(output_path, 'w') as f:
        # Save timestamps (small, no need for progress)
        f.create_dataset('timestamps', data=aligned_data['timestamps'])
        
        # Helper function to save large datasets with progress
        def save_with_progress(name, data, compression=None, compression_opts=None):
            """Save dataset in chunks with progress bar"""
            if len(data) == 0:
                return
            
            # Create dataset
            if compression:
                dataset = f.create_dataset(name, shape=data.shape, dtype=data.dtype,
                                          compression=compression, compression_opts=compression_opts)
            else:
                dataset = f.create_dataset(name, shape=data.shape, dtype=data.dtype)
            
            # Write in chunks with progress
            chunk_size = 10  # frames per chunk
            num_chunks = (len(data) + chunk_size - 1) // chunk_size
            
            with tqdm(total=num_chunks, desc=f"  Saving {name}", leave=False) as pbar:
                for i in range(0, len(data), chunk_size):
                    end = min(i + chunk_size, len(data))
                    dataset[i:end] = data[i:end]
                    pbar.update(1)
        
        # Save images with compression
        print("Saving camera_head data...")
        if len(aligned_data['camera_head_rgb']) > 0:
            save_with_progress('camera_head/rgb', aligned_data['camera_head_rgb'],
                             compression='gzip', compression_opts=4)
        if len(aligned_data['camera_head_depth']) > 0:
            save_with_progress('camera_head/depth', aligned_data['camera_head_depth'],
                             compression='gzip', compression_opts=4)
        
        print("Saving camera_left data...")
        if len(aligned_data['camera_left_rgb']) > 0:
            save_with_progress('camera_left/rgb', aligned_data['camera_left_rgb'],
                             compression='gzip', compression_opts=4)
        if len(aligned_data['camera_left_depth']) > 0:
            save_with_progress('camera_left/depth', aligned_data['camera_left_depth'],
                             compression='gzip', compression_opts=4)
        
        print("Saving camera_right data...")
        if len(aligned_data['camera_right_rgb']) > 0:
            save_with_progress('camera_right/rgb', aligned_data['camera_right_rgb'],
                             compression='gzip', compression_opts=4)
        if len(aligned_data['camera_right_depth']) > 0:
            save_with_progress('camera_right/depth', aligned_data['camera_right_depth'],
                             compression='gzip', compression_opts=4)
        
        # Save joint positions (small, no progress needed)
        print("Saving joint data...")
        if aligned_data['arm_left_pos'] is not None:
            f.create_dataset('arm_left/joint_pos', data=aligned_data['arm_left_pos'])
            
        if aligned_data['arm_right_pos'] is not None:
            f.create_dataset('arm_right/joint_pos', data=aligned_data['arm_right_pos'])
        
        # Save metadata
        f.attrs['source_bag'] = str(bag_path.name)
        f.attrs['num_frames'] = num_samples
        f.attrs['fps'] = TARGET_FPS
        f.attrs['duration'] = duration
        f.attrs['start_time'] = t_start
        f.attrs['end_time'] = t_end
    
    elapsed = time.time() - start_time
    print(f"✓ Conversion complete: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024**3:.2f} GB")
    print(f"  Time elapsed: {elapsed/60:.1f} minutes")


def _convert_single_file(args):
    """
    Wrapper function for parallel processing
    Must be at module level for pickling
    """
    bag_file, output_file = args
    try:
        bag_to_h5(bag_file, output_file)
        return (bag_file.name, True, None)
    except Exception as e:
        return (bag_file.name, False, str(e))


def batch_convert(bag_dir, output_dir=None, num_workers=1):
    """
    Convert all .bag files in a directory
    
    Args:
        bag_dir: Directory containing .bag files
        output_dir: Output directory for .h5 files (default: same as bag_dir)
        num_workers: Number of parallel workers (default: 1, sequential)
    """
    bag_dir = Path(bag_dir)
    
    if output_dir is None:
        output_dir = bag_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    bag_files = sorted(bag_dir.glob('*.bag'))
    
    print(f"Found {len(bag_files)} bag files in {bag_dir}")
    
    # Prepare conversion tasks
    tasks = []
    for bag_file in bag_files:
        output_file = output_dir / bag_file.with_suffix('.h5').name
        
        # Skip if already converted
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
    times_per_file = []
    
    if num_workers == 1:
        # Sequential processing
        for i, (bag_file, output_file) in enumerate(tasks):
            file_start = time.time()
            try:
                bag_to_h5(bag_file, output_file)
                file_elapsed = time.time() - file_start
                times_per_file.append(file_elapsed)
                completed += 1
                
                # Calculate ETA
                avg_time = sum(times_per_file) / len(times_per_file)
                remaining = len(tasks) - completed
                eta_seconds = avg_time * remaining
                eta_str = f"{eta_seconds/60:.1f} min" if eta_seconds < 3600 else f"{eta_seconds/3600:.1f} hr"
                
                print(f"\n[Progress: {completed}/{len(tasks)}] ETA: {eta_str}\n")
            except Exception as e:
                print(f"✗ Error converting {bag_file.name}: {e}")
                continue
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_convert_single_file, task): task for task in tasks}
            
            for future in as_completed(futures):
                filename, success, error = future.result()
                completed += 1
                
                # Calculate ETA based on elapsed time
                elapsed = time.time() - batch_start_time
                avg_time = elapsed / completed
                remaining = len(tasks) - completed
                eta_seconds = avg_time * remaining
                eta_str = f"{eta_seconds/60:.1f} min" if eta_seconds < 3600 else f"{eta_seconds/3600:.1f} hr"
                
                if success:
                    print(f"\n✓ Completed {filename} [{completed}/{len(tasks)}] ETA: {eta_str}")
                else:
                    print(f"\n✗ Error converting {filename}: {error}")
    
    total_elapsed = time.time() - batch_start_time
    print(f"\n{'='*60}")
    print(f"Batch conversion complete!")
    print(f"Converted: {completed}/{len(tasks)} files")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
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
        # Limit workers to number of CPU cores
        max_workers = multiprocessing.cpu_count()
        num_workers = min(args.workers, max_workers)
        if args.workers > max_workers:
            print(f"Warning: Requested {args.workers} workers, but only {max_workers} CPU cores available.")
            print(f"Using {num_workers} workers instead.\n")
        
        batch_convert(input_path, args.output, num_workers)
    else:
        bag_to_h5(input_path, args.output)

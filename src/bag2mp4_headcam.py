#!/usr/bin/env python3
"""
ROS bag to MP4 Converter (Head Camera Only)
Target: /hdas/camera_head/rgb/image_rect_color/compressed
Output: MP4 Video (Resized to 640x480, 15 FPS)

Usage:
    Single file: python3 bag2mp4_headcam.py input.bag --output output.mp4
    Batch dir:   python3 bag2mp4_headcam.py /media/pine/新加卷/R1/pour --batch --output ./vis_head/

    python3 bag2mp4_headcam.py /media/pine/Yang/R1/pour_0201 --batch --output ./vis_head_0201/
    python3 bag2mp4_headcam.py /home/pine/yzj/pour --batch --output ./vis_head_0203/
"""

import rosbag
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

# --- 配置参数 ---
TARGET_TOPIC = '/hdas/camera_head/rgb/image_rect_color/compressed'
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
TARGET_FPS = 15  # 对应参考代码中的 15Hz 对齐频率

def bag_to_mp4(bag_path, output_path):
    bag_path = Path(bag_path)
    # 如果输出路径没有后缀，自动添加 .mp4
    if output_path.suffix.lower() != '.mp4':
        output_path = output_path.with_suffix('.mp4')

    print(f"\n>>> Processing: {bag_path.name}")
    print(f"    Output: {output_path}")

    bag = rosbag.Bag(str(bag_path))
    info = bag.get_type_and_topic_info()[1]
    
    # 检查 topic 是否存在
    if TARGET_TOPIC not in info:
        print(f"    Warning: Topic {TARGET_TOPIC} not found in bag. Skipping.")
        bag.close()
        return False

    total_msgs = info[TARGET_TOPIC].message_count
    
    # 初始化 VideoWriter
    # fourcc = cv2.VideoWriter_fourcc(*'avc1') # H.264 (如果系统支持，压缩率高)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 兼容性较好的 MP4 编码
    writer = cv2.VideoWriter(str(output_path), fourcc, TARGET_FPS, (TARGET_WIDTH, TARGET_HEIGHT))

    if not writer.isOpened():
        print("    Error: Could not open video writer.")
        bag.close()
        return False

    print(f"    [1/1] Converting images to MP4 ({total_msgs} frames)...")
    
    try:
        with tqdm(total=total_msgs, leave=False) as pbar:
            for topic, msg, t in bag.read_messages(topics=[TARGET_TOPIC]):
                # 1. 解码 (CompressedImage -> BGR)
                np_arr = np.frombuffer(msg.data, np.uint8)
                img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if img_bgr is not None:
                    # 2. Resize 到 640x480 (参考代码逻辑)
                    if img_bgr.shape[1] != TARGET_WIDTH or img_bgr.shape[0] != TARGET_HEIGHT:
                        img_bgr = cv2.resize(img_bgr, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
                    
                    # 3. 写入视频帧
                    writer.write(img_bgr)
                
                pbar.update(1)

    except Exception as e:
        print(f"    Error processing bag: {e}")
        return False
    finally:
        bag.close()
        writer.release()

    print(f"    Done! Saved to {output_path.name}")
    return True

# --- 批量处理逻辑 (复用原有风格) ---

def batch_process(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有 bag 文件
    bag_files = sorted(list(input_path.glob("*.bag")))
    print(f"Found {len(bag_files)} bag files in {input_path}")
    
    success_count = 0
    for i, bag_file in enumerate(bag_files):
        # 输出文件名保持同名，但后缀改为 .mp4
        mp4_name = bag_file.with_suffix('.mp4').name
        mp4_path = output_path / mp4_name
        
        print(f"\n[{i+1}/{len(bag_files)}] Starting conversion...")
        if mp4_path.exists():
            print(f"    Skipping {mp4_name} (Already exists)")
            continue
            
        if bag_to_mp4(bag_file, mp4_path):
            success_count += 1
            
    print(f"\nBatch processing complete! Converted {success_count}/{len(bag_files)} files.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert ROS bag headcam images to resized MP4')
    parser.add_argument('input', type=str, help='Input directory or file')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output directory or file')
    parser.add_argument('--batch', '-b', action='store_true', help='Enable batch directory processing')
    
    args = parser.parse_args()
    input_path = Path(args.input)
    
    # 自动判断是否为文件夹
    is_dir = input_path.is_dir() or args.batch
    
    if is_dir:
        batch_process(args.input, args.output)
    else:
        # 单文件模式
        output_file = Path(args.output)
        if output_file.is_dir():
            output_file = output_file / input_path.with_suffix('.mp4').name
        bag_to_mp4(input_path, output_file)
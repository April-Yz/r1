#!/usr/bin/env python3
"""
重组H5文件结构：将观测数据组织到obs group下，并添加action group
action[t] = obs[t+1]

原始结构:
/arm_left/joint_pos      Dataset {T, 7}
/arm_right/joint_pos     Dataset {T, 7}
/camera_head/depth       Dataset {T, 720, 1280}
/camera_head/rgb         Dataset {T, 720, 1280, 3}
/camera_left/depth       Dataset {T, 480, 640}
/camera_left/rgb         Dataset {T, 480, 640, 3}
/camera_right/depth      Dataset {T, 480, 640}
/camera_right/rgb        Dataset {T, 480, 640, 3}
/timestamps              Dataset {T}

新结构:
/obs/arm_left/joint_pos
/obs/arm_right/joint_pos
/obs/camera_head/depth, rgb
/obs/camera_left/depth, rgb  
/obs/camera_right/depth, rgb
/action/arm_left/joint_pos   (action[t] = obs[t+1])
/action/arm_right/joint_pos  (action[t] = obs[t+1])
/timestamps

Usage:
python add_action.py <input_file_or_folder> [-o <output_file_or_folder>] [--in-place] [--verify]
"""

import h5py
import numpy as np
from pathlib import Path
import argparse
import shutil
from tqdm import tqdm


def reorganize_h5_file(input_path: str, output_path: str = None, in_place: bool = False, verbose: bool = True, show_progress: bool = True):
    """
    重组H5文件结构
    
    Args:
        input_path: 输入H5文件路径
        output_path: 输出H5文件路径（如果为None且in_place为False，则在原文件名后加_reorganized）
        in_place: 是否原地修改（会先备份原文件）
        verbose: 是否打印详细信息
        show_progress: 是否显示进度条
    """
    input_path = Path(input_path)
    
    if in_place:
        # 备份原文件
        backup_path = input_path.with_suffix('.h5.bak')
        shutil.copy2(input_path, backup_path)
        output_path = input_path
        temp_path = input_path.with_suffix('.h5.tmp')
    else:
        backup_path = None
        if output_path is None:
            output_path = input_path.with_stem(input_path.stem + '_reorganized')
        else:
            output_path = Path(output_path)
        temp_path = output_path
    
    if verbose:
        print(f"Processing: {input_path}")
        print(f"Output: {output_path}")
    
    with h5py.File(input_path, 'r') as f_in:
        # 获取时间步数
        T = f_in['timestamps'].shape[0]
        if verbose:
            print(f"Total timesteps: {T}")
        
        with h5py.File(temp_path, 'w') as f_out:
            # 复制 timestamps
            f_out.create_dataset('timestamps', data=f_in['timestamps'][:])
            
            # 创建 obs 和 action groups
            obs_group = f_out.create_group('obs')
            action_group = f_out.create_group('action')
            
            # 计算总任务数用于进度条
            tasks = []
            for arm in ['arm_left', 'arm_right']:
                if arm in f_in and 'joint_pos' in f_in[arm]:
                    tasks.append(('arm', arm))
            for camera in ['camera_head', 'camera_left', 'camera_right']:
                if camera in f_in:
                    for data_type in ['rgb', 'depth']:
                        if data_type in f_in[camera]:
                            tasks.append(('camera', camera, data_type))
            
            # 使用进度条处理数据
            pbar = tqdm(tasks, desc=f"  {input_path.name}", leave=False, disable=not show_progress)
            for task in pbar:
                if task[0] == 'arm':
                    arm = task[1]
                    pbar.set_postfix_str(f"{arm}/joint_pos")
                    
                    if arm not in obs_group:
                        arm_obs_group = obs_group.create_group(arm)
                        arm_action_group = action_group.create_group(arm)
                    else:
                        arm_obs_group = obs_group[arm]
                        arm_action_group = action_group[arm]
                    
                    joint_pos = f_in[arm]['joint_pos'][:]
                    
                    # obs: 原始数据
                    arm_obs_group.create_dataset('joint_pos', data=joint_pos)
                    
                    # action: t时刻的action = t+1时刻的obs
                    action_data = np.zeros_like(joint_pos)
                    action_data[:-1] = joint_pos[1:]  # action[t] = obs[t+1]
                    action_data[-1] = joint_pos[-1]   # 最后一帧保持不变
                    arm_action_group.create_dataset('joint_pos', data=action_data)
                    
                elif task[0] == 'camera':
                    camera, data_type = task[1], task[2]
                    pbar.set_postfix_str(f"{camera}/{data_type}")
                    
                    if camera not in obs_group:
                        camera_obs_group = obs_group.create_group(camera)
                    else:
                        camera_obs_group = obs_group[camera]
                    
                    # 使用 h5py.copy() 直接复制数据集，避免读入内存，速度快很多
                    src_path = f"{camera}/{data_type}"
                    dst_name = data_type
                    f_in.copy(src_path, camera_obs_group, name=dst_name)
            
            if verbose:
                print(f"✓ Reorganized file saved to {output_path}")
    
    # 如果是原地修改，替换原文件并删除备份
    if in_place and temp_path != output_path:
        shutil.move(temp_path, output_path)
    
    # 删除备份文件
    if backup_path and backup_path.exists():
        backup_path.unlink()


def process_folder(folder_path: str, output_folder: str = None, in_place: bool = False):
    """处理文件夹中的所有H5文件"""
    folder_path = Path(folder_path)
    
    if output_folder:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    
    h5_files = sorted(folder_path.glob("*.h5"))
    print(f"Found {len(h5_files)} H5 files in {folder_path}")
    
    success_count = 0
    failed_files = []
    
    pbar = tqdm(h5_files, desc="Processing files", unit="file")
    for h5_file in pbar:
        pbar.set_postfix_str(h5_file.name[:30])
        
        if output_folder:
            output_path = output_folder / h5_file.name
        else:
            output_path = None
        
        try:
            reorganize_h5_file(str(h5_file), str(output_path) if output_path else None, in_place, verbose=False, show_progress=True)
            success_count += 1
        except Exception as e:
            failed_files.append((h5_file.name, str(e)))
            continue
    
    print(f"\n✓ Successfully processed {success_count}/{len(h5_files)} files")
    if failed_files:
        print(f"✗ Failed files:")
        for name, err in failed_files:
            print(f"  - {name}: {err}")


def verify_structure(h5_path: str):
    """验证H5文件的新结构"""
    print(f"\nVerifying structure of {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: {obj.shape} {obj.dtype}")
        
        f.visititems(print_structure)
        
        # 验证action是否正确
        if 'obs' in f and 'action' in f:
            for arm in ['arm_left', 'arm_right']:
                if arm in f['obs'] and arm in f['action']:
                    obs = f['obs'][arm]['joint_pos'][:]
                    action = f['action'][arm]['joint_pos'][:]
                    
                    # 检查 action[t] == obs[t+1]
                    if np.allclose(action[:-1], obs[1:]):
                        print(f"  ✓ {arm} action verification passed")
                    else:
                        print(f"  ✗ {arm} action verification FAILED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize H5 files: add obs/action groups")
    parser.add_argument("input", help="Input H5 file or folder path")
    parser.add_argument("-o", "--output", help="Output path (file or folder)")
    parser.add_argument("--in-place", action="store_true", help="Modify files in place (creates backup)")
    parser.add_argument("--verify", action="store_true", help="Verify structure after processing")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        reorganize_h5_file(args.input, args.output, args.in_place)
        if args.verify:
            verify_path = args.output if args.output else (
                args.input if args.in_place else str(input_path.with_stem(input_path.stem + '_reorganized'))
            )
            verify_structure(verify_path)
    elif input_path.is_dir():
        process_folder(args.input, args.output, args.in_place)
    else:
        print(f"Error: {args.input} does not exist")

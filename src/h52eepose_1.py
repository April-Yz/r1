import h5py
import kinpy as kp
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET

# --- 1. URDF 路径 ---
# 请确保这里指向的是包含物理描述的文件
# URDF_PATH = "r1_v2.urdf" 
URDF_PATH = "/home/pine/yzj/R1_urdf/galaxea_sim/assets/r1/robot.urdf" 


# --- 2. 运动学链配置 ---
CHAIN_CONFIG = {
    'left': {
        'root_link': 'left_arm_base_link',   
        'end_link': 'left_gripper_link',     
        'joint_names': [
            'left_arm_joint1', 'left_arm_joint2', 'left_arm_joint3', 
            'left_arm_joint4', 'left_arm_joint5', 'left_arm_joint6'
        ]
    },
    'right': {
        'root_link': 'right_arm_base_link',  
        'end_link': 'right_gripper_link',    
        'joint_names': [
            'right_arm_joint1', 'right_arm_joint2', 'right_arm_joint3', 
            'right_arm_joint4', 'right_arm_joint5', 'right_arm_joint6'
        ]
    }
}

def sanitize_urdf(urdf_content):
    """
    清洗 URDF 数据：移除 SRDF/MoveIt 特有的标签（如 group, disable_collisions），
    防止 kinpy 解析报错。
    """
    # 确保内容是字符串以便解析
    if isinstance(urdf_content, bytes):
        urdf_str = urdf_content.decode('utf-8')
    else:
        urdf_str = urdf_content

    try:
        # 解析 XML
        root = ET.fromstring(urdf_str)
        
        # 定义允许的标准 URDF 标签
        allowed_tags = ['link', 'joint', 'material', 'transmission', 'gazebo', 'robot']
        
        # 找出所有顶层子节点，移除不在白名单里的标签
        removed_count = 0
        for child in list(root):
            if child.tag not in allowed_tags:
                root.remove(child)
                removed_count += 1
        
        if removed_count > 0:
            print(f"    [Auto-Fix] Removed {removed_count} non-standard tags (SRDF/MoveIt) from URDF.")
            
        # 转回 bytes 供 kinpy 读取
        return ET.tostring(root)
        
    except Exception as e:
        print(f"    [Warning] XML sanitization failed: {e}")
        print(f"    Attempting to use original content...")
        return urdf_content

def compute_fk_for_h5(h5_path):
    print(f"\n>>> Processing FK for: {h5_path}")
    
    # 1. 加载并清洗 URDF
    try:
        with open(URDF_PATH, 'rb') as f:
            raw_data = f.read()
        
        # --- 关键修复：清洗数据 ---
        urdf_data = sanitize_urdf(raw_data)
        
    except FileNotFoundError:
        print(f"Error: URDF file not found at {URDF_PATH}.")
        return

    # 2. 构建 Kinpy Chain
    chains = {}
    try:
        for side, cfg in CHAIN_CONFIG.items():
            # build_serial_chain_from_urdf 支持传入 bytes 数据
            chains[side] = kp.build_serial_chain_from_urdf(
                urdf_data, 
                cfg['end_link'], 
                cfg['root_link']
            )
    except Exception as e:
        print(f"Error building kinematic chain: {e}")
        print("Tip: Check if your joint names in CHAIN_CONFIG match the URDF exactly.")
        return

    # 3. 读取 HDF5 并计算
    with h5py.File(h5_path, 'r+') as f: 
        obs_grp = f['obs']
        
        for side in ['left', 'right']:
            joint_key = f'arm_{side}/joint_pos'
            
            if joint_key not in obs_grp:
                continue
                
            joint_positions = obs_grp[joint_key][:] 
            num_frames, num_joints_in_h5 = joint_positions.shape
            
            # 校验关节数量
            expected_joints = len(CHAIN_CONFIG[side]['joint_names'])
            if num_joints_in_h5 != expected_joints:
                print(f"    Warning: {side} arm joint count mismatch! (HDF5: {num_joints_in_h5}, URDF: {expected_joints})")
                continue

            eef_pos = np.zeros((num_frames, 3))
            eef_quat = np.zeros((num_frames, 4))
            
            chain = chains[side]
            joint_names = CHAIN_CONFIG[side]['joint_names']

            # 计算 FK
            for i in tqdm(range(num_frames), desc=f"    Calc {side} FK", leave=False):
                th = dict(zip(joint_names, joint_positions[i]))
                transform = chain.forward_kinematics(th)
                eef_pos[i] = transform.pos
                eef_quat[i] = transform.rot 

            # 保存
            pos_dest = f'arm_{side}/eef_pos'
            if pos_dest in obs_grp: del obs_grp[pos_dest]
            obs_grp.create_dataset(pos_dest, data=eef_pos)
            
            quat_dest = f'arm_{side}/eef_quat'
            if quat_dest in obs_grp: del obs_grp[quat_dest]
            obs_grp.create_dataset(quat_dest, data=eef_quat)
            
            # 标记参考坐标系
            obs_grp[pos_dest].attrs['reference_frame'] = CHAIN_CONFIG[side]['root_link']
            
    print(f"    Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to HDF5 file")
    args = parser.parse_args()
    
    p = Path(args.path)
    if p.is_dir():
        for f in p.glob("*.h5"):
            compute_fk_for_h5(f)
    else:
        compute_fk_for_h5(p)
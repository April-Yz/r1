import h5py
import kinpy as kp
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

# --- 配置区域 (需要根据你的机器人修改) ---

# 1. 你的 URDF 文件路径
URDF_PATH = "robot.urdf" 

# 2. 定义运动学链的起点和终点
# 对于双臂机器人，通常有两个链：Base -> Left Hand, Base -> Right Hand
CHAIN_CONFIG = {
    'left': {
        'root_link': 'base_link',   # 根节点名称 (查看 URDF)
        'end_link': 'left_hand_link', # 左手末端坐标系名称
        # HDF5 中关节数据的顺序必须与这里列出的关节名称一一对应！
        'joint_names': [
            'left_arm_joint1', 'left_arm_joint2', 'left_arm_joint3', 
            'left_arm_joint4', 'left_arm_joint5', 'left_arm_joint6', 'left_arm_joint7'
        ]
    },
    'right': {
        'root_link': 'base_link',
        'end_link': 'right_hand_link', # 右手末端坐标系名称
        'joint_names': [
            'right_arm_joint1', 'right_arm_joint2', 'right_arm_joint3',
            'right_arm_joint4', 'right_arm_joint5', 'right_arm_joint6', 'right_arm_joint7'
        ]
    }
}

def compute_fk_for_h5(h5_path):
    print(f"Processing FK for: {h5_path}")
    
    # 加载 URDF 链
    try:
        with open(URDF_PATH, 'rb') as f:
            urdf_data = f.read()
    except FileNotFoundError:
        print(f"Error: URDF file not found at {URDF_PATH}")
        return

    chains = {}
    for side, cfg in CHAIN_CONFIG.items():
        chains[side] = kp.build_serial_chain_from_urdf(
            urdf_data, 
            cfg['end_link'], 
            cfg['root_link']
        )

    with h5py.File(h5_path, 'r+') as f: # 'r+' 模式允许读写
        # 遍历左臂和右臂
        for side in ['left', 'right']:
            joint_key = f'obs/arm_{side}/joint_pos'
            
            if joint_key not in f:
                continue
                
            joint_positions = f[joint_key][:] # Shape: (T, N_joints)
            num_frames = joint_positions.shape[0]
            num_joints = joint_positions.shape[1]
            
            # 检查关节数量是否匹配
            expected_joints = len(CHAIN_CONFIG[side]['joint_names'])
            if num_joints != expected_joints:
                print(f"Warning: {side} arm joint count mismatch! HDF5: {num_joints}, Config: {expected_joints}")
                continue

            # 准备输出数组
            eef_pos = np.zeros((num_frames, 3)) # x, y, z
            eef_quat = np.zeros((num_frames, 4)) # w, x, y, z (kinpy 输出 w 在前或后需确认，kinpy通常是 w,x,y,z)

            chain = chains[side]
            joint_names = CHAIN_CONFIG[side]['joint_names']

            # 批量计算 FK (Kinpy 支持简单的批量，但为了稳健这里用循环)
            print(f"  Computing {side} arm FK...")
            for i in tqdm(range(num_frames), leave=False):
                # 构造 {joint_name: angle} 字典
                th = dict(zip(joint_names, joint_positions[i]))
                
                # 正运动学计算
                transform = chain.forward_kinematics(th)
                
                eef_pos[i] = transform.pos
                eef_quat[i] = transform.rot # kinpy 返回的四元数通常是 [w, x, y, z]

            # 保存回 HDF5
            # 注意：常用的四元数顺序是 [x, y, z, w] (scipy/ros) 或 [w, x, y, z] (kinpy/mujoco)
            # 这里如果不做转换，直接存 kinpy 的原始输出
            
            obs_grp = f['obs']
            # 如果已存在则覆盖，否则创建
            # 位置
            pos_key = f'arm_{side}/eef_pos'
            if pos_key in obs_grp: del obs_grp[pos_key]
            obs_grp.create_dataset(pos_key, data=eef_pos)
            
            # 旋转
            rot_key = f'arm_{side}/eef_quat'
            if rot_key in obs_grp: del obs_grp[rot_key]
            obs_grp.create_dataset(rot_key, data=eef_quat)
            
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to HDF5 file or directory")
    args = parser.parse_args()
    
    p = Path(args.path)
    if p.is_dir():
        for f in p.glob("*.h5"):
            compute_fk_for_h5(f)
    else:
        compute_fk_for_h5(p)
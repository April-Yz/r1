import h5py
import kinpy as kp
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
# # 处理单个文件
# python3 h52eepose.py /home/pine/yzj/h5out/r1_test.h5

# # 或者处理整个文件夹
# python3 h52eepose.py /home/pine/yzj/h5out

# --- 1. URDF 路径 ---
# 请确保该文件在当前目录下，或者修改为绝对路径
URDF_PATH = "/home/pine/yzj/R1_urdf/galaxea_sim/assets/r1/robot.urdf" 

# --- 2. 运动学链配置 (根据你的 URDF 填充) ---
# 注意：由于 HDF5 里只有手臂数据没有躯干数据，
# 我们将 root_link 设为手臂基座 (arm_base_link)，计算相对于肩膀的坐标。
CHAIN_CONFIG = {
    'left': {
        'root_link': 'base_link',   # 统一为base_link
        'end_link': 'left_gripper_link',
        # torso+arm
        'joint_names': [
            'torso_joint1', 'torso_joint2', 'torso_joint3', 'torso_joint4',
            'left_arm_joint1', 'left_arm_joint2', 'left_arm_joint3', 'left_arm_joint4', 'left_arm_joint5', 'left_arm_joint6',
        ]
    },
    'right': {
        'root_link': 'base_link',
        'end_link': 'right_gripper_link',
        'joint_names': [
            'torso_joint1', 'torso_joint2', 'torso_joint3', 'torso_joint4',
            'right_arm_joint1', 'right_arm_joint2', 'right_arm_joint3', 'right_arm_joint4', 'right_arm_joint5', 'right_arm_joint6',
        ]
    }
}

def compute_fk_for_h5(h5_path):
    print(f"\n>>> Processing FK for: {h5_path}")
    
    # 1. 加载 URDF
    try:
        with open(URDF_PATH, 'rb') as f:
            urdf_data = f.read()
    except FileNotFoundError:
        print(f"Error: URDF file not found at {URDF_PATH}. Please save your XML as 'r1_v2.urdf'.")
        return

    # 2. 构建 Kinpy Chain
    chains = {}
    try:
        for side, cfg in CHAIN_CONFIG.items():
            chains[side] = kp.build_serial_chain_from_urdf(
                urdf_data, 
                cfg['end_link'], 
                cfg['root_link']
            )
            print(f"    [{side}] Chain built: {cfg['root_link']} -> {cfg['end_link']} ({len(cfg['joint_names'])} joints)")
    except Exception as e:
        print(f"Error building kinematic chain: {e}")
        return

    # 3. 读取 HDF5 并计算
    with h5py.File(h5_path, 'r+') as f: # 'r+' 模式开启读写
        obs_grp = f['obs']
        # 新增：action group
        action_grp = f['action'] if 'action' in f else None

        for side in ['left', 'right']:
            # ------- 1. 普通 arm_{side}/joint_pos -------
            joint_key = f'arm_{side}/joint_pos'
            if joint_key in obs_grp:
                joint_positions = obs_grp[joint_key][:]
                # 删除最后一个自由度（夹爪），只保留前6个
                if joint_positions.shape[1] > 6:
                    joint_positions = joint_positions[:, :6]
                num_frames = joint_positions.shape[0]
                # 拼接torso四个关节的固定值
                torso_fixed = np.array([0.25, -0.4, -0.85, 0], dtype=np.float32)
                joint_positions_full = np.concatenate([
                    np.tile(torso_fixed, (num_frames, 1)),
                    joint_positions
                ], axis=1)
                num_joints_in_h5 = joint_positions_full.shape[1]
                expected_joints = len(CHAIN_CONFIG[side]['joint_names'])
                if num_joints_in_h5 != expected_joints:
                    print(f"    Warning: {side} arm joint count mismatch after adding torso!")
                    print(f"    - URDF expects: {expected_joints} joints")
                    print(f"    - Data contains: {num_joints_in_h5} joints")
                    print("    Skipping this arm to prevent calculation errors.")
                else:
                    # 计算并保存
                    eef_pos = np.zeros((num_frames, 3))
                    eef_quat = np.zeros((num_frames, 4))
                    eef_euler = np.zeros((num_frames, 3))
                    chain = chains[side]
                    joint_names = CHAIN_CONFIG[side]['joint_names']
                    for i in tqdm(range(num_frames), desc=f"    Calc {side} FK", leave=False):
                        th = dict(zip(joint_names, joint_positions_full[i]))
                        transform = chain.forward_kinematics(th)
                        eef_pos[i] = transform.pos
                        eef_quat[i] = transform.rot
                        if hasattr(transform, 'as_euler_angles'):
                            eef_euler[i] = transform.as_euler_angles()
                        else:
                            from scipy.spatial.transform import Rotation as R
                            quat = transform.rot
                            r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
                            eef_euler[i] = r.as_euler('zyx')
                    pos_dest = f'arm_{side}/eef_pos'
                    if pos_dest in obs_grp: del obs_grp[pos_dest]
                    obs_grp.create_dataset(pos_dest, data=eef_pos)
                    quat_dest = f'arm_{side}/eef_quat'
                    if quat_dest in obs_grp: del obs_grp[quat_dest]
                    obs_grp.create_dataset(quat_dest, data=eef_quat)
                    euler_dest = f'arm_{side}/eef_euler'
                    if euler_dest in obs_grp: del obs_grp[euler_dest]
                    obs_grp.create_dataset(euler_dest, data=eef_euler)
                    obs_grp[pos_dest].attrs['reference_frame'] = CHAIN_CONFIG[side]['root_link']
            else:
                print(f"    Skipping {side}: key {joint_key} not found.")

            # ------- 2. action/arm_{side}/joint_pos -------
            if action_grp is not None:
                action_arm_key = f'arm_{side}/joint_pos'
                if action_arm_key in action_grp:
                    joint_positions = action_grp[action_arm_key][:]
                    # 直接用6自由度，不做裁剪
                    num_frames = joint_positions.shape[0]
                    torso_fixed = np.array([0.25, -0.4, -0.85, 0], dtype=np.float32)
                    joint_positions_full = np.concatenate([
                        np.tile(torso_fixed, (num_frames, 1)),
                        joint_positions
                    ], axis=1)
                    num_joints_in_h5 = joint_positions_full.shape[1]
                    expected_joints = len(CHAIN_CONFIG[side]['joint_names'])
                    if num_joints_in_h5 != expected_joints:
                        print(f"    Warning: {side} action arm joint count mismatch after adding torso!")
                        print(f"    - URDF expects: {expected_joints} joints")
                        print(f"    - Data contains: {num_joints_in_h5} joints")
                        print("    Skipping this action arm to prevent calculation errors.")
                    else:
                        # 计算并保存
                        eef_pos = np.zeros((num_frames, 3))
                        eef_quat = np.zeros((num_frames, 4))
                        eef_euler = np.zeros((num_frames, 3))
                        chain = chains[side]
                        joint_names = CHAIN_CONFIG[side]['joint_names']
                        for i in tqdm(range(num_frames), desc=f"    Calc {side} FK (action)", leave=False):
                            th = dict(zip(joint_names, joint_positions_full[i]))
                            transform = chain.forward_kinematics(th)
                            eef_pos[i] = transform.pos
                            eef_quat[i] = transform.rot
                            if hasattr(transform, 'as_euler_angles'):
                                eef_euler[i] = transform.as_euler_angles()
                            else:
                                from scipy.spatial.transform import Rotation as R
                                quat = transform.rot
                                r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
                                eef_euler[i] = r.as_euler('zyx')
                        pos_dest = f'arm_{side}/eef_pos'
                        if pos_dest in action_grp: del action_grp[pos_dest]
                        action_grp.create_dataset(pos_dest, data=eef_pos)
                        quat_dest = f'arm_{side}/eef_quat'
                        if quat_dest in action_grp: del action_grp[quat_dest]
                        action_grp.create_dataset(quat_dest, data=eef_quat)
                        euler_dest = f'arm_{side}/eef_euler'
                        if euler_dest in action_grp: del action_grp[euler_dest]
                        action_grp.create_dataset(euler_dest, data=eef_euler)
                        action_grp[pos_dest].attrs['reference_frame'] = CHAIN_CONFIG[side]['root_link']
                else:
                    print(f"    Skipping {side}: key action/arm_{side}/joint_pos not found in action group.")
            else:
                print(f"    Skipping {side}: action group not found.")
        print(f"    Done. Results saved to {h5_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FK from HDF5 joint angles")
    parser.add_argument("path", help="Path to HDF5 file or directory containing .h5 files")
    args = parser.parse_args()
    
    p = Path(args.path)
    if p.is_dir():
        h5_files = sorted(list(p.glob("*.h5")))
        print(f"Found {len(h5_files)} HDF5 files in directory.")
        for f in h5_files:
            compute_fk_for_h5(f)
    else:
        compute_fk_for_h5(p)
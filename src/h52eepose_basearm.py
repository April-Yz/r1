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
        'root_link': 'left_arm_base_link',   # 起点：左臂基座
        'end_link': 'left_gripper_link',     # 终点：左手末端
        # 必须与 HDF5 中存储的数据顺序严格一致
        'joint_names': [
            'left_arm_joint1',
            'left_arm_joint2',
            'left_arm_joint3',
            'left_arm_joint4',
            'left_arm_joint5',
            'left_arm_joint6',
        ]
    },
    'right': {
        'root_link': 'right_arm_base_link',  # 起点：右臂基座
        'end_link': 'right_gripper_link',    # 终点：右手末端
        'joint_names': [
            'right_arm_joint1',
            'right_arm_joint2',
            'right_arm_joint3',
            'right_arm_joint4',
            'right_arm_joint5',
            'right_arm_joint6',
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
        
        for side in ['left', 'right']:
            joint_key = f'arm_{side}/joint_pos'
            
            if joint_key not in obs_grp:
                print(f"    Skipping {side}: key {joint_key} not found.")
                continue
                
            # 读取关节角 (N, DOF)
            joint_positions = obs_grp[joint_key][:]
            # 删除最后一个自由度（夹爪），只保留前6个
            if joint_positions.shape[1] > 6:
                joint_positions = joint_positions[:, :6]
            num_frames, num_joints_in_h5 = joint_positions.shape
            # 校验关节数量
            expected_joints = len(CHAIN_CONFIG[side]['joint_names'])
            if num_joints_in_h5 != expected_joints:
                print(f"    Warning: {side} arm joint count mismatch after removing gripper!")
                print(f"    - URDF expects: {expected_joints} joints")
                print(f"    - HDF5 contains: {num_joints_in_h5} joints")
                print("    Skipping this arm to prevent calculation errors.")
                continue

            # 准备结果容器
            eef_pos = np.zeros((num_frames, 3)) # x, y, z
            eef_quat = np.zeros((num_frames, 4)) # w, x, y, z
            eef_euler = np.zeros((num_frames, 3)) # ZYX欧拉角

            chain = chains[side]
            joint_names = CHAIN_CONFIG[side]['joint_names']

            for i in tqdm(range(num_frames), desc=f"    Calc {side} FK", leave=False):
                th = dict(zip(joint_names, joint_positions[i]))
                transform = chain.forward_kinematics(th)
                eef_pos[i] = transform.pos
                eef_quat[i] = transform.rot # [w, x, y, z]
                # kinpy的Transform对象有as_euler_angles方法（ZYX顺序）
                if hasattr(transform, 'as_euler_angles'):
                    eef_euler[i] = transform.as_euler_angles()
                else:
                    # 兼容性处理：用四元数转欧拉
                    from scipy.spatial.transform import Rotation as R
                    quat = transform.rot # [w, x, y, z]
                    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
                    eef_euler[i] = r.as_euler('zyx')

            # 4. 保存结果回 HDF5
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
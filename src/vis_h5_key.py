# import h5py
# f = h5py.File('/home/pine/yzj/h5out/r1_test.h5')
# print(f['obs/arm_left/joint_pos'].shape)
# print(f['obs/arm_right/joint_pos'].shape)
# f.close()

import h5py

def print_h5_keys_shapes(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}: {obj.shape}")
    elif isinstance(obj, h5py.Group):
        print(f"{name}/ (Group)")

# with h5py.File('/home/pine/yzj/h5out/r1_test.h5', 'r') as f:
with h5py.File('/home/pine/yzj/h5out_small/r1_data_20260127_164419.h5', 'r') as f:

    f.visititems(print_h5_keys_shapes)

    # 输出 action/arm_left/joint_pos 和 obs/arm_left/joint_pos 的前5帧
    action_left = f['action/arm_left/joint_pos'][:5]
    obs_left = f['obs/arm_left/joint_pos'][:5]
    print("\naction/arm_left/joint_pos 前5帧:")
    print(action_left)
    print("\nobs/arm_left/joint_pos 前5帧:")
    print(obs_left)
    print("\n两者差值(前5帧):")
    print(action_left - obs_left[:,:6])  # 注意 obs 有7个关节，这里只比较前6个

    # 输出帧间变化（后一帧减前一帧）
    print("\naction/arm_left/joint_pos 帧间变化:")
    print(action_left[1:] - action_left[:-1])
    print("\nobs/arm_left/joint_pos 帧间变化:")
    print(obs_left[1:] - obs_left[:-1])

    # 处理 eef_pos 和 eef_quat
    obs_eef_pos = f['obs/arm_left/eef_pos'][:5]
    obs_eef_quat = f['obs/arm_left/eef_quat'][:5]
    action_eef_pos = f['action/arm_left/eef_pos'][:5]
    action_eef_quat = f['action/arm_left/eef_quat'][:5]

    print("\nobs/arm_left/eef_pos 前5帧:")
    print(obs_eef_pos)
    print("action/arm_left/eef_pos 前5帧:")
    print(action_eef_pos)
    print("两者差值(前5帧):")
    print(action_eef_pos - obs_eef_pos)
    print("obs/arm_left/eef_pos 帧间变化:")
    print(obs_eef_pos[1:] - obs_eef_pos[:-1])
    print("action/arm_left/eef_pos 帧间变化:")
    print(action_eef_pos[1:] - action_eef_pos[:-1])

    print("\nobs/arm_left/eef_quat 前5帧:")
    print(obs_eef_quat)
    print("action/arm_left/eef_quat 前5帧:")
    print(action_eef_quat)
    print("两者差值(前5帧):")
    print(action_eef_quat - obs_eef_quat)
    print("obs/arm_left/eef_quat 帧间变化:")
    print(obs_eef_quat[1:] - obs_eef_quat[:-1])
    print("action/arm_left/eef_quat 帧间变化:")
    print(action_eef_quat[1:] - action_eef_quat[:-1])
    
    # 新增：间隔15帧采样（第0, 15, 30, 45, 60帧）
    idxs = [0, 15, 30, 45, 60]
    action_left_15 = f['action/arm_left/joint_pos'][idxs]
    obs_left_15 = f['obs/arm_left/joint_pos'][idxs]
    print("\n==== 间隔15帧采样 (第0,15,30,45,60帧) ====")
    print("action/arm_left/joint_pos:")
    print(action_left_15)
    print("obs/arm_left/joint_pos:")
    print(obs_left_15)
    print("两者差值:")
    print(action_left_15 - obs_left_15[:,:6])
    print("帧间变化 action:")
    print(action_left_15[1:] - action_left_15[:-1])
    print("帧间变化 obs:")
    print(obs_left_15[1:] - obs_left_15[:-1])

    obs_eef_pos_15 = f['obs/arm_left/eef_pos'][idxs]
    obs_eef_quat_15 = f['obs/arm_left/eef_quat'][idxs]
    action_eef_pos_15 = f['action/arm_left/eef_pos'][idxs]
    action_eef_quat_15 = f['action/arm_left/eef_quat'][idxs]
    print("\nobs/arm_left/eef_pos (间隔15帧):")
    print(obs_eef_pos_15)
    print("action/arm_left/eef_pos (间隔15帧):")
    print(action_eef_pos_15)
    print("两者差值:")
    print(action_eef_pos_15 - obs_eef_pos_15)
    print("帧间变化 obs_eef_pos:")
    print(obs_eef_pos_15[1:] - obs_eef_pos_15[:-1])
    print("帧间变化 action_eef_pos:")
    print(action_eef_pos_15[1:] - action_eef_pos_15[:-1])

    print("\nobs/arm_left/eef_quat (间隔15帧):")
    print(obs_eef_quat_15)
    print("action/arm_left/eef_quat (间隔15帧):")
    print(action_eef_quat_15)
    print("两者差值:")
    print(action_eef_quat_15 - obs_eef_quat_15)
    print("帧间变化 obs_eef_quat:")
    print(obs_eef_quat_15[1:] - obs_eef_quat_15[:-1])
    print("帧间变化 action_eef_quat:")
    print(action_eef_quat_15[1:] - action_eef_quat_15[:-1])
    # 如需继续输出全部键shape，可保留如下行
    # f.visititems(print_h5_keys_shapes)

"""
(d515_data_coll) ➜  yzj python debug.py                                     
action/ (Group)
action/arm_left/ (Group)
action/arm_left/joint_pos: (672, 6)
action/arm_right/ (Group)
action/arm_right/joint_pos: (672, 6)
action/gripper_left/ (Group)
action/gripper_left/commanded_pos: (17521,)
action/gripper_left/commanded_pos_timestamps: (17521,)
action/gripper_left/joint_pos: (672, 1)
action/gripper_right/ (Group)
action/gripper_right/commanded_pos: (17551,)
action/gripper_right/commanded_pos_timestamps: (17551,)
action/gripper_right/joint_pos: (672, 1)
obs/ (Group)
obs/arm_left/ (Group)
obs/arm_left/eef_euler: (672, 3)
obs/arm_left/eef_pos: (672, 3)
obs/arm_left/eef_quat: (672, 4)
obs/arm_left/joint_pos: (672, 7)
obs/arm_right/ (Group)
obs/arm_right/eef_euler: (672, 3)
obs/arm_right/eef_pos: (672, 3)
obs/arm_right/eef_quat: (672, 4)
obs/arm_right/joint_pos: (672, 7)
obs/camera_head/ (Group)
obs/camera_head/depth: (672, 720, 1280)
obs/camera_head/rgb: (672, 720, 1280, 3)
obs/camera_left/ (Group)
obs/camera_left/depth: (672, 720, 1280)
obs/camera_left/rgb: (672, 480, 640, 3)
obs/camera_right/ (Group)
obs/camera_right/depth: (672, 720, 1280)
obs/camera_right/rgb: (672, 480, 640, 3)
obs/gripper_left/ (Group)
obs/gripper_left/joint_pos: (672, 1)
obs/gripper_right/ (Group)
obs/gripper_right/joint_pos: (672, 1)
timestamps: (672,)
"""
import os
import h5py
import numpy as np
import cv2
import argparse
import json
import shutil
from scipy.spatial.transform import Rotation
# bash script/0-3run_painting_visual_smooth.sh 7 "pour" 0 500 5 world_rot_h10_base020-050_smoothed > log/1211_pour_25_world_rot_h1.0_base0.20-0.50_0-500.json
# bash script/0-3run_painting_visual_smooth.sh 4 "pour" 500 1000 5 world_rot_h10_base020-050_smoothed > log/1211_pour_26_world_rot_h1.0_base0.20-0.50_500-1000.json
# CUDA_VISIBLE_DEVICES=1 python scripts/process_data_egodex_wrist.py -s 0 -e 1000 --use_ee_pose
# rsync -avP pour-demo_egodex-1001 zjyang@223.0.15.42:/data4/zjyang/program/RoboTwin/policy/pi0/training_data/demo_egodex/
# rsync -avP pour-demo_egodex-1001 zjyang@223.0.15.42:/data4/zjyang/program/RoboTwin/policy/pi0/training_data/demo_clean_egodex_01-04/
# rsync -avP pour-demo_egodex-1001 zjyang@223.0.15.42:/data4/zjyang/program/RoboTwin/policy/pi0/training_data/demo_clean_egodex_01-12-wrist/
# 上面这个视角有问题
# rsync -avP pour-demo_egodex-1001 zjyang@223.0.15.42:/data4/zjyang/program/RoboTwin/policy/pi0/training_data/demo_clean_egodex_01-15-wrist/
# 上面的wxyz和xyzw错误了（但其实没问题，主要endpose和ee_pose错误)
# rsync -avP pour-demo_egodex-1001 zjyang@223.0.15.42:/data4/zjyang/program/RoboTwin/policy/pi0/training_data/demo_clean_egodex_01-24-wrist-wxyz/

# rsync -avP pour-demo_egodex-1001 zjyang@223.0.15.42:/data4/zjyang/program/RoboTwin/policy/pi0/training_data/demo_clean_egodex_01-24-wrist-wxyz/
# rsync -a --info=progress2 pour-demo_egodex-1001/ zjyang@223.0.15.42:/data4/zjyang/program/RoboTwin/policy/pi0/training_data/demo_clean_egodex_01-24-wrist-wxyz/

# ================= 硬编码路径配置 (参照你的参考代码) =================
# NPZ 文件夹路径
# DATA_NAME = "debug_basehightchange_with_states"  "world_rot_h10_base020-050_smoothed"
DATA_NAME = "wxyz_world_rot_h10_base020-050" # "world_rot_h10_base020-050_smoothed_linkwrist" # "world_rot_h10_base020-050_smoothed_wrist" #"world_rot_h10_base020-050_smoothed" # before0103 "world_rot_h10_base020-050"
NPZ_DIR_BASE = f"/data1/zjyang/program/third/RoboTwin/code_painting/pour/{DATA_NAME}"
# MP4 文件夹路径
MP4_DIR_BASE = f"/data1/zjyang/program/third/Inpaint-Anything/results/pour/{DATA_NAME}"
# ===============================================================

def load_video(video_path, resize_dims=None):
    """
    读取 MP4 视频并转换为 list of numpy arrays.
    注意：为了后续 images_encoding (cv2.imencode) 能正常工作，
    这里保留 (H, W, C) 格式，不转换为 (C, H, W)。
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if not cap.isOpened():
        print(f"Warning: Cannot open video: {video_path}")
        return []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGR -> RGB (保留参考代码的逻辑，但注意 cv2.imencode 默认期望 BGR，
        # 如果训练代码期望 RGB 编码的 JPEG，这里转 RGB 是对的；
        # 如果训练代码用 cv2.imdecode 读取，通常会读回 BGR)
        # 这里严格遵循你的参考代码逻辑：转 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize: (1920, 1080) -> (320, 180) (参照参考代码)
        if resize_dims is not None:
            frame = cv2.resize(frame, resize_dims, interpolation=cv2.INTER_AREA)
            
        frames.append(frame)
    
    cap.release()
    return frames

def quaternion_to_euler(quat, order='xyz'):
    """
    将四元数转换为欧拉角（roll, pitch, yaw）
    
    Args:
        quat: 四元数，默认格式为 [w, x, y, z]
        order: 欧拉角的旋转顺序，默认 'xyz' (roll, pitch, yaw)
    
    Returns:
        欧拉角 (3维): [roll, pitch, yaw] 或根据order指定的顺序
    """
    if len(quat) != 4:
        raise ValueError(f"Quaternion must have 4 elements, got {len(quat)}")
    
    # 输入格式是 [w, x, y, z]，需要转换为 [x, y, z, w]（scipy的Rotation.from_quat期望的格式）
    quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
    
    try:
        r = Rotation.from_quat(quat_scipy)
        euler = r.as_euler(order, degrees=False)
        return euler
    except Exception as e:
        # 如果转换失败，输出红色错误信息
        error_msg = f"\033[91mERROR: Quaternion conversion failed!\033[0m"
        error_msg += f"\n  Input quaternion: {quat}"
        error_msg += f"\n  Expected format: [w, x, y, z]"
        error_msg += f"\n  Error: {str(e)}"
        print(error_msg)
        raise ValueError(f"Failed to convert quaternion {quat} to Euler angles. Expected format: [w, x, y, z]") from e


def quaternion_to_rotvec(quat):
    """
    将四元数转换为旋转向量（axis-angle representation）
    
    Args:
        quat: 四元数，默认格式为 [w, x, y, z]
    
    Returns:
        旋转向量 (3维): [rx, ry, rz]，表示绕轴旋转的角度
    """
    if len(quat) != 4:
        raise ValueError(f"Quaternion must have 4 elements, got {len(quat)}")
    
    # 输入格式是 [w, x, y, z]，需要转换为 [x, y, z, w]（scipy的Rotation.from_quat期望的格式）
    quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
    
    try:
        r = Rotation.from_quat(quat_scipy)
        rotvec = r.as_rotvec()
        return rotvec
    except Exception as e:
        # 如果转换失败，输出红色错误信息
        error_msg = f"\033[91mERROR: Quaternion conversion failed!\033[0m"
        error_msg += f"\n  Input quaternion: {quat}"
        error_msg += f"\n  Expected format: [w, x, y, z]"
        error_msg += f"\n  Error: {str(e)}"
        print(error_msg)
        raise ValueError(f"Failed to convert quaternion {quat} to rotation vector. Expected format: [w, x, y, z]") from e


def images_encoding(imgs):
    """
    目标格式要求的图像编码函数：将图像列表压缩为 JPEG 字节流并进行 Padding
    """
    encode_data = []
    padded_data = []
    max_len = 0
    
    for i in range(len(imgs)):
        # 注意：imgs[i] 是 RGB 格式。cv2.imencode 默认把输入当 BGR 处理。
        # 如果你想保持颜色正确，这里可能需要转回 BGR 再 encode，或者就假定后续解码知道这是 RGB。
        # 这里为了保证数据内容与 load_video 读取的一致，直接编码。
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        
        if success:
            jpeg_data = encoded_image.tobytes()
            encode_data.append(jpeg_data)
            max_len = max(max_len, len(jpeg_data))
        else:
            print(f"Warning: Encoding failed for frame {i}")

    # Padding 对齐
    for i in range(len(encode_data)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
        
    return padded_data, max_len

def data_transform(task_name, start_id, end_id, fps, save_root_path, use_end_pose=True, use_euler=True):
    
    # 关键参数：Resize 尺寸 (参照参考代码)
    resize_dims = (640,480) #(320, 240) # D435 | egidex (320, 180) 

    # 定义要跳过的ID列表
    skip_ids = {9, 11, 13, 16, 21, 22, 23, 26, 29, 30, 33, 34, 36, 42, 47, 50, 51, 53, 55, 56, 58, 64, 68, 69, 72, 74, 78, 84, 87, 93, 95, 96, 97, 98, 99, 100, 101, 109, 111, 113, 114, 116, 117, 120, 126, 132, 134, 139, 141, 146, 149, 151, 155, 165, 166, 167, 174, 175, 178, 179, 180, 181, 182, 183, 189, 192, 194, 197, 202, 204, 205, 208, 210, 215, 216, 219, 222, 225, 230, 232, 233, 236, 238, 241, 242, 243, 244, 247, 254, 256, 258, 260, 261, 262, 265, 266, 269, 271, 272, 283, 285, 287, 288, 291, 294, 295, 300, 301, 310, 313, 317, 330, 332, 333, 336, 338, 339, 340, 341, 347, 348, 353, 354, 358, 363, 366, 368, 369, 370, 374, 375, 378, 384, 385, 387, 388, 390, 401, 403, 404, 408, 409, 414, 416, 417, 419, 420, 423, 427, 429, 437, 442, 443, 444, 445, 446, 453, 460, 462, 463, 467, 468, 474, 483, 484, 495, 496, 501, 504, 505, 507, 513, 514, 515, 516, 521, 523, 524, 526, 535, 538, 541, 553, 554, 566, 569, 574, 578, 583, 585, 586, 587, 593, 594, 597, 600, 606, 614, 621, 622, 623, 624, 625, 626, 627, 628, 635, 636, 639, 641, 644, 645, 646, 647, 652, 653, 654, 655, 657, 664, 665, 666, 669, 671, 682, 683, 685, 686, 696, 699, 701, 704, 711, 713, 719, 721, 723, 725, 728, 729, 732, 733, 734, 739, 740, 741, 745, 747, 752, 753, 761, 763, 765, 767, 778, 779, 781, 782, 785, 789, 791, 794, 795, 796, 799, 802, 806, 810, 812, 813, 814, 815, 819, 820, 821, 823, 825, 832, 834, 840, 842, 844, 847, 849, 850, 851, 855, 859, 861, 865, 867, 868, 869, 870, 872, 877, 878, 880, 881, 883, 887, 889, 890, 892, 893, 902, 911, 913, 914, 915, 921, 922, 923, 924, 927, 938, 939, 940, 946, 951, 954, 955, 958, 960, 961, 962, 963, 965, 966, 967, 970, 972, 973, 976, 983, 984, 986, 991, 998}

    valid_count = 0

    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)

    for current_id in range(start_id, end_id + 1):
        # 跳过指定的ID
        if current_id in skip_ids:
            print(f"Skipping episode: {current_id} (in skip_ids)")
            continue
            
        print(f"Processing episode: {current_id} (Range {start_id}-{end_id})")

        # ================= 1. 构建路径 (参照参考代码命名规则) =================
        # 视频文件名: target_with_original_{task_name}_{id}_lg2_rg2_{fps}fps.mp4
        # target_with_original_pour_259_5fps_replay_ee_pose
        # NOT smoothed
        # video_filename = f"target_with_original_{task_name}_{current_id}_lg2_rg2_{fps}fps.mp4"
        # smoothed
        # video_filename = f"target_with_original_{task_name}_{current_id}_{fps}fps_replay_ee_pose.mp4"
        # target_with_original_pour_39_lg2_rg2_5fps.mp4 
        video_filename = f"target_with_original_{task_name}_{current_id}_lg2_rg2_{fps}fps.mp4"
        video_path = os.path.join(MP4_DIR_BASE, video_filename)
        
        # 腕部相机视频文件
        #  pour_322_lg2_rg2_5fps_rightwrist.mp4
        # video_right_wrist_filename = f"{task_name}_{current_id}_{fps}fps_replay_ee_pose_rightwrist.mp4"
        video_right_wrist_filename = f"{task_name}_{current_id}_lg2_rg2_{fps}fps_rightwrist.mp4"
        video_right_wrist_path = os.path.join(NPZ_DIR_BASE, video_right_wrist_filename)
        
        # video_left_wrist_filename = f"{task_name}_{current_id}_{fps}fps_replay_ee_pose_leftwrist.mp4"
        video_left_wrist_filename = f"{task_name}_{current_id}_lg2_rg2_{fps}fps_leftwrist.mp4"
        video_left_wrist_path = os.path.join(NPZ_DIR_BASE, video_left_wrist_filename)
        
        # NPZ文件名: {task_name}_{id}_lg2_rg2_{fps}fps_dataset.npz
        # npz_filename = f"{task_name}_{current_id}_lg2_rg2_{fps}fps_dataset.npz"
        # pour_549_lg2_rg2_5fps_dataset.npz
        # npz_filename = f"{task_name}_{current_id}_{fps}fps_replay_ee_pose_dataset.npz"
        npz_filename = f"{task_name}_{current_id}_lg2_rg2_{fps}fps_dataset.npz"
        # pour_99_5fps_replay_ee_pose_dataset.npz
        npz_path = os.path.join(NPZ_DIR_BASE, npz_filename)

        # 检查文件 (参照参考代码的容错逻辑)
        if not os.path.exists(video_path):
            print(f"  Skipping: Video file not found: {video_path}")
            continue
        
        if not os.path.exists(video_right_wrist_path):
            print(f"  Skipping: Right wrist video file not found: {video_right_wrist_path}")
            continue
        
        if not os.path.exists(video_left_wrist_path):
            print(f"  Skipping: Left wrist video file not found: {video_left_wrist_path}")
            continue
            
        if not os.path.exists(npz_path):
            # 尝试 Fallback
            fallback_npz = f"{task_name}_{current_id}_{fps}fps_dataset.npz"
            fallback_path = os.path.join(NPZ_DIR_BASE, fallback_npz)
            if os.path.exists(fallback_path):
                npz_path = fallback_path
                print(f"  Using fallback NPZ: {fallback_npz}")
            else:
                print(f"  Skipping: NPZ file not found: {npz_path}")
                continue

        try:
            # ================= 2. 读取数据 =================
            # 读取视频并 Resize
            video_frames = load_video(video_path, resize_dims)
            video_right_wrist_frames = load_video(video_right_wrist_path, resize_dims)
            video_left_wrist_frames = load_video(video_left_wrist_path, resize_dims)
            
            # 读取 NPZ
            npz_data = np.load(npz_path, allow_pickle=True)
            # 根据参数决定使用 ee_pose 还是 qpos
            # ee_pose shape: (N, 14) = [左臂end pose(7维: 位置3+四元数4), 右臂end pose(7维: 位置3+四元数4)]
            # 注意：如果使用 end pose，会将四元数转换为旋转角（欧拉角或旋转向量），最终每臂6维（位置3+旋转角3）
            # 保存到 HDF5 时会使用对应的字段名：ee_pose 或 qpos
            
            # 如果 use_end_pose 是 None，自动检测
            if use_end_pose is None:
                if 'ee_pose' in npz_data:
                    use_end_pose_episode = True
                elif 'qpos' in npz_data:
                    use_end_pose_episode = False
                else:
                    print(f"  Error: Neither 'ee_pose' nor 'qpos' found in NPZ file")
                    print(f"  Skipping episode: {current_id}")
                    continue
            else:
                use_end_pose_episode = use_end_pose
            
            # 根据决定使用哪种数据
            if use_end_pose_episode:
                if 'ee_pose' not in npz_data:
                    if use_end_pose is not None:
                        # 用户明确指定了使用 end pose，但数据中没有
                        print(f"  Error: 'ee_pose' not found in NPZ file, but --use_ee_pose was specified")
                        print(f"  Skipping episode: {current_id}")
                        continue
                    else:
                        # 自动检测时，回退到 qpos
                        print(f"  Warning: 'ee_pose' not found, falling back to 'qpos' for episode {current_id}")
                        use_end_pose_episode = False
                        qpos_raw = npz_data['qpos']
                else:
                    ee_pose_raw = npz_data['ee_pose']    # shape (N, 14)
            else:
                if 'qpos' not in npz_data:
                    print(f"  Error: 'qpos' not found in NPZ file")
                    print(f"  Skipping episode: {current_id}")
                    continue
                qpos_raw = npz_data['qpos']    # 假设 shape (N, dim)
            gripper_raw = npz_data['gripper'] # 假设 shape (N, 2)
            # 获取语言指令 (如果有的话，没有则用默认)
            language_instruction = str(npz_data['language'][0]) if 'language' in npz_data else task_name
            # print(f"  Instruction: {language_instruction}")

            # ================= 3. 数据对齐与切片 =================
            if use_end_pose_episode:
                data_len = len(ee_pose_raw)
            else:
                data_len = len(qpos_raw)
            min_len = min(len(video_frames), len(video_right_wrist_frames), len(video_left_wrist_frames), data_len)
            if min_len < 2:
                print(f"  Skipping: Not enough frames ({min_len})")
                continue

            # 截断
            video_frames = video_frames[:min_len]
            video_right_wrist_frames = video_right_wrist_frames[:min_len]
            video_left_wrist_frames = video_left_wrist_frames[:min_len]
            if use_end_pose_episode:
                ee_pose_raw = ee_pose_raw[:min_len]
            else:
                qpos_raw = qpos_raw[:min_len]
            gripper_raw = gripper_raw[:min_len]

            # 拼接 State: [Left Arm End Pose, Left Gripper, Right Arm End Pose, Right Gripper]
            # 如果使用 end pose + 四元数: [左臂end pose(7维), 左gripper(1), 右臂end pose(7维), 右gripper(1)] = 16维
            # 如果使用 end pose + 欧拉角: [左臂end pose(6维), 左gripper(1), 右臂end pose(6维), 右gripper(1)] = 14维
            # 如果使用 qpos: [左臂关节(6), 左gripper(1), 右臂关节(6), 右gripper(1)] = 14维
            states = []
            left_arm_dims = []
            right_arm_dims = []

            for k in range(min_len):
                if use_end_pose_episode:
                    # ee_pose: [左臂end pose(7), 右臂end pose(7)]
                    # 每臂7维 = 位置3 + 四元数4
                    left_arm_full = ee_pose_raw[k, :7]
                    right_arm_full = ee_pose_raw[k, 7:14]
                    
                    # 提取位置和四元数
                    left_pos = left_arm_full[:3]  # 位置 (x, y, z)
                    left_quat = left_arm_full[3:7]  # 四元数
                    right_pos = right_arm_full[:3]  # 位置 (x, y, z)
                    right_quat = right_arm_full[3:7]  # 四元数
                    
                    # 将四元数转换为旋转角
                    if use_euler:
                        # 使用欧拉角 (roll, pitch, yaw)
                        left_rot = quaternion_to_euler(left_quat, order='xyz')
                        right_rot = quaternion_to_euler(right_quat, order='xyz')
                    else:
                        # 使用旋转向量 (axis-angle)
                        left_rot = quaternion_to_rotvec(left_quat)
                        right_rot = quaternion_to_rotvec(right_quat)
                    
                    # 拼接位置和旋转角
                    left_arm_pose = np.concatenate([left_pos, left_rot])
                    right_arm_pose = np.concatenate([right_pos, right_rot])
                    left_arm_dim = 6  # 位置3 + 旋转角3
                    right_arm_dim = 6
                else:
                    # qpos: [左臂关节(6), 右臂关节(6)]
                    arm_dim = qpos_raw.shape[1] // 2
                    left_arm_pose = qpos_raw[k, :arm_dim]
                    right_arm_pose = qpos_raw[k, arm_dim:]
                    left_arm_dim = arm_dim
                    right_arm_dim = arm_dim
                
                left_gripper = gripper_raw[k, 0]
                right_gripper = gripper_raw[k, 1]
                
                # 拼接成目标需要的 1D state
                state = np.concatenate([left_arm_pose, [left_gripper], right_arm_pose, [right_gripper]]).astype(np.float32)
                states.append(state)
                
                left_arm_dims.append(left_arm_dim)
                right_arm_dims.append(right_arm_dim)
            
            states = np.array(states)

            # ================= 4. 构建 HDF5 数据结构 (目标格式逻辑) =================
            # 逻辑：
            # state (observations) 取 0 到 T-1
            # images (observations) 取 0 到 T-1
            # action 取 1 到 T (即当前帧动作导致下一帧状态)
            
            state_out = states[:-1]  # observations: 0 到 T-1
            action_out = states[1:]  # actions: 1 到 T
            
            left_arm_dim_out = np.array(left_arm_dims[:-1])
            right_arm_dim_out = np.array(right_arm_dims[:-1])
            
            # 图像切片 (取前 T-1 帧)
            cam_high_imgs = video_frames[:-1]
            cam_right_wrist_imgs = video_right_wrist_frames[:-1]
            cam_left_wrist_imgs = video_left_wrist_frames[:-1]

            # ================= 5. 保存文件 =================
            # 创建 episode 文件夹
            # 这里的命名规则参照 data_transform: "episode_{i}" (注意这里 i 是循环的索引，不是绝对ID)
            # 为了方便对应，这里我们使用 loop 的 index 作为文件夹名，或者直接用 episode_ID
            # 按照你给的 data_transform 示例，它是 range(episode_num)，这里我们也用相对索引 0, 1, 2...
            episode_save_dir = os.path.join(save_root_path, f"episode_{valid_count}")
            os.makedirs(episode_save_dir, exist_ok=True)

            # 保存 instructions.json
            save_instructions_json = {"instructions": [language_instruction]} # 包装成 list 匹配常见格式
            with open(os.path.join(episode_save_dir, "instructions.json"), "w") as f:
                json.dump(save_instructions_json, f, indent=2)

            # 保存 HDF5
            hdf5_path = os.path.join(episode_save_dir, f"episode_{valid_count}.hdf5")
            
            # 根据实际数据类型确定字段名，减少混淆
            # 注意：如果使用 ee_pose，训练配置中的 RepackTransform 需要相应更新：
            # "state": "observation.ee_pose" 而不是 "observation.state" 或 "observation.qpos"
            if use_end_pose_episode:
                state_field_name = "ee_pose"  # 存储的是 end effector pose (位置3+旋转角3)
            else:
                state_field_name = "qpos"  # 存储的是 joint angles (关节角度)
            
            with h5py.File(hdf5_path, "w") as f:
                # 1. Action
                f.create_dataset("action", data=action_out)
                
                # 2. Observations Group
                obs = f.create_group("observations")
                # 根据实际数据类型使用正确的字段名：ee_pose 或 qpos
                obs.create_dataset(state_field_name, data=state_out)
                obs.create_dataset("left_arm_dim", data=left_arm_dim_out)
                obs.create_dataset("right_arm_dim", data=right_arm_dim_out)
                
                # 3. Images Group
                image_grp = obs.create_group("images")
                
                # 编码 cam_high (主视角)
                print(f"  Encoding {len(cam_high_imgs)} frames for cam_high...")
                cam_high_enc, len_high = images_encoding(cam_high_imgs)
                image_grp.create_dataset("cam_high", data=cam_high_enc, dtype=f"S{len_high}")
                
                # 编码腕部相机视角 (cam_right_wrist, cam_left_wrist)
                print(f"  Encoding {len(cam_right_wrist_imgs)} frames for cam_right_wrist...")
                cam_right_enc, len_right = images_encoding(cam_right_wrist_imgs)
                image_grp.create_dataset("cam_right_wrist", data=cam_right_enc, dtype=f"S{len_right}")
                
                print(f"  Encoding {len(cam_left_wrist_imgs)} frames for cam_left_wrist...")
                cam_left_enc, len_left = images_encoding(cam_left_wrist_imgs)
                image_grp.create_dataset("cam_left_wrist", data=cam_left_enc, dtype=f"S{len_left}")

            print(f"  Saved episode_{valid_count} to {hdf5_path}")
            valid_count += 1

        except Exception as e:
            print(f"  Error processing ID {current_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return valid_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert generated data to Training format.")
    # 参数设置参照参考代码
    parser.add_argument("--task_name", "-n", type=str, default="pour", help="Task name")
    parser.add_argument("--start_id", "-s", type=int, default=0, help="Start Episode ID")
    parser.add_argument("--end_id", "-e", type=int, default=10, help="End Episode ID")
    parser.add_argument("--fps", "-f", type=int, default=5, help="FPS of the video")
    parser.add_argument("--use_ee_pose", action="store_true", help="Use end effector pose (ee_pose) instead of joint angles (qpos).")
    parser.add_argument("--use_qpos", action="store_true", help="Use joint angles (qpos) instead of end effector pose (ee_pose).")
    parser.add_argument("--use_euler", action="store_true", help="Convert quaternion to Euler angles (roll, pitch, yaw). This is the default behavior.")
    parser.add_argument("--use_rotvec", action="store_true", help="Convert quaternion to rotation vector (axis-angle) instead of Euler angles.")

    args = parser.parse_args()

    # 参数冲突检查
    if args.use_ee_pose and args.use_qpos:
        parser.error("Cannot specify both --use_ee_pose and --use_qpos. Please choose one.")
    
    if args.use_rotvec:
        use_euler = False
    else:
        use_euler = True  # 默认使用欧拉角

    # 确定使用哪种数据格式
    # 如果指定了 --use_ee_pose，强制使用 end pose
    # 如果指定了 --use_qpos，强制使用 qpos
    # 如果都没指定，自动检测（优先使用 end pose 如果数据中有）
    if args.use_ee_pose:
        use_end_pose = True
        print("Using end effector pose (ee_pose) as specified by --use_ee_pose")
    elif args.use_qpos:
        use_end_pose = False
        print("Using joint angles (qpos) as specified by --use_qpos")
    else:
        # 默认行为：自动检测，优先使用 end pose
        use_end_pose = None  # None 表示自动检测
        print("Note: No format specified. Will auto-detect: prefer end pose (ee_pose) if available, otherwise use qpos.")

    task_name = args.task_name
    expert_data_num = args.end_id - args.start_id + 1

    # 目标输出目录 (参照目标格式命名: task-setting-num)
    # setting 这里为了简单设为 "sim" 或者你可以从参数传入
    setting = "demo_egodex" 
    target_dir = f"processed_data/{task_name}-{setting}-{expert_data_num}"

    print(f"Reading data from:")
    print(f"  NPZ: {NPZ_DIR_BASE}")
    print(f"  MP4: {MP4_DIR_BASE}")
    print(f"Saving to: {target_dir}")

    count = data_transform(
        task_name,
        args.start_id,
        args.end_id,
        args.fps,
        target_dir,
        use_end_pose=use_end_pose,
        use_euler=use_euler
    )
    
    print(f"\nProcessing complete. Total valid episodes: {count}")
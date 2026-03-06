Data collection phase:
All windows before action:
export ROS_MASTER_URI=http://192.168.123.15:11311  
export ROS_IP=192.168.123.15

# 1. window1
roscore
	# Then activate script on r1
# 2. window2: remote control
conda activate brs1
cd brs-ctrl
python scripts/joylo/real_joylo.py

# 3. window3: wrist cams
roslaunch hands_d405 hands_d405.launch



# 4. 
roslaunch record_15hz.launch

# 5.
rosbag record --lz4 -O /home/pine/yzj/pour/r1_data_$(date +%Y%m%d_%H%M%S).bag \
  /left/camera/color/image_raw/compressed \
  /right/camera/color/image_raw/compressed \
  /left/camera/depth/image_rect_raw \
  /right/camera/depth/image_rect_raw \
  /hdas/camera_head/depth/depth_registered \
  /hdas/camera_head/rgb/image_rect_color/compressed \
  /hdas/feedback_arm_left_low \
  /hdas/feedback_arm_right_low \
  /hdas/feedback_gripper_left_low \
  /hdas/feedback_gripper_right_low \
  /motion_target/target_joint_state_arm_left_low \
  /motion_target/target_joint_state_arm_right_low \
  /motion_control/position_control_gripper_left_low \
  /motion_control/position_control_gripper_right_low











1.仿照R1文件中原来的单臂真机代码算法帮我写一个pi0的测试代码。 2. 原来的eval流程见代码，3.我训练得到的做了一些修改，得到的结果末端位置，所以还要按照h52eepose的方式通过ik计算关节角度，最后再应用（torsor和fk中的保持一致）。 4.初始位置先设置为0.





# cd yzj
# chmod +x throttle_record.sh
# ./throttle_record.sh


# 4. window4: recording rosbag
# rosbag record -O /media/pine/T7/roco_real/task3/r1_data_$(date +%Y%m%d_%H%M%S).bag \
#   /left/camera/color/image_raw/compressed \
#   /right/camera/color/image_raw/compressed \
#   /left/camera/depth/image_rect_raw \
#   /right/camera/depth/image_rect_raw \
#   /hdas/camera_head/depth/depth_registered \
#   /hdas/camera_head/rgb/image_rect_color/compressed \
#   /hdas/feedback_arm_left \
#   /hdas/feedback_arm_right \
#   /hdas/feedback_gripper_left \
#   /hdas/feedback_gripper_right \
#   /motion_target/target_joint_state_arm_left \
#   /motion_target/target_joint_state_arm_right \
#   /motion_control/position_control_gripper_left \
#   /motion_control/position_control_gripper_right
  

torso pos: [0.25, -0.4, -0.85, 0]
chassis pos: Let lower edge of the image view align with edge of the desk, and right edge of thw view with upperright corner of the desk, with tiny margin.

head cam: autoexposure off, exposure 10, gain 15

After R1 startup: wifi connection for timestamp; wired connection set to 1




# # yzj pour 
# rosbag record -O /home/pine/yzj/pour/r1_data_$(date +%Y%m%d_%H%M%S).bag \
#   /left/camera/color/image_raw/compressed \
#   /right/camera/color/image_raw/compressed \
#   /left/camera/depth/image_rect_raw \
#   /right/camera/depth/image_rect_raw \
#   /hdas/camera_head/depth/depth_registered \
#   /hdas/camera_head/rgb/image_rect_color/compressed \
#   /hdas/feedback_arm_left \
#   /hdas/feedback_arm_right \
#   /hdas/feedback_gripper_left \
#   /hdas/feedback_gripper_right \
#   /motion_target/target_joint_state_arm_left \
#   /motion_target/target_joint_state_arm_right \
#   /motion_control/position_control_gripper_left \
#   /motion_control/position_control_gripper_right
  
  
#   # 压缩
#   # yzj pour 
# rosbag record  --lz4 -O /home/pine/yzj/pour/r1_data_$(date +%Y%m%d_%H%M%S).bag \
#   /left/camera/color/image_raw/compressed \
#   /right/camera/color/image_raw/compressed \
#   /left/camera/depth/image_rect_raw \
#   /right/camera/depth/image_rect_raw \
#   /hdas/camera_head/depth/depth_registered \
#   /hdas/camera_head/rgb/image_rect_color/compressed \
#   /hdas/feedback_arm_left \
#   /hdas/feedback_arm_right \
#   /hdas/feedback_gripper_left \
#   /hdas/feedback_gripper_right \
#   /motion_target/target_joint_state_arm_left \
#   /motion_target/target_joint_state_arm_right \
#   /motion_control/position_control_gripper_left \
#   /motion_control/position_control_gripper_right
  
rosbag record --lz4 -O /home/pine/yzj/pour/r1_data_$(date +%Y%m%d_%H%M%S).bag \
  /left/camera/color/image_raw/compressed \
  /right/camera/color/image_raw/compressed \
  /left/camera/depth/image_rect_raw \
  /right/camera/depth/image_rect_raw \
  /hdas/camera_head/depth/depth_registered \
  /hdas/camera_head/rgb/image_rect_color/compressed \
  /hdas/feedback_arm_left_low \
  /hdas/feedback_arm_right_low \
  /hdas/feedback_gripper_left_low \
  /hdas/feedback_gripper_right_low \
  /motion_target/target_joint_state_arm_left_low \
  /motion_target/target_joint_state_arm_right_low \
  /motion_control/position_control_gripper_left_low \
  /motion_control/position_control_gripper_right_low


rosbag record --lz4 -O /home/pine/yzj/cup/r1_data_$(date +%Y%m%d_%H%M%S).bag \
  /left/camera/color/image_raw/compressed \
  /right/camera/color/image_raw/compressed \
  /left/camera/depth/image_rect_raw \
  /right/camera/depth/image_rect_raw \
  /hdas/camera_head/depth/depth_registered \
  /hdas/camera_head/rgb/image_rect_color/compressed \
  /hdas/feedback_arm_left_low \
  /hdas/feedback_arm_right_low \
  /hdas/feedback_gripper_left_low \
  /hdas/feedback_gripper_right_low \
  /motion_target/target_joint_state_arm_left_low \
  /motion_target/target_joint_state_arm_right_low \
  /motion_control/position_control_gripper_left_low \
  /motion_control/position_control_gripper_right_low




  cd /home/pine/yzj/src

# 1. Dummy 模式测试（不需要 ROS，用随机图像）
./run_test_pi0.sh dummy

# 2. ROS 模式（从 ROS topics 读取，仅打印 action）
./run_test_pi0.sh ros

# 3. ROS 模式 + IK 计算（输出关节角度）
./run_test_pi0.sh ros_ik

# 仅输出 action 结果
python test_pi0_ros.py \
    --train_config_name R1_FFT_pour_35_0130_5k \
    --checkpoint_path /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/10000 \
    --task_prompt "pour" \
    --dummy_mode \
    --print_only

# 使用 ROS 并计算 IK
python test_pi0_ros.py \
    --train_config_name R1_FFT_pour_35_0130_5k \
    --checkpoint_path /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/10000 \
    --task_prompt "pour" \
    --ros_mode \
    --compute_ik


cd /home/pine/yzj/src && ./run_test_pi0.sh bag /home/pine/yzj/pour/r1_data_20260202_172011.b
ag 2>&1 | head -60




# 设置 ROS 环境
export ROS_MASTER_URI=http://192.168.123.15:11311
export ROS_IP=192.168.123.15

# 使用系统 Python 运行桥接
/usr/bin/python3 /home/pine/yzj/src/ros_bridge.py

source .venv/bin/activate

conda activate RoboTwin
source ../RoboTwin/policy/pi0/.venv/bin/activate
cd /home/pine/yzj/src

# 仅输出 action
./run_test_pi0.sh zmq

# 或者 action + IK 关节角度
./run_test_pi0.sh zmq_ik


# github
ssh-keygen -t ed25519 -C "small_lap_40602979193155@qq.com" -f ~/.ssh/id_rsa_yzj
 cat ~/.ssh/id_rsa_yzj.pub

 echo "# r1" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
# git remote add origin git@github.com:April-Yz/r1.git
git remote add origin git@github-april.com:April-Yz/r1.git
git push -u origin main
# 先删除现有的 origin
git remote remove origin

# 添加新的 origin，注意地址里的 github.com 换成了 github-april
git remote add origin git@github-april:April-Yz/r1.git


cd /home/pine/yzj/src

# ========== 基本测试 ==========
# 测试模型加载
./run_test_pi0.sh dummy

# 从 bag 读取，仅打印 action
./run_test_pi0.sh bag /home/pine/yzj/pour/r1_data_20260202_172011.bag

# ========== 带 IK 计算 ==========
# 从 ZMQ 读取，计算 IK + 显示关节变化量
./run_test_pi0.sh zmq_ik_delta

# ========== 精度分析 ==========
# 从 bag 读取，比较预测与真实 action 误差
./run_test_pi0.sh bag_compare /home/pine/yzj/pour/r1_data_20260202_172011.bag
./run_test_pi0.sh bag_compare /media/pine/Yang/R1/pour/r1_data_20260127_154637.bag

# 从 bag 读取，完整分析（IK + 关节变化量 + GT 比较）
./run_test_pi0.sh bag_full /home/pine/yzj/pour/r1_data_20260202_172011.bag
./run_test_pi0.sh bag_full /media/pine/Yang/R1/pour_0203/r1_data_20260203_215507.bag


# ===== 终端 1: 启动 ROS 桥接 =====
export ROS_MASTER_URI=http://192.168.123.15:11311
export ROS_IP=192.168.123.15
/usr/bin/python3 /home/pine/yzj/src/ros_bridge.py

# ===== 终端 2: 启动控制模式 =====
cd /home/pine/yzj/src
./run_test_pi0.sh zmq_control


./run_test_pi0.sh zmq_control_auto

rclone copy /media/pine/Yang/R1/pour_0203_1 gdrive_yzj:R1/pour/0203_1 -P
rclone copy /media/pine/Yang/R1/pour_0203 gdrive_yzj:R1/pour/0203 -P
rclone copy /media/pine/Yang/R1/pour_0202 gdrive_yzj:R1/pour/0202 -P
rclone copy /media/pine/Yang/R1/pour_0201 gdrive_yzj:R1/pour/0201 -P

tmux new -s rclone_job  # 创建并进入名为 rclone_job 的会话

# 在 tmux 窗口内粘贴以下命令：
rclone copy /media/pine/Yang/R1/ gdrive_yzj:R1/pour/ \
  --include "/pour_0203_1/**" \
  --include "/pour_0203/**" \
  --include "/pour_0202/**" \
  --include "/pour_0201/**" \
  --transfers 10 \
  --drive-chunk-size 256M \
  --buffer-size 128M \
  --drive-acknowledge-abuse \
  -P


  # 实时高频
  # 不要运行 record_15hz.launch！
# 终端1：启动ros_bridge（会自动检测并使用高频topics）
/usr/bin/python3 /home/pine/yzj/src/ros_bridge.py

# 终端2：运行PI0控制
./run_test_pi0.sh zmq_control_auto


# ============低频 ==============
# 终端1：启动降频节点（创建 _low topics）
roslaunch record_15hz.launch

# 终端2：启动ros_bridge（会自动检测并使用低频topics）
/usr/bin/python3 /home/pine/yzj/src/ros_bridge.py

# 终端3：录制bag
rosbag record -o data.bag /hdas/feedback_arm_left_low /hdas/feedback_arm_right_low ...

























# 注意：这里去掉了 --run_colmap 参数
python scripts/colmap2nerf.py --video data/bottle_dataset/bottle.mp4
# 1. 去背景 (生成 mask)
python scripts/remove_bg.py data/bottle_dataset/images

# 2. 开始重建 (Stage 0)
python main.py data/bottle_dataset/ \
    --workspace trial_bottle \
    -O \
    --data_format colmap \
    --bound 1 \
    --scale 0.3 \
    --stage 0 \
    --clean_min_f 16 \
    --clean_min_d 10 \
    --visibility_mask_dilation 50 \
    --sdf

    # 1. 自动去背景 (生成 mask)
# 如果报错 No module named 'rembg'，请先 pip install rembg
python scripts/remove_bg.py data/bottle_dataset/images

# 2. 开始重建 (使用 mask)
# 注意这里 --scale 0.3 是关键，要把瓶子缩放到单位立方体内
python main.py data/bottle_dataset/ -O --data_format colmap --bound 1 --scale 0.3 --stage 0 --clean_min_f 16 --clean_min_d 10 --visibility_mask_dilation 50 --sdf



# ========================
# 1. 提取图片 (使用 ffmpeg，设为 3fps)
mkdir -p data/cup_dataset/images
ffmpeg -i data/cup_dataset/cup.mp4 -vf "fps=3" -qscale:v 1 -qmin 1 data/cup_dataset/images/%04d.jpg

# 2. 特征提取 (强制使用 CPU，避免崩溃)
# colmap feature_extractor \
#     --database_path data/cup_dataset/colmap.db \
#     --image_path data/cup_dataset/images \
#     --ImageReader.camera_model OPENCV \
#     --SiftExtraction.use_gpu 0

# 3. 特征匹配 (强制使用 CPU，避免崩溃)
colmap exhaustive_matcher \
    --database_path data/cup_dataset/colmap.db \
    --SiftMatching.use_gpu 0

# 4. 稀疏重建 (这一步时间稍长，请耐心等待)
mkdir -p data/cup_dataset/colmap_sparse
colmap mapper \
    --database_path data/cup_dataset/colmap.db \
    --image_path data/cup_dataset/images \
    --output_path data/cup_dataset/colmap_sparse

# 5. 转换格式生成 JSON
python scripts/colmap2nerf.py --video data/cup_dataset/cup.mp4

# 1. 自动去背景 (生成 mask)
python scripts/remove_bg.py data/cup_dataset/images

# 2. 运行重建 (Stage 0)
# 注意：这里 workspace 改为了 trial_cup，scale 保持 0.3 (适合杯子这种小物体)
python main.py data/cup_dataset/ \
    --workspace trial_cup \
    -O \
    --data_format colmap \
    --bound 1 \
    --scale 0.3 \
    --stage 0 \
    --clean_min_f 16 \
    --clean_min_d 10 \
    --visibility_mask_dilation 50 \
    --sdf


# 1. 创建专门的文件夹
mkdir -p data/bottle_dataset

# 2. 把你的视频移进去 (假设原本在 data/bottle.mp4)
mv data/bottle.mp4 data/bottle_dataset/

python scripts/colmap2nerf.py --video data/cup_dataset/cup.mp4  --run_colmap 

# 1. 自动去背景 (生成 mask)
# 如果报错 No module named 'rembg'，请先 pip install rembg
python scripts/remove_bg.py data/cup_dataset/images

# 2. 开始重建 (使用 mask)
# 注意这里 --scale 0.3 是关键，要把瓶子缩放到单位立方体内
python main.py data/cup_dataset/ -O --data_format colmap --bound 1 --scale 0.3 --stage 0 --clean_min_f 16 --clean_min_d 10 --visibility_mask_dilation 50 --sdf


rclone copy  gdrive_yzj:R1/pi0_checkpoints/R1_FT_pour_112_0209_lora/R1_Lora_pour/5000.tar.gz /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/R1_FT_pour_112_0209_lora -P
rclone copy  gdrive_yzj:R1/pi0_checkpoints/R1_FT_pour_112_0209_lora/R1_Lora_pour/10000.tar.gz /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/R1_FT_pour_112_0209_lora -P
rclone copy  gdrive_yzj:R1/pi0_checkpoints/R1_FT_pour_112_0209_lora/R1_Lora_pour/15000.tar.gz /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/R1_FT_pour_112_0209_lora -P
rclone copy  gdrive_yzj:R1/pi0_checkpoints/R1_FT_pour_112_0209_lora/R1_Lora_pour/1000.tar.gz /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/R1_FT_pour_112_0209_lora -P

rclone copy  gdrive_yzj:R1/pi0_checkpoints/R1_FT_pour_35_0208_lora/R1_Lora/10000.tar.gz /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/R1_FT_pour_35_0208_lora -P
rclone copy  gdrive_yzj:R1/pi0_checkpoints/R1_FT_pour_35_0208_lora/R1_Lora/5000.tar.gz /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/R1_FT_pour_35_0208_lora -P


rclone copy  gdrive_yzj:R1/pi0_checkpoints/R1_FT_stack_cup_0210_48_lora/R1_Lora/5000.tar.gz /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/R1_FT_stack_cup_0210_48_lora -P
rclone copy  gdrive_yzj:R1/pi0_checkpoints/R1_FT_stack_cup_0210_48_lora/R1_Lora/2000.tar.gz /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/R1_FT_stack_cup_0210_48_lora -P
rclone copy  gdrive_yzj:R1/pi0_checkpoints/R1_FT_stack_cup_0210_48_lora/R1_Lora/10000.tar.gz /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/R1_FT_stack_cup_0210_48_lora -P


rclone copy  /home/pine/yzj/src/results gdrive_yzj:R1/pi0_result/ -P --dry-run
rclone copy  /media/pine/Yang/R1/pour gdrive_yzj:R1/0127/ -P --dry-run
rclone copy  /home/pine/yzj/pnp_apple_star gdrive_yzj:R1/ros_data/pnp_apple_star -P --dry-run

rclone copy  /home/pine/yzj/vis_head/ gdrive_yzj:R1/pour_vis/pour_0127/ -P --dry-run


# 分析：
[fail]head_cam_20260211_153048.mp4 两个动作连续执行：只记住了后面一个


r1_data_20260203_223930 # 前面一个pour姿势

# 0213新的
rclone copy  gdrive_yzj:R1/pi0_checkpoints/R1_FT_pour_0212_48_lora/R1_Lora_pour/10000.tar.gz /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/R1_FT_pour_0212_48_lora -P


rclone copy  gdrive_yzj:R1/pi0_checkpoints/R1_FT_pnp_0212_59_lora/R1_Lora_pnp/15000.tar.gz /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/R1_FT_pnp_0212_59_lora -P
# server42 fft
rclone copy  gdrive_yzj:R1/pi0_checkpoints/R1_Full-FT_pnp_0212_55_lora/R1_Lora_pnp_server42/30000.tar.gz /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/R1_Full-FT_pnp_0212_55_lora -P
# 
rclone copy  gdrive_yzj:R1/pi0_checkpoints/R1_FT_pour_0212_48_lora_2/R1_Lora_pour_2/30000.tar.gz /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/R1_FT_pour_0212_48_lora_2 -P


rclone copy  gdrive_yzj:R1/pi0_checkpoints/R1_FT_pour_0212_48_lora_2/R1_Lora_pour_2/10000.tar.gz /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/R1_FT_pour_0212_48_lora_2 -P

rclone copy  gdrive_yzj:R1/pi0_checkpoints/R1_FT_pour_0212_48_lora_2/R1_Lora_pour_2/20000.tar.gz /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/R1_FT_pour_0212_48_lora_2 -P

rclone copy  gdrive_yzj:R1/pi0_checkpoints/R1_FT_pour_0212_48_lora_2/R1_Lora_pour_2/15000.tar.gz /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/R1_FT_pour_0212_48_lora_2 -P

# 叠杯子 0215 双卡
rclone copy  gdrive_yzj:R1/pi0_checkpoints/R1_FT_stack_cup_0215_48_lora_2/R1_Lora/10000.tar.gz /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/R1_FT_stack_cup_0215_48_lora_2 -P
rclone copy  gdrive_yzj:R1/pi0_checkpoints/R1_FT_stack_cup_0215_48_lora_2/R1_Lora/30000.tar.gz /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/R1_FT_stack_cup_0215_48_lora_2 -P


rclone copy  gdrive_yzj:R1/pi0_checkpoints/R1_FT_pnp_0212_55_lora_2/R1_Lora_pnp_2/10000.tar.gz /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/R1_FT_pnp_0212_55_lora_2 -P
rclone copy  gdrive_yzj:R1/pi0_checkpoints/R1_FT_pnp_0212_55_lora_2/R1_Lora_pnp_2/30000.tar.gz /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/R1_FT_pnp_0212_55_lora_2 -P

rclone copy  gdrive_yzj:R1/pi0_checkpoints/R1_lora_cuda1/R1_Lora_pnp_apple_star_1/20000.tar.gz /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/R1_Lora_pnp_apple_star_1 -P

rclone copy  gdrive_yzj:R1/pi0_checkpoints/pine_new/pi05_pour_50_0215/29999 /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/pi05_pour_50_0215/29999 -P
rclone copy  gdrive_yzj:R1/pi0_checkpoints/pine_new/pi05_pnp0218/29999 /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/pi05_pnp0218/29999 -P # 待测


rclone copy  gdrive_yzj:R1/pi0_checkpoints/pine_new/pi05_pour_48_0226zaijia/29999 /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/pi05_pour_48_0226zaijia/29999 -P # 待测
rclone copy  gdrive_yzj:R1/pi0_checkpoints/pine_new/pi05_pnp_apple_star_0226zaijia/29999 /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/pi05_pnp_apple_star_0226zaijia/29999 -P # 待测



rclone copy /home/pine/yzj/hand/ gdrive_yzj:R1/hand/ -P --transfers 8 --checkers 16 --dry-run
rclone copy /home/pine/yzj/src/videos gdrive_yzj:R1/pi0_result/pi05 -P --transfers 8 --checkers 16 --dry-run

# Pi0.5 推理（先激活 openpi 环境）
source ../openpi/.venv/bin/activate

# dummy 测试
./run_test_pi0.sh pi05_dummy

# ZMQ 实时推理
./run_test_pi0.sh pi05_zmq

# ZMQ 控制机器人
./run_test_pi0.sh pi05_zmq_control

# PI0
./run_test_pi0.sh zmq_control_lock_euler

# PI0.5
./run_test_pi0.sh pi05_zmq_control_lock_euler

XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_zaijia_0215 \
  --policy.dir /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/pi05_pour_50_0215/29999


# ===========================
# Terminal 1: ROS bridge
/usr/bin/python3 /home/pine/yzj/src/ros_bridge.py

# Terminal 2: Policy server（注意加 --default_prompt）
cd /home/pine/yzj/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 uv run scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config pi05_zaijia_0215 \
    --policy.dir /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/pi05_pour_50_0215/29999 \
    --default_prompt "pour"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_zaijia_0215 \
  --policy.dir /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/pi05_pour_50_0215/29999

XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_zaijia_0215 \
  --policy.dir /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/pi05_pnp_apple_star_0226zaijia/29999

XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_zaijia_0215 \
  --policy.dir /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/pi05_pour_48_0226zaijia/29999

XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_zaijia_0215 \
  --policy.dir /home/pine/yzj/RoboTwin/policy/pi0/checkpoint/pi05_pnp0218/29999
  
# Terminal 3: Controller
cd /home/pine/yzj/src
python deploy_pi0_1030.py --task_prompt "pour" --n_iterations 200
python deploy_pi0_1030.py --task_prompt "pour" --n_iterations 200  --action_index 5

python deploy_pi0_1030.py --task_prompt "pour" --n_iterations 200 --action_as_obs
python deploy_pi0_1030.py --task_prompt "pour" --n_iterations 200 --action_as_obs --action_index 5
python deploy_pi0_1030.py --task_prompt "pour" --n_iterations 200 --repeat_actions 3

python deploy_pi0_1030.py --task_prompt "pour" --n_iterations 200 --action_as_obs --action_index 5

python deploy_pi0_1030.py --task_prompt "pour" --action_as_obs --ensemble_size 4 --joint_tolerance 0.001
python deploy_pi0_1030.py --task_prompt "pick up the banana and the pear, then place them on the plate" --action_as_obs --ensemble_size 4 --joint_tolerance 0.001
python deploy_pi0_1030.py --task_prompt "pick up the banana and the pear, then place them on the plate" --n_iterations 200 
python deploy_pi0_1030.py --task_prompt "pick up the banana and the pear, then place them on the plate" --action_as_obs --ensemble_size 2 --joint_tolerance 0.01

# 必须
conda deactivate
# 2. 临时将 CUDA 12.1 设为当前终端的首选编译器
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# 1. 删掉刚才用旧版 CUDA 编译失败留下的缓存文件
rm -rf build/ 
rm -rf *.egg-info/

# 2. 重新开始编译安装（这一步可能需要几分钟，请耐心等待）
pip install -e .
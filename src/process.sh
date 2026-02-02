
rsync -ah --info=progress2 /home/pine/yzj/pour/ /media/pine/EXTERNAL_USB/yzj/R1

rsync -ah --info=progress2 /home/pine/yzj/h5out_small /media/pine/EXTERNAL_USB/yzj/R1

rsync -a --info=progress2 /home/pine/yzj/pour/ ./

# 解压到名为 target_dir 的目录
tar -zxvf 10000.tar.gz -C /home/pine/yzj/src/ckpt



# 执行压缩命令
tar -czvf "/media/pine/新加卷/R1/h5out_small.tar.gz" -C "/home/pine/yzj" h5out_small

# 有问题的bag文件
r1_data_20260127_155219.bag 
r1_data_20260127_155245.bag
r1_data_20260127_161902.bag

r1_data_20260127_163912.mp4

python3 bag2h5x.py /home/pine/yzj/pour --batch --output /home/pine/yzj/h5out

# 保存下来的h5文件
r1_data_20260127_154637.bag  r1_data_20260127_161518.bag  r1_data_20260127_163522.bag
r1_data_20260127_155219.bag  r1_data_20260127_161607.bag  r1_data_20260127_163709.bag
r1_data_20260127_155245.bag  r1_data_20260127_161701.bag  r1_data_20260127_163814.bag
r1_data_20260127_155430.bag  r1_data_20260127_161804.bag  r1_data_20260127_163912.bag
r1_data_20260127_155514.bag  r1_data_20260127_161902.bag  r1_data_20260127_164030.bag
r1_data_20260127_155558.bag  r1_data_20260127_162037.bag  r1_data_20260127_164201.bag
r1_data_20260127_155641.bag  r1_data_20260127_162754.bag  r1_data_20260127_164251.bag
r1_data_20260127_155904.bag  r1_data_20260127_162838.bag  r1_data_20260127_164336.bag
r1_data_20260127_160042.bag  r1_data_20260127_162928.bag  r1_data_20260127_164419.bag
r1_data_20260127_161056.bag  r1_data_20260127_163052.bag  r1_data_20260127_164504.bag
r1_data_20260127_161205.bag  r1_data_20260127_163142.bag  r1_data_20260127_164613.bag
r1_data_20260127_161325.bag  r1_data_20260127_163316.bag  r1_data_20260127_164746.bag
r1_data_20260127_161414.bag  r1_data_20260127_163417.bag  r1_data_20260127_164834.bag



action/ (Group)
action/action/ (Group)
action/action/arm_left/ (Group)
action/action/arm_left/eef_euler: (672, 3)
action/action/arm_left/eef_pos: (672, 3)
action/action/arm_left/eef_quat: (672, 4)
action/action/arm_right/ (Group)
action/action/arm_right/eef_euler: (672, 3)
action/action/arm_right/eef_pos: (672, 3)
action/action/arm_right/eef_quat: (672, 4)
action/arm_left/ (Group)
action/arm_left/eef_euler: (672, 3)
action/arm_left/eef_pos: (672, 3)
action/arm_left/eef_quat: (672, 4)
action/arm_left/joint_pos: (672, 6)
action/arm_right/ (Group)
action/arm_right/eef_euler: (672, 3)
action/arm_right/eef_pos: (672, 3)
action/arm_right/eef_quat: (672, 4)
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



conda install cuda -c "nvidia/label/cuda-12.1.1"



# 1. 强制指定 CUDA 安装目录为你这个环境的目录
export CUDA_HOME=/home/pine/miniconda3/envs/RoboTwin2

# 2. 把环境里的 bin 目录加到 PATH 最前面 (确保用到里面的 nvcc)
export PATH=/home/pine/miniconda3/envs/RoboTwin2/bin:$PATH

# 3. 编译库需要的链接库路径
export LD_LIBRARY_PATH=/home/pine/miniconda3/envs/RoboTwin2/lib:$LD_LIBRARY_PATH



rsync -avP user@192.168.1.100:/data/checkpoint/ ./local_checkpoint_folder/


rsync -avP lab-server42:/data4/zjyang/program/RoboTwin/policy/pi0/checkpoints/R1_FFT_pour_35_0130/demo_clean/3000 ./

rsync -avh --info=progress2 lab-server42:/data4/zjyang/program/RoboTwin/policy/pi0/checkpoints/R1_FFT_pour_35_0130/demo_clean/3000 ./
# #!/bin/bash
# """
# header: 
#   seq: 360795
#   stamp: 
#     secs: 1769932438
#     nsecs: 766441080
#   frame_id: ''
# name: 
#   - left_arm
# position: [-0.16744680851063828, 2.0108510638297874, -0.6593617021276595, 2.002127659574468, 0.39382978723404255, -1.7193617021276595, -1.6229787234042554]


# position: [0.19234042553191488, 1.8925531914893616, -0.6874468085106383, -1.6057446808510638, -0.10148936170212766, 1.3085106382978724, -2.4682978723404254]


# gripper开到关： -2.7-0

# /home/pine/yzj/src/ckpt/10000
# bash eval.sh pour demo_clean R1_FFT_pour_35_0130 demo_clean 0
# bash eval.sh pour demo_clean R1_FFT_pour_35_0130_5k demo_clean 0 0 10000

# """
# PI0 测试脚本 - 使用 IK 将末端位置转换为关节角度

# 配置参数
TRAIN_CONFIG_NAME="R1_FFT_pour_35_0130" # "your_train_config"
MODEL_NAME="pi0" # "your_model_name"
CHECKPOINT_ID="10000" #"latest"
PI0_STEP=10
TASK_PROMPT="pour" #"Pick up the object"
N_ITERATIONS=50
CHUNK_SIZE=10
SIDE="left"  # or "right"

# GPU 设置
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 # ensure GPU < 24G

# ===== 模式 1: 虚拟模式测试（不需要真实硬件） =====
# echo "Running in DUMMY mode for testing..."
# python test_pi0_with_ik.py \
#     --train_config_name ${TRAIN_CONFIG_NAME} \
#     --model_name ${MODEL_NAME} \
#     --checkpoint_id ${CHECKPOINT_ID} \
#     --pi0_step ${PI0_STEP} \
#     --task_prompt "${TASK_PROMPT}" \
#     --n_iterations ${N_ITERATIONS} \
#     --chunk_size ${CHUNK_SIZE} \
#     --side ${SIDE} \
#     --use_ik \
#     --dummy_mode

# ===== 模式 2: ROS模式（从ROS topics读取数据） =====
echo "Running in ROS mode..."
python test_pi0_with_ik.py \
    --train_config_name ${TRAIN_CONFIG_NAME} \
    --model_name ${MODEL_NAME} \
    --checkpoint_id ${CHECKPOINT_ID} \
    --pi0_step ${PI0_STEP} \
    --task_prompt "${TASK_PROMPT}" \
    --n_iterations ${N_ITERATIONS} \
    --chunk_size ${CHUNK_SIZE} \
    --side ${SIDE} \
    --use_ik \
    --ros_mode

# ===== 模式 3: RealSense相机模式 =====
# echo "Running in RealSense mode..."
# python test_pi0_with_ik.py \
#     --train_config_name ${TRAIN_CONFIG_NAME} \
#     --model_name ${MODEL_NAME} \
#     --checkpoint_id ${CHECKPOINT_ID} \
#     --pi0_step ${PI0_STEP} \
#     --task_prompt "${TASK_PROMPT}" \
#     --n_iterations ${N_ITERATIONS} \
#     --chunk_size ${CHUNK_SIZE} \
#     --side ${SIDE} \
#     --use_ik

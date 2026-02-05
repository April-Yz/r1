#!/bin/bash
# PI0 测试脚本 - 加载 checkpoint/10000 模型
# 
# 用法:
#   1. 仅测试模型加载（dummy 模式）：
#      ./run_test_pi0.sh dummy
#
#   2. 使用 rosbag 文件作为输入（推荐，不需要 ROS 运行时）：
#      ./run_test_pi0.sh bag /path/to/file.bag
#
#   3. 使用 ZMQ 桥接模式（推荐用于实时测试，解决 Python 版本兼容问题）：
#      # 终端1: 运行 ROS 桥接（使用系统 Python 3.8）
#      /usr/bin/python3 /home/pine/yzj/src/ros_bridge.py
#      # 终端2: 运行 PI0 测试（使用 openpi 虚拟环境 Python 3.11）
#      ./run_test_pi0.sh zmq
#      ./run_test_pi0.sh zmq_ik  # 带 IK 计算
#
#   4. 使用 ROS topics 作为输入（需要 ROS 环境，可能有 Python 版本兼容问题）：
#      ./run_test_pi0.sh ros
#      ./run_test_pi0.sh ros_ik
#
# 参考原 eval.sh 命令:
#   bash eval.sh pour demo_clean R1_FFT_pour_35_0130_5k demo_clean 0 0 10000

# 设置 GPU 显存比例
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# 设置 GPU
export CUDA_VISIBLE_DEVICES=0

# 进入工作目录
cd /home/pine/yzj/src

# 模型配置
TRAIN_CONFIG_NAME="R1_FFT_pour_35_0130_5k"
CHECKPOINT_PATH="/home/pine/yzj/RoboTwin/policy/pi0/checkpoint/10000"
PI0_STEP=30
EXECUTE_STEPS=30  # 每次预测后执行多少步（应 <= PI0_STEP）
EXECUTION_DELAY=0.1  # 执行完动作后等待多少秒再进行下一次预测
TASK_PROMPT="pour"
N_ITERATIONS=10

# 检查参数
MODE=${1:-"dummy"}
BAG_FILE=${2:-""}

echo "========================================"
echo "PI0 Test Script"
echo "========================================"
echo "Train Config: ${TRAIN_CONFIG_NAME}"
echo "Checkpoint:   ${CHECKPOINT_PATH}"
echo "PI0 Step:     ${PI0_STEP}"
echo "Execute Steps:${EXECUTE_STEPS}"
echo "Task Prompt:  ${TASK_PROMPT}"
echo "Mode:         ${MODE}"
if [ -n "${BAG_FILE}" ]; then
    echo "Bag File:     ${BAG_FILE}"
fi
echo "========================================"

case ${MODE} in
    "dummy")
        echo "Running in DUMMY mode (random images)..."
        python test_pi0_ros.py \
            --train_config_name ${TRAIN_CONFIG_NAME} \
            --checkpoint_path ${CHECKPOINT_PATH} \
            --pi0_step ${PI0_STEP} \
            --task_prompt "${TASK_PROMPT}" \
            --n_iterations ${N_ITERATIONS} \
            --dummy_mode \
            --print_only
        ;;
    
    "bag")
        if [ -z "${BAG_FILE}" ]; then
            echo "Error: Please provide bag file path"
            echo "Usage: ./run_test_pi0.sh bag /path/to/file.bag"
            exit 1
        fi
        echo "Running in BAG FILE mode..."
        python test_pi0_ros.py \
            --train_config_name ${TRAIN_CONFIG_NAME} \
            --checkpoint_path ${CHECKPOINT_PATH} \
            --pi0_step ${PI0_STEP} \
            --task_prompt "${TASK_PROMPT}" \
            --n_iterations ${N_ITERATIONS} \
            --bag_file "${BAG_FILE}" \
            --print_only
        ;;
    
    "ros")
        echo "Running in ROS mode (from topics)..."
        python test_pi0_ros.py \
            --train_config_name ${TRAIN_CONFIG_NAME} \
            --checkpoint_path ${CHECKPOINT_PATH} \
            --pi0_step ${PI0_STEP} \
            --task_prompt "${TASK_PROMPT}" \
            --n_iterations ${N_ITERATIONS} \
            --ros_mode \
            --print_only
        ;;
    
    "ros_ik")
        echo "Running in ROS mode with IK computation..."
        python test_pi0_ros.py \
            --train_config_name ${TRAIN_CONFIG_NAME} \
            --checkpoint_path ${CHECKPOINT_PATH} \
            --pi0_step ${PI0_STEP} \
            --task_prompt "${TASK_PROMPT}" \
            --n_iterations ${N_ITERATIONS} \
            --ros_mode \
            --compute_ik
        ;;
    
    "zmq")
        echo "Running in ZMQ BRIDGE mode..."
        echo "=============================================="
        echo "IMPORTANT: First run ros_bridge.py in another terminal:"
        echo "  /usr/bin/python3 /home/pine/yzj/src/ros_bridge.py"
        echo "=============================================="
        python test_pi0_ros.py \
            --train_config_name ${TRAIN_CONFIG_NAME} \
            --checkpoint_path ${CHECKPOINT_PATH} \
            --pi0_step ${PI0_STEP} \
            --task_prompt "${TASK_PROMPT}" \
            --n_iterations ${N_ITERATIONS} \
            --zmq_mode \
            --print_only
        ;;
    
    "zmq_ik")
        echo "Running in ZMQ BRIDGE mode with IK computation..."
        echo "=============================================="
        echo "IMPORTANT: First run ros_bridge.py in another terminal:"
        echo "  /usr/bin/python3 /home/pine/yzj/src/ros_bridge.py"
        echo "=============================================="
        python test_pi0_ros.py \
            --train_config_name ${TRAIN_CONFIG_NAME} \
            --checkpoint_path ${CHECKPOINT_PATH} \
            --pi0_step ${PI0_STEP} \
            --task_prompt "${TASK_PROMPT}" \
            --n_iterations ${N_ITERATIONS} \
            --zmq_mode \
            --compute_ik
        ;;
    
    "zmq_ik_delta")
        echo "Running in ZMQ BRIDGE mode with IK + joint delta..."
        echo "=============================================="
        echo "IMPORTANT: First run ros_bridge.py in another terminal:"
        echo "  /usr/bin/python3 /home/pine/yzj/src/ros_bridge.py"
        echo "=============================================="
        python test_pi0_ros.py \
            --train_config_name ${TRAIN_CONFIG_NAME} \
            --checkpoint_path ${CHECKPOINT_PATH} \
            --pi0_step ${PI0_STEP} \
            --task_prompt "${TASK_PROMPT}" \
            --n_iterations ${N_ITERATIONS} \
            --zmq_mode \
            --compute_ik \
            --show_joint_delta
        ;;
    
    "bag_compare")
        if [ -z "${BAG_FILE}" ]; then
            echo "Error: Please provide bag file path"
            echo "Usage: ./run_test_pi0.sh bag_compare /path/to/file.bag"
            exit 1
        fi
        echo "Running in BAG FILE mode with GT comparison..."
        python test_pi0_ros.py \
            --train_config_name ${TRAIN_CONFIG_NAME} \
            --checkpoint_path ${CHECKPOINT_PATH} \
            --pi0_step ${PI0_STEP} \
            --task_prompt "${TASK_PROMPT}" \
            --n_iterations ${N_ITERATIONS} \
            --bag_file "${BAG_FILE}" \
            --compare_gt_action
        ;;
    
    "bag_full")
        if [ -z "${BAG_FILE}" ]; then
            echo "Error: Please provide bag file path"
            echo "Usage: ./run_test_pi0.sh bag_full /path/to/file.bag"
            exit 1
        fi
        echo "Running in BAG FILE mode with IK + GT comparison + joint delta..."
        python test_pi0_ros.py \
            --train_config_name ${TRAIN_CONFIG_NAME} \
            --checkpoint_path ${CHECKPOINT_PATH} \
            --pi0_step ${PI0_STEP} \
            --task_prompt "${TASK_PROMPT}" \
            --n_iterations ${N_ITERATIONS} \
            --bag_file "${BAG_FILE}" \
            --compute_ik \
            --show_joint_delta \
            --compare_gt_action
        ;;
    
    "zmq_control")
        echo "=============================================="
        echo "⚠️  WARNING: CONTROL MODE - ROBOT WILL MOVE!"
        echo "=============================================="
        echo "IMPORTANT: First run ros_bridge.py in another terminal:"
        echo "  /usr/bin/python3 /home/pine/yzj/src/ros_bridge.py"
        echo ""
        echo "Prediction: ${PI0_STEP} steps, Execute: ${EXECUTE_STEPS} steps"
        echo ""
        echo "Control topics:"
        echo "  /motion_target/target_joint_state_arm_left"
        echo "  /motion_target/target_joint_state_arm_right"
        echo "  /motion_control/position_control_gripper_left"
        echo "  /motion_control/position_control_gripper_right"
        echo ""
        echo "NOTE: Will run continuously until Ctrl+C"
        echo "=============================================="
        read -p "Press Enter to continue or Ctrl+C to cancel..."
        python test_pi0_ros.py \
            --train_config_name ${TRAIN_CONFIG_NAME} \
            --checkpoint_path ${CHECKPOINT_PATH} \
            --pi0_step ${PI0_STEP} \
            --execute_steps ${EXECUTE_STEPS} \
            --execution_delay ${EXECUTION_DELAY} \
            --task_prompt "${TASK_PROMPT}" \
            --n_iterations 999999 \
            --zmq_mode \
            --compute_ik \
            --publish_command \
            --show_joint_delta
        ;;
    
    "zmq_control_confirm")
        echo "=============================================="
        echo "⚠️  WARNING: CONTROL MODE (WITH CONFIRMATION)"
        echo "=============================================="
        echo "IMPORTANT: First run ros_bridge.py in another terminal:"
        echo "  /usr/bin/python3 /home/pine/yzj/src/ros_bridge.py"
        echo ""
        echo "Prediction: ${PI0_STEP} steps, Execute: ${EXECUTE_STEPS} steps"
        echo ""
        echo "This mode will:"
        echo "  1. Send init position → wait for 'yes' to start"
        echo "  2. Execute ${EXECUTE_STEPS} actions per prediction"
        echo "  3. Before each batch → show info + wait for ENTER"
        echo ""
        echo "Control topics:"
        echo "  /motion_target/target_joint_state_arm_left"
        echo "  /motion_target/target_joint_state_arm_right"
        echo "  /motion_control/position_control_gripper_left"
        echo "  /motion_control/position_control_gripper_right"
        echo ""
        echo "NOTE: Will run continuously until Ctrl+C"
        echo "=============================================="
        read -p "Press Enter to continue or Ctrl+C to cancel..."
        python test_pi0_ros.py \
            --train_config_name ${TRAIN_CONFIG_NAME} \
            --checkpoint_path ${CHECKPOINT_PATH} \
            --pi0_step ${PI0_STEP} \
            --execute_steps ${EXECUTE_STEPS} \
            --execution_delay ${EXECUTION_DELAY} \
            --task_prompt "${TASK_PROMPT}" \
            --n_iterations 999999 \
            --zmq_mode \
            --compute_ik \
            --publish_command \
            --show_joint_delta \
            --confirm_each_command
        ;;
    
    "zmq_control_auto")
        echo "=============================================="
        echo "⚠️  WARNING: CONTROL MODE (AUTO EXECUTE)"
        echo "=============================================="
        echo "IMPORTANT: First run ros_bridge.py in another terminal:"
        echo "  /usr/bin/python3 /home/pine/yzj/src/ros_bridge.py"
        echo ""
        echo "Prediction: ${PI0_STEP} steps, Execute: ${EXECUTE_STEPS} steps"
        echo ""
        echo "This mode will:"
        echo "  1. Send init position → wait for 'yes' to start"
        echo "  2. Execute ${EXECUTE_STEPS} actions per prediction"
        echo "  3. AUTO EXECUTE if delta_action < 0.05 AND delta_joint < 0.2"
        echo "  4. Otherwise ask for ENTER confirmation"
        echo ""
        echo "Control topics:"
        echo "  /motion_target/target_joint_state_arm_left"
        echo "  /motion_target/target_joint_state_arm_right"
        echo "  /motion_control/position_control_gripper_left"
        echo "  /motion_control/position_control_gripper_right"
        echo ""
        echo "NOTE: Will run continuously until Ctrl+C"
        echo "=============================================="
        read -p "Press Enter to continue or Ctrl+C to cancel..."
        python test_pi0_ros.py \
            --train_config_name ${TRAIN_CONFIG_NAME} \
            --checkpoint_path ${CHECKPOINT_PATH} \
            --pi0_step ${PI0_STEP} \
            --execute_steps ${EXECUTE_STEPS} \
            --execution_delay ${EXECUTION_DELAY} \
            --task_prompt "${TASK_PROMPT}" \
            --n_iterations 999999 \
            --zmq_mode \
            --compute_ik \
            --publish_command \
            --show_joint_delta \
            --confirm_each_command \
            --auto_execute_threshold 0.10 \
            --auto_joint_threshold 0.2
        ;;
    
    *)
        echo "Unknown mode: ${MODE}"
        echo "Usage: ./run_test_pi0.sh <mode> [bag_file_path]"
        echo ""
        echo "Available modes:"
        echo "  dummy               - Test model loading with random images"
        echo "  bag                 - Read from rosbag file (print action only)"
        echo "  bag_compare         - Read from rosbag file + compare with GT action"
        echo "  bag_full            - Read from rosbag file + IK + GT comparison + joint delta"
        echo "  ros                 - Read from ROS topics (may have Python version issues)"
        echo "  ros_ik              - Read from ROS topics + IK computation"
        echo "  zmq                 - Read via ZMQ bridge (print action only)"
        echo "  zmq_ik              - Read via ZMQ bridge + IK computation"
        echo "  zmq_ik_delta        - Read via ZMQ bridge + IK + joint delta"
        echo "  zmq_control         - ⚠️ CONTROL MODE: Send commands to robot via ZMQ"
        echo "  zmq_control_confirm - ⚠️ CONTROL MODE: Confirm before each command"
        echo "  zmq_control_auto    - ⚠️ CONTROL MODE: Auto execute if delta < threshold"
        exit 1
        ;;
esac
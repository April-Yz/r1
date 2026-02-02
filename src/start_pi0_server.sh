#!/bin/bash
# 启动 PI0 WebSocket 服务器
# 模型将以服务方式运行，客户端通过 WebSocket 发送请求
#
# 用法:
#   ./start_pi0_server.sh
#
# 然后在另一个终端运行客户端测试

# 设置环境变量减少内存使用
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0

# 进入 pi0 目录
cd /home/pine/yzj/RoboTwin/policy/pi0

# 激活虚拟环境
source .venv/bin/activate

echo "========================================"
echo "Starting PI0 WebSocket Server"
echo "========================================"
echo "Config: R1_FFT_pour_35_0130_5k"
echo "Checkpoint: ./checkpoint/10000"
echo "Port: 8000"
echo "========================================"

# 启动服务器
python scripts/serve_policy.py \
    --config R1_FFT_pour_35_0130_5k \
    --dir ./checkpoint/10000 \
    --port 8000

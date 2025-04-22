#!/bin/bash
# 机械臂预测执行脚本
# 使用方法: ./run_predict.sh

# 配置参数
CONFIG=(
    "--model_dir ./robot_arm_model"
    "--input_dir ./sample_data/predict/origin"
    "--output_dir ./sample_data/predict/predict_50frames"
    "--prompt '预测50帧后的机械臂状态'"
    "--seed 42"
)

# 环境检查
echo "=== 机械臂状态预测系统 ==="
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "----------------------------------------"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到Python3环境"
    exit 1
fi

# 检查模型目录
if [ ! -d "./robot_arm_model" ]; then
    echo "[错误] 模型目录不存在，请先运行微调脚本"
    exit 1
fi

# 检查输入文件
if [ ! -d "./sample_data/predict/origin" ]; then
    echo "[错误] 输入目录不存在"
    exit 1
fi

# 创建日志目录
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/predict_$(date +%Y%m%d_%H%M%S).log"

# 执行预测
echo "[系统] 启动预测过程..."
echo "使用参数:"
printf "  %s\n" "${CONFIG[@]}"
echo "----------------------------------------"

python3 predict_with_model.py ${CONFIG[@]} 2>&1 | tee "$LOG_FILE"

# 结果统计
SUCCESS=$(grep "预测成功" "$LOG_FILE" | wc -l)
FAILED=$(grep "预测失败" "$LOG_FILE" | wc -l)

echo "----------------------------------------"
echo "[结果] 预测完成:"
echo "  成功: $SUCCESS 张"
echo "  失败: $FAILED 张"
echo "日志文件: $LOG_FILE"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=== 操作结束 ==="
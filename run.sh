#!/bin/bash

# 视频处理脚本启动器


# 设置环境变量
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_SCRIPT="$SCRIPT_DIR/combine_detect.py"  # 替换为你的Python脚本实际文件名
CONFIG_FILE="$SCRIPT_DIR/config.ini"  # 替换为你的配置文件实际文件名

# 检查Python是否可用
if ! command -v python3 &> /dev/null
then
    echo "Python3未找到，请先安装Python3"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 未找到配置文件 $CONFIG_FILE"
    echo "请创建配置文件并设置必要的路径参数"
    exit 1
fi

# 检查Python脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: 未找到Python处理脚本 $PYTHON_SCRIPT"
    exit 1
fi

# 运行Python脚本
echo "启动视频处理流程..."
echo "配置文件: $CONFIG_FILE"
python3 "$PYTHON_SCRIPT"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo "视频处理成功完成!"
else
    echo "视频处理过程中出现错误!"
fi

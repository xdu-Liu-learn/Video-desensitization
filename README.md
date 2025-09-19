# Video-desensitization
Blur the faces and license plates in the video.

## Environment configuration1 (WSL+conda)
    #GPU RTX5090算力过高，目前适配cuda12.8以及torch2.7版本，cudnn8.9.6
    conda  create -n FLPR python=3.10
    export LD_LIBRARY_PATH=/path/to/library:$LD_LIBRARY_PATH #配置recordDeal库，/path/to/library为.so文件所处目录
    source ~/.bashrc
    conda activate FLPR
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements.txt
    wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.11/flash_attn-2.8.3+cu128torch2.7-cp310-cp310-linux_x86_64.whl
    pip install flash_attn-2.8.3+cu128torch2.7-cp310-cp310-linux_x86_64.whl
    pip install cyber_record
    pip install av
    pip install ultralytics
    python combine_detect.py 或者   bash run.sh
    
### Install ffmpeg for decoding and encoding
    sudo apt-get update
    sudo apt install ffmpeg
    ffmpeg -version

### face model_path
    推理所需的Retinaface_resnet50.pth等文件可以在百度云下载。
    链接: https://pan.baidu.com/s/1Jt9Bo2UVP03bmEMuUpk_9Q 提取码: qknw

## Environment configuration2(docker)
    # docker配置环境+手动安装flash_attn-2.8.3+cu128torch2.7-cp310-cp310-linux_x86_64.whl

## 安装Docker Desktop
    开启windows可选功能虚拟化
	用WSL作为默认内核
## 创建dockerfile文件"FLPR.dockerfile"
	# 基础镜像
	FROM ubuntu:22.04
	
	# 避免交互式安装卡住
	ENV DEBIAN_FRONTEND=noninteractive
	
	# 安装基础工具、curl、ffmpeg 和证书
	RUN apt-get update && apt-get install -y \
	    software-properties-common \
	    curl \
	    git \
	    wget \
	    build-essential \
	    ca-certificates \
	    ffmpeg \
	    python3.10 \
	    python3.10-dev \
	    python3.10-venv \
	    python3-pip \
	    && rm -rf /var/lib/apt/lists/*
	
	# 设置 Python3.10 和 pip 默认
	RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
	    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
	
	# 安装 Python 包（来自 requirements.txt）
	RUN pip install --no-cache-dir \
	    timm==1.0.14 \
	    albumentations==2.0.4 \
	    onnx==1.12.0 \
	    onnxruntime==1.15.1 \
	    pycocotools==2.0.7 \
	    PyYAML==6.0.1 \
	    scipy==1.13.0 \
	    onnxslim==0.1.31 \
	    onnxruntime-gpu==1.18.0 \
	    gradio==4.44.1 \
	    opencv-python==4.9.0.80 \
	    psutil==5.9.8 \
	    py-cpuinfo==9.0.0 \
	    huggingface-hub==0.23.2 \
	    safetensors==0.4.3 \
	    numpy==1.26.4 \
	    supervision==0.22.0
	
	# 安装 PyTorch GPU 指定版本
	RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
	
	# 安装 flash-attn wheel（用 curl 下载）
	# RUN curl -L -o /tmp/flash_attn.whl "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.11/flash_attn-2.8.3+cu128torch2.7-cp310-cp310-linux_x86_64.whl" \
	#     && pip install /tmp/flash_attn.whl \
	#     && rm /tmp/flash_attn.whl
	
	# 安装其他 Python 包
	RUN pip install cyber_record av ultralytics retina-face
	
	# 设置工作目录
	WORKDIR /workspace
	
	# 默认命令
	CMD [ "bash" ]

## dockerfile创建image“ubt2204_ccic”
    docker build -t ubt2204_ccic -f "D:\ubt2204_CCIC.dockerfile" "D:\"

## 以image创建docker，并使其运行 ，进入docker环境，挂载磁盘，开启gpu加速
    docker run -it --user root --name ubtV7 --gpus all -v /mnt/d/P_WorkSpace/Video-desensitization:/mnt/d/P_WorkSpace/Video-desensitization ubt2204_ccic

## 已有docker，进入docker环境
    docker start -ai ubtV7

 ## 手动安装flash_attn(源下载速度慢，提前下载，手动配置)
	cd /mnt/d/P_WorkSpace/Video-desensitization/
  	pip install flash_attn-2.8.3+cu128torch2.7-cp310-cp310-linux_x86_64.whl 
   





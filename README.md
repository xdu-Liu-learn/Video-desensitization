# Video-desensitization
Blur the faces and license plates in the video.

# Environment configuration
    #GPU RTX5090算力过高，目前适配cuda12.8以及torch2.7版本，cudnn8.9.6
    conda  create -n FLPR python=3.10
    export LD_LIBRARY_PATH=/path/to/library:$LD_LIBRARY_PATH #配置recordDeal库，/path/to/library为.so文件所处目录
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements.txt
    wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.11/flash_attn-2.8.3+cu128torch2.7-cp310-cp310-linux_x86_64.whl
    pip install flash_attn-2.8.3+cu128torch2.7-cp310-cp310-linux_x86_64.whl
    pip install cyber_record
    pip install av
    pip install ultralytics
    python combine_detect.py 或者   bash run.sh
    


# Install ffmpeg for decoding and encoding
    sudo apt-get update
    sudo apt install ffmpeg
    ffmpeg -version



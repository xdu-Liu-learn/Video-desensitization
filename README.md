# Video-desensitization
Blur the faces and license plates in the video.

# Environment configuration
    conda create -n FLPR python==3.8
    conda activate FLPR
    pip install -r requirements.txt
    pip install --upgrade "tensorflow<=2.13.0"
    pip install ultralytics
    pip install cyber_record,protobuf
    pip install pickle

# Install ffmpeg for decoding and encoding
    sudo apt-get update
    sudo apt install ffmpeg
    ffmpeg -version



# Video-desensitization
Blur the faces and license plates in the video.
# Environment configuration
pip install -r requirements.txt
pip install --upgrade "tensorflow<=2.13.0"
pip install ultralytics
# Install ffmpeg for decoding and encoding
sudo apt-get update
sudo apt install ffmpeg
ffmpeg -version

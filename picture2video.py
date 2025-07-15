import os
import glob
import subprocess
import shutil
import sys
import cv2

#def create_h265_video(input_dir, output_video, fps=30):
#    """
#    Create H.265 raw video bitstream from image sequence using FFmpeg
#    :param input_dir: Input image directory
#    :param output_video: Output video path (should end with .h265)
#    :param fps: Frame rate (default 30)
#    :return: True if successful, False otherwise
#    """
#    # Ensure output directory exists
#    output_dir = os.path.dirname(output_video) or '.'
#    if not os.path.exists(output_dir):
#        os.makedirs(output_dir)
#    
#    # Check FFmpeg installation
#    if shutil.which("ffmpeg") is None:
#        print("Error: FFmpeg is not installed or not in system PATH")
#        print("Please install FFmpeg to use this function")
#        return False
#    
#    # Get all image files
#    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
#    image_paths = []
#    for ext in image_extensions:
#        image_paths.extend(glob.glob(os.path.join(input_dir, f'*.{ext}')))
#        image_paths.extend(glob.glob(os.path.join(input_dir, f'*.{ext.upper()}')))
#    
#    if not image_paths:
#        print(f"Error: No image files found in directory '{input_dir}'")
#        return False
#    
#    # Sort by filename
#    image_paths.sort()
#    
#    print("Creating H.265 raw bitstream using FFmpeg...")
#    
#    try:
#        # Create temporary text file with image paths
#        list_file = "ffmpeg_input.txt"
#        with open(list_file, 'w') as f:
#            for path in image_paths:
#                f.write(f"file '{path}'\n")
#        
#        # FFmpeg command for H.265 raw bitstream encoding
#        ffmpeg_cmd = [
#            "ffmpeg",
#            "-y",  # Overwrite output without asking
#            "-f", "concat",
#            "-safe", "0",
#            "-r", str(fps),
#            "-i", list_file,
#            "-c:v", "libx265",
#            "-crf", "23",  # Quality level (0-51, lower is better)
#            "-preset", "medium",  # Encoding speed/compression tradeoff
#            "-pix_fmt", "yuv420p",  # Pixel format for compatibility
#            "-x265-params", "log-level=error",  # Suppress unnecessary logs
#            "-f", "hevc",  # Force HEVC raw bitstream output
#            "-an",  # No audio
#            output_video  # Output should have .h265 extension
#        ]
#        
#        # Run FFmpeg command
#        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
#        
#        # Clean up temporary file
#        os.remove(list_file)
#        
#        if result.returncode == 0:
#            print(f"Successfully created H.265 raw bitstream: {output_video}")
#            print(f"Note: This is a raw HEVC bitstream. Play with: ffplay -f hevc {output_video}")
#            return True
#        
#        print(f"FFmpeg error: {result.stderr}")
#        return False
#        
#    except Exception as e:
#        print(f"Video creation error: {str(e)}")
#        return False

# 替换原有的 create_h265_video 函数
def create_video(image_folder, output_video_path, fps=60):
    """
    通用视频创建函数，支持多种格式
    :param image_folder: 包含图片的文件夹路径
    :param output_video_path: 输出视频文件路径（包含扩展名）
    :param fps: 视频帧率
    :return: 是否成功创建视频
    """
    # 获取输出视频的扩展名（不含点）
    video_ext = os.path.splitext(output_video_path)[1].lower().lstrip('.')
    
    # 根据文件扩展名选择合适的编码器
    codec_map = {
        'h265': 'hevc',
        'hevc': 'hevc',
        '265': 'hevc',
        'mp4': 'mp4v',
        'mov': 'avc1',
        'avi': 'XVID',
        'mkv': 'X264'
    }
    
    # 设置默认编码器为 mp4v (适用于 .mp4 文件)
    default_codec = 'mp4v'
    
    # 选择合适的编码器
    codec_str = codec_map.get(video_ext, default_codec)
    
    # 获取图片列表（按文件名排序）
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    images = [img for img in os.listdir(image_folder) 
              if os.path.splitext(img)[1].lower() in image_extensions]
    
    if not images:
        print(f"错误: 在目录 {image_folder} 中未找到图片")
        return False
    
    # 按文件名排序以确保正确的顺序
    images.sort()
    
    # 读取第一张图片获取尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"无法读取第一张图片: {first_image_path}")
        return False
        
    height, width = frame.shape[:2]
    
    # 尝试获取编码器
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec_str)
    except:
        print(f"警告: 不支持的编码器 '{codec_str}', 尝试使用默认编码器")
        fourcc = cv2.VideoWriter_fourcc(*default_codec)
    
    # 创建视频写入对象
    try:
        output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not output_video.isOpened():
            print(f"无法创建视频文件: {output_video_path}")
            print(f"请检查路径、权限、编码器是否支持。使用的FourCC: {fourcc}")
            return False
    except Exception as e:
        print(f"创建视频写入器失败: {e}")
        return False
    
    # 写入图片
    print(f"开始创建视频: {output_video_path} (格式: {video_ext}, 编码: {codec_str})")
    frame_count = 0
    for i, image in enumerate(images):
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"跳过无法读取的图片: {image}")
            continue
            
        # 调整图片尺寸以匹配第一帧
        if frame.shape[:2] != (height, width):
            try:
                frame = cv2.resize(frame, (width, height))
            except:
                print(f"无法调整图片尺寸: {image}")
                continue
        
        output_video.write(frame)
        frame_count += 1
        
        # 打印进度
        if (i+1) % 100 == 0:
            print(f"已写入 {i+1}/{len(images)} 帧...")
    
    output_video.release()
    print(f"视频保存成功: {output_video_path}, 共写入 {frame_count} 帧")
    return True

if __name__ == "__main__":
    input_directory = "/home/24181214123/yolo/runs/detect/predict4"
    output_video = "/home/24181214123/yolo/output.h265"  # Changed to .h265 extension
    frame_rate = 60
    
    success = create_h265_video(input_directory, output_video, frame_rate)
    
    if success:
        print("H.265 raw bitstream created successfully!")
        # Open output directory on Windows
        if os.name == 'nt' and os.path.exists(output_video):
            output_dir = os.path.dirname(output_video) or '.'
            os.startfile(output_dir)
    else:
        print("Failed to create video")
        sys.exit(1)

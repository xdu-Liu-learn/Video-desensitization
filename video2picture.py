import cv2
import os
from tqdm import tqdm

def check_h265_support():
    """检查系统是否支持H.265解码的改进方法"""
    # 创建临时测试文件路径
    test_file = "test_h265_check.hevc"
    
    # 尝试使用不同的FourCC代码
    for codec in ['HEVC', 'H265', 'hvc1']:
        try:
            # 尝试创建视频写入器（如果能创建写入器，通常也能读取）
            fourcc = cv2.VideoWriter_fourcc(*codec)
            temp_writer = cv2.VideoWriter(test_file, fourcc, 30, (640, 480))
            if temp_writer.isOpened():
                temp_writer.release()
                os.remove(test_file)  # 删除临时文件
                return True
        except:
            continue
    
    # 如果上面的方法不行，尝试直接打开一个不存在的文件
    cap = cv2.VideoCapture()
    for codec in ['HEVC', 'H265', 'hvc1']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        if cap.open('nonexistent_file.hevc', cv2.CAP_FFMPEG):
            cap.release()
            return True
    cap.release()
    return False

def convert_video_to_frames(video_path, output_folder, format='jpg', prefix='frame'):
    """
    使用OpenCV将单个视频转换为图片帧
    
    参数:
        video_path (str): 输入视频文件路径
        output_folder (str): 输出图片文件夹
        format (str): 输出图片格式(jpg/png)
        prefix (str): 输出图片前缀
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        print("可能原因:")
        print("1. 文件路径不正确")
        print("2. 系统缺少H.265/HEVC编解码器")
        print("3. OpenCV版本不支持此视频格式")
        return 0
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 使用进度条
    with tqdm(total=total_frames, desc=f"处理 {os.path.basename(video_path)}", unit='frame') as pbar:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # 保存帧为图片
            output_path = os.path.join(output_folder, f"{prefix}_{frame_count:06d}.{format}")
            success = cv2.imwrite(output_path, frame)
            
            if not success:
                print(f"无法保存帧 {frame_count} 到 {output_path}")
                break
                
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    return frame_count

def batch_convert_videos(input_dir, output_parent_dir, format='jpg'):
    """
    批量转换文件夹中的所有视频文件
    
    参数:
        input_dir (str): 包含视频的输入文件夹
        output_parent_dir (str): 输出父文件夹
        format (str): 输出图片格式(jpg/png)
    """
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录 {input_dir} 不存在")
        return
    
    # 检查H.265支持
    print("正在检查H.265/HEVC支持...")
    if not check_h265_support():
        print("警告: 系统可能不支持H.265解码，转换可能失败")
        print("建议解决方案:")
        print("1. 安装FFmpeg (推荐)")
        print("2. 安装系统编解码器包")
        print("   - Windows: 安装HEVC视频扩展或K-Lite Codec Pack")
        print("   - Ubuntu: sudo apt install libx265-dev")
        print("3. 使用FFmpeg替代方案")
        proceed = input("是否继续尝试? (y/n): ").lower()
        if proceed != 'y':
            return
    
    # 确保输出目录存在
    os.makedirs(output_parent_dir, exist_ok=True)
    
    # 获取所有视频文件(支持多种扩展名)
    video_extensions = ('.h265', '.hevc', '.265', '.mp4', '.mov', '.avi')
    video_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(video_extensions)]
    
    if not video_files:
        print(f"在 {input_dir} 中没有找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件，开始转换...")
    
    total_frames = 0
    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        
        # 为每个视频创建单独的输出文件夹
        video_name = os.path.splitext(video_file)[0]
        output_dir = os.path.join(output_parent_dir, video_name)
        
        # 转换视频
        print(f"\n开始处理: {video_file}")
        frames_saved = convert_video_to_frames(video_path, output_dir, format)
        
        if frames_saved > 0:
            print(f"{video_file} 转换完成，保存了 {frames_saved} 帧到 {output_dir}")
            total_frames += frames_saved
        else:
            print(f"{video_file} 转换失败")
    
    print(f"\n所有视频处理完成！共保存了 {total_frames} 张图片")

# if __name__ == "__main__":
#     # 配置参数
#     INPUT_DIR = "C:\\Users\\Sprite\\Desktop\\hevcs\\video_2"  # 使用原始字符串表示Windows路径
#     OUTPUT_DIR = "video2picture"  # 输出父文件夹
#     IMAGE_FORMAT = "jpg"  # 输出图片格式(jpg/png)
    
#     # 执行批量转换
#     batch_convert_videos(INPUT_DIR, OUTPUT_DIR, IMAGE_FORMAT)

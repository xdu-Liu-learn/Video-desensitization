import cv2
import numpy as np
import os
import glob
import time
import configparser
import shutil
import torch
from ultralytics import YOLO
import logging
import sys
# from skimage import transform as trans
# from dataloader import VideoFrameDataset
import subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
from detect_face.face import Retinaface
from foreign import recordDeal  
import tarfile

# 配置全局日志器
def setup_logger(log_file='video_processing.log'):
    logger = logging.getLogger('VideoProcessor')
    logger.setLevel(logging.DEBUG)
    
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 清除已有的处理器
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 创建控制台处理器和文件处理器
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)
    
    # 设置日志级别
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def check_available_codecs():
    """检查系统可用的视频编码器"""
    logger = logging.getLogger('VideoProcessor')
    
    # 测试常用编码器
    test_codecs = [
        ('mp4v', 'MP4V'),
        ('avc1', 'H.264'),
        ('XVID', 'XVID'),
        ('MJPG', 'MJPEG'),
        ('X264', 'H.264'),
        ('HEVC', 'HEVC/H.265')
    ]
    
    available_codecs = []
    width, height, fps = 640, 480, 30
    
    for codec_code, codec_name in test_codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_code)
            test_path = f'test_codec_{codec_code}.mp4'
            test_writer = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
            
            if test_writer.isOpened():
                available_codecs.append((codec_code, codec_name))
                test_writer.release()
            
            try:
                if os.path.exists(test_path):
                    os.remove(test_path)
            except:
                pass
                
        except Exception as e:
            logger.debug(f"编码器 {codec_code} 测试失败: {e}")
    
    return available_codecs
    


def mosaic_rectangle_region(self, batch_imgs, x1, y1, x2, y2, mosaic_level=8):
        """
        对批量图像中的矩形区域打马赛克
        Args:
            batch_imgs: list of np.ndarray, shape (H, W, 3), RGB or BGR
            x1, y1: 矩形左上角坐标
            x2, y2: 矩形右下角坐标
            mosaic_level: 马赛克强度，值越大块越大（建议 4~16）

        Returns:
            list of np.ndarray, 打码后的图像列表
        """
        mosaic_imgs = []

        for img in batch_imgs:
            img = img.copy()
            h, w = img.shape[:2]

            # 边界裁剪
            x1_clipped = max(0, x1)
            y1_clipped = max(0, y1)
            x2_clipped = min(w, x2)
            y2_clipped = min(h, y2)

            # 检查有效区域
            if x2_clipped > x1_clipped and y2_clipped > y1_clipped:
                # 提取区域
                area = img[y1_clipped:y2_clipped, x1_clipped:x2_clipped]

                # 缩小 -> 放大 = 马赛克效果
                small_h = max(1, (y2_clipped - y1_clipped) // mosaic_level)
                small_w = max(1, (x2_clipped - x1_clipped) // mosaic_level)

                small = cv2.resize(area, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
                mosaic = cv2.resize(small, (x2_clipped - x1_clipped, y2_clipped - y1_clipped),
                                interpolation=cv2.INTER_NEAREST)
                # 替换原图区域
                img[y1_clipped:y2_clipped, x1_clipped:x2_clipped] = mosaic

            # 添加到结果列表
            mosaic_imgs.append(img)

        return mosaic_imgs
        
  
def mosaic_rectangle_region_single(img, x1, y1, x2, y2, mosaic_level=8):
    """
    对单张图像的矩形区域打马赛克
    """
    img = img.copy()
    h, w = img.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return img

    area = img[y1:y2, x1:x2]
    small_h = max(1, (y2 - y1) // mosaic_level)
    small_w = max(1, (x2 - x1) // mosaic_level)

    small = cv2.resize(area, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = mosaic

    return img


# -------------------------------
# 图像加载函数（OpenCV + RGB 转换）
# -------------------------------
def load_image_rgb(image_path):
    """ 使用 OpenCV 加载并转为 RGB numpy array """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -------------------------------
# 异步保存函数
# -------------------------------
def save_output_image(image_array, output_path):
    """ 保存 RGB numpy 数组为图像 """
    bgr_img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr_img)


def batch_process_images(input_dir, output_dir, face_detector, plate_detector, batch_size=16):
    logger = logging.getLogger('VideoProcessor.batch_process_images')
    
    image_paths = [
        os.path.join(input_dir, fname) for fname in os.listdir(input_dir)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    total_images = len(image_paths)
    print(f"找到 {total_images} 张待处理图片。")
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录已创建: {output_dir}")
    
    total_processed = 0
    total_faces = 0
    total_plates = 0
    
    num_workers = 6
    executor = ThreadPoolExecutor(max_workers=num_workers)
    save_futures = []

    for i in range(0, len(image_paths), batch_size):
        batch_files = image_paths[i:i + batch_size]
        batch_start_time = time.time()

        # 多线程加载图像
        load_start = time.time()
        with ThreadPoolExecutor(max_workers=num_workers) as loader:
            batch_images = list(loader.map(load_image_rgb, batch_files))
        print("批处理中加载图像总耗时: {:.2f}s".format(time.time() - load_start))
        # 并行执行两个模型
        with ThreadPoolExecutor(max_workers=2) as infer_executor:
            face_start = time.time()
            future_face = infer_executor.submit(face_detector.detect_images, batch_images.copy())
            future_plate = infer_executor.submit(plate_detector, batch_images.copy(),verbose=False, conf=0.5)
            try:
                result_time_start = time.time()
                face_results = future_face.result()  # List of (img, boxes)
                result_middle_time = time.time()
                print("批处理中人脸推理总耗时: {:.2f}s".format(result_middle_time - face_start))
                plate_results = future_plate.result()  # List of (img, boxes)
                print("批处理中模型推理总耗时: {:.2f}s".format(time.time() - face_start))
                
            except Exception as e:
                logger.error(f"并行推理出错: {e}")
                continue

        # 提取检测框（假设 detect_faces/detect_plates 返回 (image, boxes) 元组列表）
        # 或者你也可以只返回 boxes，不返回 image（更高效）
        # 示例格式: [(img1, [box1, box2]), ...]
        mosaic_time_start = time.time()
        processed_batch = []
        for j, img in enumerate(batch_images):
            # 获取人脸框
            face_boxes = face_results[j][1] if isinstance(face_results[j], tuple) else []
            # 获取车牌框
            plate_boxes = plate_results[j][1] if isinstance(plate_results[j], tuple) else []

            # 合并所有框
            all_boxes = []
            all_boxes.extend([(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in face_boxes])
            all_boxes.extend([(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in plate_boxes])

            # 打马赛克
            img_with_mosaic = img.copy()
            for (x1, y1, x2, y2) in all_boxes:
                img_with_mosaic = mosaic_rectangle_region_single(img_with_mosaic, x1, y1, x2, y2, mosaic_level=8)

            processed_batch.append(img_with_mosaic)

            # 统计
            total_faces += len(face_boxes)
            total_plates += len(plate_boxes)
        print("批处理中马赛克总耗时: {:.2f}s".format(time.time() - mosaic_time_start))
        # 异步保存
        save_time_start = time.time()
        for img_path, result_img in zip(batch_files, processed_batch):
            output_path = os.path.join(output_dir, f"processed_{os.path.basename(img_path)}")
            future = executor.submit(save_output_image, result_img, output_path)
            save_futures.append(future)
        print("批处理中保存总耗时: {:.2f}s".format(time.time() - save_time_start))
        total_processed += len(batch_files)
        batch_time = time.time() - batch_start_time
        print(f"批次 {i//batch_size+1} 处理完成，耗时: {batch_time:.2f}s | 处理 {len(batch_files)} 张")

    # 等待所有保存完成
    for future in as_completed(save_futures):
        try:
            future.result()
        except Exception as e:
            logger.error(f"保存图像失败: {e}")

    logger.info(f"批处理完成! 共处理 {total_processed} 张图片")
    logger.info(f"检测结果: {total_faces} 个人脸, {total_plates} 个车牌")
    return total_processed, total_faces, total_plates

def convert_video_to_frames(video_path, output_dir, interval=1):
    """WSL2 专用 GPU 抽帧函数（解决小文件 I/O 瓶颈）"""
    logger = logging.getLogger('VideoProcessor.convert_video_to_frames')
    
    # 开始计时
    start_time = time.time()
    gpu_processing_time = 0  # 单独计时 GPU 处理
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 清空旧文件
    for f in glob.glob(os.path.join(output_dir, '*.jpg')):
        os.remove(f)
    
    # 确定输出帧格式
    output_pattern = os.path.join(output_dir, "frame_%06d.jpg")
    saved_count = 0
    
    try:
        # === 检测WSL2环境 ===
        is_wsl2 = False
        try:
            with open('/proc/sys/kernel/osrelease', 'r') as f:
                is_wsl2 = 'microsoft' in f.read().lower()
        except:
            pass
        
        # === 关键：路径处理策略 ===
        local_temp_dir = None
        win_video_path = video_path
        win_output_pattern = output_pattern
        
        if is_wsl2 and video_path.startswith('/mnt/'):
            logger.info("🖥️ 检测到WSL2环境，应用小文件I/O优化策略")
            
            # 创建临时目录（在WSL2本地）
            local_temp_dir = "/tmp/video_processing"
            os.makedirs(local_temp_dir, exist_ok=True)
            
            # 复制视频到WSL2本地
            local_video_path = os.path.join(
                local_temp_dir, 
                os.path.basename(video_path)
            )
            if not os.path.exists(local_video_path):
                logger.info(f"📂 复制视频到WSL2本地: {video_path} → {local_video_path}")
                shutil.copy2(video_path, local_video_path)
            
            # 使用本地路径
            win_video_path = local_video_path
            
            # 为输出创建本地临时目录
            local_output_dir = os.path.join(local_temp_dir, "frames")
            os.makedirs(local_output_dir, exist_ok=True)
            win_output_pattern = os.path.join(local_output_dir, "frame_%06d.jpg")
        
        # === 检查GPU可用性 ===
        gpu_available = False
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                logger.info(f"✅ 检测到GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
        except:
            pass
        
        # === 构建GPU命令 ===
        if gpu_available and is_wsl2:
            logger.info("🔥 尝试使用NVIDIA CUDA硬件加速 (小文件I/O优化模式)...")
            
            gpu_command = [
                'ffmpeg',
                '-hide_banner', '-loglevel', 'warning',
                '-hwaccel', 'cuda',
                '-extra_hw_frames', '32',
                '-c:v', 'hevc_cuvid',
                '-i', win_video_path,
                '-vsync', '0',
                '-qscale:v', '2',
                win_output_pattern
            ]
            
            if interval > 1:
                gpu_command.extend(['-vf', f'select=not(mod(n\\,{interval-1}))'])
            
            logger.info(f"🎬 执行GPU命令: {' '.join(gpu_command)}")
            
            # === 关键：单独计时GPU处理 ===
            gpu_start = time.time()
            try:
                # 尝试GPU抽帧
                result = subprocess.run(
                    gpu_command,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
                
                # 检查结果
                saved_frames = glob.glob(win_output_pattern.replace('%06d', '*'))
                saved_count = len(saved_frames)
                
                if saved_count > 0:
                    gpu_processing_time = time.time() - gpu_start
                    logger.info(f"⚡ GPU处理完成 (本地): {saved_count}帧/{gpu_processing_time:.2f}秒 = {saved_count/gpu_processing_time:.1f}帧/秒")
                    
                    # === 关键：优化复制策略 ===
                    if is_wsl2 and video_path.startswith('/mnt/'):
                        logger.info("💾 优化复制: 打包帧文件后一次性传输")
                        
                        # 打包帧文件
                        tar_path = os.path.join(local_temp_dir, "frames.tar")
                        with tarfile.open(tar_path, "w") as tar:
                            for f in saved_frames:
                                tar.add(f, arcname=os.path.basename(f))
                        
                        # 复制tar文件
                        shutil.copy2(tar_path, output_dir)
                        
                        # 在目标目录解压
                        with tarfile.open(os.path.join(output_dir, "frames.tar"), "r") as tar:
                            tar.extractall(path=output_dir)
                        
                        # 清理
                        os.remove(os.path.join(output_dir, "frames.tar"))
                        shutil.rmtree(os.path.dirname(win_output_pattern), ignore_errors=True)
                        
                        logger.info("✅ 帧文件已优化复制到Windows目录")
                    
                    logger.info(f"✅ GPU抽帧成功! 共保存 {saved_count} 帧到 {output_dir}")
                    return saved_count
                else:
                    logger.warning("⚠️ GPU命令执行成功但未生成帧文件")
                    
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ GPU抽帧失败 (返回码: {e.returncode})")
                logger.error(f"FFmpeg错误输出:\n{e.stderr}")
        
        # === CPU回退 ===
        logger.info("💻 使用CPU模式进行抽帧...")
        cpu_command = [
            'ffmpeg',
            '-hide_banner', '-loglevel', 'warning',
            '-threads', str(max(1, os.cpu_count()-2)),
            '-i', video_path,
            '-qscale:v', '2',
            output_pattern
        ]
        
        if interval > 1:
            cpu_command.extend(['-vf', f'select=not(mod(n\\,{interval-1}))'])
        
        logger.info(f"🎬 执行CPU命令: {' '.join(cpu_command)}")
        
        result = subprocess.run(
            cpu_command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        saved_frames = glob.glob(os.path.join(output_dir, '*.jpg'))
        saved_count = len(saved_frames)
        if saved_count > 0:
            logger.info(f"✅ CPU抽帧完成，共保存 {saved_count} 帧到 {output_dir}")
    
    except Exception as e:
        logger.exception(f"❌ 抽帧过程中发生致命错误: {str(e)}")
    finally:
        # 计算性能指标
        end_time = time.time()
        total_time = end_time - start_time
        fps = saved_count / max(total_time, 0.1)
        
        # 生成详细性能报告
        perf_msg = f"⏱️ 总处理时间: {total_time:.2f}秒"
        
        if gpu_available and gpu_processing_time > 0:
            gpu_fps = saved_count / max(gpu_processing_time, 0.1)
            copy_time = total_time - gpu_processing_time
            
            perf_msg += f"\n   ⚡ GPU处理: {gpu_processing_time:.2f}秒 ({gpu_fps:.1f}帧/秒)"
            perf_msg += f"\n   💾 复制操作: {copy_time:.2f}秒"
            
            if gpu_fps > 200:
                perf_msg += "\n   ✅ GPU加速成功!"
            if copy_time > 3:
                perf_msg += "\n   ⚠️ 复制操作是主要瓶颈!"
        
        perf_msg += f"\n   📊 最终速度: {fps:.1f}帧/秒"
        
        logger.info(perf_msg)
        
    return saved_count


def create_video(frame_dir, output_path, fps=60):
    """将图片帧合成为视频"""
    logger = logging.getLogger('VideoProcessor.create_video')
    
    # 获取所有帧文件并排序
    frame_files = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')))
    if not frame_files:
        logger.error(f"在目录 {frame_dir} 中未找到图片帧")
        return False
    
    # 获取帧尺寸
    sample_frame = cv2.imread(frame_files[0])
    if sample_frame is None:
        logger.error(f"无法读取样本帧: {frame_files[0]}")
        return False
        
    height, width = sample_frame.shape[:2]
    
    # 确定输出视频编码器
    ext = os.path.splitext(output_path)[1].lower()
    
    # 定义编码器优先级列表
    codec_priority = {
        '.mp4': [('mp4v', 'mp4v'), ('avc1', 'H.264'), ('X264', 'H.264')],
        '.m4v': [('mp4v', 'mp4v'), ('avc1', 'H.264')],
        '.avi': [('XVID', 'XVID'), ('MJPG', 'MJPEG')],
        '.h265': [('HEVC', 'HEVC'), ('avc1', 'H.264')],
        '.hevc': [('HEVC', 'HEVC'), ('avc1', 'H.264')],
        '.mkv': [('mp4v', 'mp4v'), ('avc1', 'H.264'), ('XVID', 'XVID')]
    }
    
    fourcc = None
    codec_name = ""
    
    # 根据扩展名获取编码器列表
    if ext in codec_priority:
        codecs = codec_priority[ext]
    else:
        logger.warning(f"不支持的视频格式 {ext}，使用默认MP4格式")
        ext = '.mp4'
        codecs = codec_priority[ext]
        output_path = os.path.splitext(output_path)[0] + '.mp4'
    
    # 测试可用的编码器
    for codec_code, codec_desc in codecs:
        test_fourcc = cv2.VideoWriter_fourcc(*codec_code)
        test_path = output_path + f'.test_{codec_code}'
        test_writer = cv2.VideoWriter(test_path, test_fourcc, fps, (width, height))
        
        if test_writer.isOpened():
            test_writer.release()
            fourcc = test_fourcc
            codec_name = codec_desc
            try:
                os.remove(test_path)
            except:
                pass
            logger.info(f"使用编码器: {codec_desc} ({codec_code})")
            break
        else:
            test_writer.release()
            try:
                os.remove(test_path)
            except:
                pass
    
    # 如果所有编码器都不可用，使用最通用的MP4V
    if fourcc is None:
        logger.warning("所有编码器都不可用，使用默认MP4V编码器")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.splitext(output_path)[0] + '.mp4'
        codec_name = "MP4V"
    
    # 创建视频写入器
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        logger.error(f"无法创建视频写入器: {output_path}")
        logger.error(f"编码器信息: fourcc={fourcc}, 尺寸={width}x{height}, fps={fps}")
        logger.error("可能的解决方案:")
        logger.error("1. 安装FFmpeg并添加到系统PATH")
        logger.error("2. 使用管理员权限运行程序")
        logger.error("3. 安装OpenCV的完整版本: pip install opencv-python-headless")
        logger.error("4. 使用MP4格式输出")
        
        # 尝试使用备用编码器
        backup_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        backup_path = os.path.splitext(output_path)[0] + '_backup.mp4'
        out = cv2.VideoWriter(backup_path, backup_fourcc, fps, (width, height))
        if out.isOpened():
            logger.info(f"使用备用编码器成功创建: {backup_path}")
            output_path = backup_path
        else:
            logger.error("备用编码器也创建失败")
            return False
    
    # 写入帧
    total_frames = len(frame_files)
    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(frame_file)
        if frame is None:
            logger.warning(f"跳过无效帧: {frame_file}")
            continue
            
        # 确保帧尺寸匹配
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
            
        out.write(frame)
        
        # 进度提示
        if i % 100 == 0:
            progress = (i / total_frames) * 100
            logger.debug(f"合成进度: {progress:.1f}% ({i}/{total_frames})")
    
    out.release()
    logger.info(f"视频合成完成: {output_path}")
    return True

def process_video_pipeline(input_video_path, output_video_path, face_detector, plate_detector, temp_dir="temp_processing", fps=60, batch_size=16):
    """完整的视频处理流程：视频 -> 图片 -> 批处理 -> 视频"""
    logger = logging.getLogger('VideoProcessor.pipeline')
    
    # 记录视频处理开始时间
    video_start_time = time.time()
    
    os.makedirs(temp_dir, exist_ok=True)
    frame_dir = os.path.join(temp_dir, "frames")
    processed_dir = os.path.join(temp_dir, "processed")
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    logger.info(f"开始处理视频: {os.path.basename(input_video_path)}")
    
    # 视频拆解为图片帧
    extract_start = time.time()
    frame_count = convert_video_to_frames(input_video_path, frame_dir)
    if frame_count == 0:
        logger.error("错误: 视频拆解失败")
        return False, 0, 0, 0
    extract_time = time.time() - extract_start
    
    # 批处理图片
    logger.info("步骤2: 批处理图片...")
    batch_start = time.time()
    processed_frames, total_faces_processed, total_plates_processed = batch_process_images(
        frame_dir, processed_dir, face_detector, plate_detector, batch_size
    )
    batch_time = time.time() - batch_start
    
    # 将处理后的图片合成为视频
    compile_start = time.time()
    success = create_video(processed_dir, output_video_path, fps)
    compile_time = time.time() - compile_start
    if not success:
        logger.error("错误: 视频合成失败")
        return False, processed_frames, total_faces_processed, total_plates_processed
    
    # 清理临时文件
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # 计算总耗时并输出
    total_video_time = time.time() - video_start_time
    logger.info(f"视频处理完成: {os.path.basename(input_video_path)} → {os.path.basename(output_video_path)}")
    logger.info(f"总耗时: {total_video_time:.1f}s (抽帧 {extract_time:.1f}s | 批处理 {batch_time:.1f}s | 合成 {compile_time:.1f}s)")
    logger.info(f"处理 {frame_count} 帧, 人脸 {total_faces_processed} 个, 车牌 {total_plates_processed} 个")
    
    return True, processed_frames, total_faces_processed, total_plates_processed

def process_single_video(video_path, output_videos_dir, face_detector, plate_detector, temp_base_dir, cleanup=True, batch_size=16):
    """处理单个视频的完整流程"""
    logger = logging.getLogger('VideoProcessor.process_single_video')
    video_filename = os.path.basename(video_path)
    video_name, video_ext = os.path.splitext(video_filename)
    
    # 获取视频原始格式
    original_format = video_ext.lstrip('.').lower() if video_ext else ''
    
    # 为每个视频创建唯一的输出文件名
    output_filename = f"{video_name}_processed.{original_format}"
    output_video_path = os.path.join(output_videos_dir, output_filename)
    
    # 为每个视频创建唯一的临时目录
    video_temp_dir = os.path.join(temp_base_dir, f"temp_{video_name}")
    
    if not original_format:
        logger.warning(f"无法确定视频格式: {video_filename}，将直接复制而不处理")
        return False
    
    try:
        # 使用处理管道
        success, processed_frames, faces_processed, plates_processed = process_video_pipeline(
            input_video_path=video_path,
            output_video_path=output_video_path,
            face_detector=face_detector,
            plate_detector=plate_detector,
            temp_dir=video_temp_dir,
            fps=60,
            batch_size=batch_size
        )
        
        if success:
            logger.info(f"视频处理成功: {video_filename} - "
                       f"{processed_frames} 帧, "
                       f"实际马赛克处理 {faces_processed} 人脸, {plates_processed} 车牌")
            return True
        else:
            logger.error(f"视频处理失败: {video_filename}")
            return False
            
    except Exception as e:
        logger.error(f"处理视频 {video_filename} 时出错: {str(e)}", exc_info=True)
        return False
    finally:
        # 清理临时文件
        if cleanup and os.path.exists(video_temp_dir):
            try:
                shutil.rmtree(video_temp_dir)
                logger.debug(f"已清理临时目录: {video_temp_dir}")
            except Exception as e:
                logger.warning(f"清理临时目录失败: {str(e)}")

def copy_unprocessed_video(video_path, output_dir):
    """复制未处理的视频到输出目录"""
    try:
        video_filename = os.path.basename(video_path)
        output_path = os.path.join(output_dir, video_filename)
        
        # 确保目标目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        shutil.copy2(video_path, output_path)
        logger.info(f"已复制未处理视频: {video_filename}")
        return True
    except Exception as e:
        logger.error(f"复制视频 {video_filename} 失败: {e}")
        return False

def load_config(config_file='config.ini'):
    """加载配置文件并返回解析结果"""
    config = configparser.ConfigParser()
    config.read(config_file)
    
    if 'PATHS' not in config:
        raise ValueError(f"配置文件中缺少 [PATHS] 部分: {config_file}")
    
    paths = config['PATHS']
    required_keys = [
        'model_path',
        'model_weights',
        'record_dir',
        'output_h265_dir',
        'output_videos_dir', 
        'temp_directory_base',
        'record_output_dir'
    ]
    
    missing = [key for key in required_keys if key not in paths]
    if missing:
        raise ValueError(f"配置文件中缺少必要的键: {', '.join(missing)}")
    
    # 获取视频格式设置
    if 'SETTINGS' in config:
        settings = config['SETTINGS']
        video_formats = settings.get('video_formats', 'h265,hevc,265,mp4,mov,avi')
        video_formats = [ext.strip() for ext in video_formats.split(',')]
        cleanup_temp = settings.getboolean('cleanup_temp', True)
        copy_unprocessed = settings.getboolean('copy_unprocessed_videos', True)
        batch_size = settings.getint('batch_size', 16)
    else:
        video_formats = ['h265', 'hevc', '265', 'mp4', 'mov', 'avi']
        cleanup_temp = True
        copy_unprocessed = True
        batch_size = 16
    
    return {
        'model_path': paths['model_path'],
        'model_weights': paths['model_weights'],
        'record_dir': paths['record_dir'],
        'output_h265_dir': paths['output_h265_dir'],
        'output_videos_dir': paths['output_videos_dir'],
        'temp_directory_base': paths['temp_directory_base'],
        'record_output_dir': paths['record_output_dir'],
        'video_formats': video_formats,
        'cleanup_temp': cleanup_temp,
        'copy_unprocessed': copy_unprocessed,
        'batch_size': batch_size
    }

def process_mf4(file_path, output_dir):
    """特殊处理 .mf4 文件"""
    logger = logging.getLogger('VideoProcessor.process_mf4')
    filename = os.path.basename(file_path)
    logger.info(f"处理 .mf4 文件: {filename}")
    
    # 这里添加MF4文件处理逻辑
    try:
        # 示例：仅复制文件
        output_path = os.path.join(output_dir, filename)
        shutil.copy2(file_path, output_path)
        logger.info(f".mf4 文件处理完成: {filename}")
        return True
    except Exception as e:
        logger.error(f".mf4 文件处理失败: {str(e)}")
        return False


if __name__ == "__main__":
    # 初始化日志器
    logger = setup_logger('video_processing.log')
    logger.info("===== 程序启动 =====")
    starttime = time.time()
    # 记录系统基本信息
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"工作目录: {os.getcwd()}")
    logger.info(f"OpenCV版本: {cv2.__version__}")
    logger.info(f"PyTorch版本: {torch.__version__}")
    
    try:
        # 加载配置
        logger.info("加载配置文件...")
        config = load_config()
        logger.info("配置文件加载成功")
        
        # 获取配置参数
        model_path = config['model_path']
        plate_model_path = config['model_weights']
        record_dir = config['record_dir']
        output_h265_dir = config['output_h265_dir']
        output_videos_dir = config['output_videos_dir']
        temp_directory_base = config['temp_directory_base']
        record_output_dir = config['record_output_dir']
        video_formats = config['video_formats']
        cleanup_temp = config['cleanup_temp']
        copy_unprocessed = config['copy_unprocessed']
        batch_size = config['batch_size']
        input_videos_dir = os.path.join(output_h265_dir, "hevcs")
        
        logger.info("配置参数:")
        logger.info(f"模型权重: {plate_model_path}")
        logger.info(f"retinaface模型路径: {model_path}")
        logger.info(f"record输入: {record_dir}")
        logger.info(f"视频输入目录: {input_videos_dir}")
        logger.info(f"视频输出目录: {output_videos_dir}")
        logger.info(f"临时目录: {temp_directory_base}")
        logger.info(f"record打包路径: {record_output_dir}")
        logger.info(f"支持格式: {', '.join(video_formats)}")
        logger.info(f"批处理大小: {batch_size}")
        
        # 检查可用编码器
        logger.info("检查系统可用视频编码器...")
        available_codecs = check_available_codecs()
        if available_codecs:
            logger.info("可用编码器: " + ", ".join([f"{name}({code})" for code, name in available_codecs]))
        else:
            logger.warning("未检测到可用编码器，请安装FFmpeg")
        
       # 解包record文件，获取摄像头数据
        logger.info("开始解包数据...")
        unpack_time= time.time()
        result1 = recordDeal.read_record2h265_all(record_dir, output_h265_dir)
        logger.info(f"解包完成,耗时: {time.time() - unpack_time:.2f}秒")
        
        # 初始化模型
        logger.info("开始初始化检测模型...")
        start_init_time = time.time()
        
        # 检查GPU可用性
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"检测到GPU: {gpu_count}个 - {gpu_name}")
        else:
            logger.info("未检测到GPU,将使用CPU")
        
        # 初始化MTCNN人脸检测模型
        logger.info("正在加载retinaface人脸检测模型...")
        logger.info(f"reitnaface模型路径: {model_path}")
        face_detector = Retinaface(
                                    model_path=model_path,
                                    backbone="resnet50",
                                    input_shape=[640, 640, 3],
                                    confidence=0.5,
                                    nms_iou=0.4,
                                    letterbox_image=True,
                                    cuda=True  # 自动使用 GPU
                                )
        device = face_detector.device
        print(f"face模型运行设备: {device}")
        logger.info("retinaface人脸检测模型加载完成")
        
        # 初始化YOLOv8车牌检测模型
        logger.info("正在加载YOLOv8车牌检测模型...")
        plate_detector = YOLO(plate_model_path).cuda()
        
        # 检查YOLOv8使用的设备
        if torch.cuda.is_available():
            yolo_device = next(plate_detector.model.parameters()).device
            logger.info(f"YOLOv8车牌检测模型运行在 GPU: {yolo_device}")
        else:
            logger.info("YOLOv8车牌检测模型运行在: CPU")
        
        init_time = time.time() - start_init_time
        logger.info(f"模型初始化完成，总耗时: {init_time:.2f}秒")
        
        # 确保目录存在
        os.makedirs(output_videos_dir, exist_ok=True)
        logger.info(f"输出目录已创建/确认: {output_videos_dir}")
        os.makedirs(temp_directory_base, exist_ok=True)
        logger.info(f"临时根目录已创建/确认: {temp_directory_base}")
        
        
        
        # 开始文件处理
        logger.info(f"在目录 {input_videos_dir} 中查找文件...")
        all_files = []
        if os.path.exists(input_videos_dir):
            for root, _, files in os.walk(input_videos_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)
                    logger.debug(f"找到文件: {file_path}")
        
        file_count = len(all_files)
        logger.info(f"共找到 {file_count} 个文件待处理")
        
        # 文件处理统计
        success_count = 0
        copy_count = 0
        skip_count = 0
        mf4_count = 0
        

        # 处理每个文件
        for i, file_path in enumerate(all_files, 1):
            filename = os.path.basename(file_path)
            _, file_ext = os.path.splitext(filename)
            file_ext = file_ext.lstrip('.').lower() if file_ext else ''
            
            logger.info(f"\n处理文件 ({i}/{file_count}): {filename}")
            
            # 根据文件类型处理
            if file_ext == 'mf4':
                logger.info("处理 .mf4 特殊格式")
                if process_mf4(file_path, output_videos_dir):
                    mf4_count += 1
                    logger.info(f".mf4 文件处理成功")
                else:
                    logger.error(".mf4 文件处理失败")
                    skip_count += 1
            elif file_ext in video_formats:
                logger.info(f"处理视频文件 (.{file_ext})")
                success = process_single_video(
                    file_path, output_videos_dir, 
                    face_detector, plate_detector, temp_directory_base, cleanup_temp, batch_size
                )
                if success:
                    success_count += 1
                    logger.info(f"视频处理成功")
                else:
                    logger.error(f"视频处理失败")
                    skip_count += 1
            elif copy_unprocessed:
                logger.info("复制非视频文件")
                if copy_unprocessed_video(file_path, output_videos_dir):
                    copy_count += 1
                    logger.info(f"文件复制成功")
                else:
                    logger.error(f"文件复制失败")
                    skip_count += 1
            else:
                logger.info(f"跳过不符合格式的文件")
                skip_count += 1
        model_end_time = time.time()
        total_model_time = model_end_time - start_init_time
        logger.info(f"\n模型运行总耗时: {total_model_time:.2f}秒")
        # record文件打包
        logger.info("开始重新打包record文件...")
        pack_time = time.time()
        result2 = recordDeal.write_allH265_record_all(record_dir, output_videos_dir, record_output_dir)
        logger.info(f"打包完成, 耗时: {time.time() - pack_time:.2f}秒")
        
        # 最终统计信息
        logger.info("\n===== 处理完成! 最终统计 =====")
        logger.info(f"总文件数: {file_count}")
        logger.info(f"成功处理视频: {success_count}")
        logger.info(f"处理特殊格式 (.mf4): {mf4_count}")
        logger.info(f"复制文件: {copy_count}")
        logger.info(f"跳过文件: {skip_count}")
        
        # 打开日志文件位置（仅Windows）
        if os.name == 'nt':
            log_dir = os.path.abspath(os.getcwd())
            logger.info(f"日志文件位于: {log_dir}/video_processing.log")
            try:
                os.startfile(log_dir)
            except:
                pass
        
        logger.info("程序正常结束")
        
    except Exception as e:
        import traceback
        print("="*50)
        print("发生未捕获的异常:")
        traceback.print_exc()
        print("="*50)
        if 'logger' in locals():
            logger.exception("程序发生致命错误:")
        sys.exit(1)
    endtime = time.time()
    logger.info(f"总耗时: {endtime - starttime:.1f}秒")

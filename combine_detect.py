from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
import os
import glob
import time
import configparser  # 新增导入模块
import shutil  # 确保导入shutil用于清理临时文件
from ultralytics import YOLO
from ultralytics.yolo.utils import ops
import torch
from utils import batch_convert_videos, convert_video_to_frames  # 导入视频转图片函数
#from picture2video import create_h265_video  # 导入图片转视频函数
from utils import create_video  # 导入图片转视频函数
import logging
import sys
from record_read_write import extract_camera_data,repack_record #record文件解包和打包

# 配置全局日志器
def setup_logger(log_file='video_processing.log'):
    # 创建 logger 实例
    logger = logging.getLogger('VideoProcessor')
    logger.setLevel(logging.DEBUG)
    return logger
    
class CombinedProcessor:
    def __init__(self, face_detector, plate_detector, output_dir):
#        self.face_detector = face_detector
#        self.plate_detector = plate_detector
#        self.output_dir = output_dir
#        os.makedirs(output_dir, exist_ok=True)
        self.face_detector = face_detector
        self.plate_detector = plate_detector
        self.output_dir = output_dir
        self.logger = logging.getLogger('VideoProcessor.CombinedProcessor')
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"输出目录已创建: {output_dir}")  

        
    def get_optimal_ellipse(self, face, img_shape):
        """计算刚好包含人脸的椭圆参数"""
        x, y, w, h = face['box']
        keypoints = face['keypoints']
        
        # 初始椭圆：基于矩形框的中心和轴长
        center = (x + w // 2, y + h // 2)
        axes = (w // 2, h // 2)
        
        # 获取所有关键点坐标
        points = np.array(list(keypoints.values()))
        
        # 动态调整椭圆轴长
        for _ in range(2):
            for (px, py) in points:
                dx = (px - center[0]) / axes[0]
                dy = (py - center[1]) / axes[1]
                distance = dx**2 + dy**2
                
                if distance > 0.9:
                    scale = min(1.1, 1 / (distance**0.5))
                    axes = (int(axes[0] * scale), int(axes[1] * scale))
        
        # 限制椭圆不超出图像边界
        axes = (
            min(axes[0], center[0], img_shape[1] - center[0]),
            min(axes[1], center[1], img_shape[0] - center[1])
        )
        
        return center, axes

    def mosaic_ellipse_region(self, img, center, axes):
        """对椭圆区域进行马赛克处理"""
        # 创建椭圆掩膜
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # 获取椭圆区域
        x, y = center[0] - axes[0], center[1] - axes[1]
        w, h = 2 * axes[0], 2 * axes[1]
        
        if w > 0 and h > 0:
            # 应用马赛克效果
            region = img[y:y+h, x:x+w]
            if region.size > 0:  # 确保区域有效
                small = cv2.resize(region, (max(1, w//8), max(1, h//8)), interpolation=cv2.INTER_LINEAR)
                mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                # 应用马赛克区域
                mask_roi = mask[y:y+h, x:x+w]
                img_roi = img[y:y+h, x:x+w]
                img_roi[mask_roi == 255] = mosaic[mask_roi == 255]
        
        return img

    def mosaic_rectangle_region(self, img, x1, y1, x2, y2, mosaic_level=8):
        """对矩形区域进行马赛克处理（用于车牌）"""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        
        if x2 > x1 and y2 > y1:
            area = img[y1:y2, x1:x2]
            small = cv2.resize(area, (max(1, (x2-x1)//mosaic_level), max(1, (y2-y1)//mosaic_level)), 
                              interpolation=cv2.INTER_NEAREST)
            mosaic = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
            img[y1:y2, x1:x2] = mosaic
        
        return img

    def detect_faces(self, img):
        """检测人脸并打码（已废弃，使用批处理方法）"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.face_detector.detect_faces(img_rgb)
        
        for face in faces:
            center, axes = self.get_optimal_ellipse(face, img.shape)
            img = self.mosaic_ellipse_region(img, center, axes)
        
        return img, len(faces), 0

    def detect_plates(self, img):
        """检测车牌并打码（已废弃，使用批处理方法）"""
        results = self.plate_detector(img)
        plates_count = 0
        
        for result in results:
            boxes = result.boxes
            plates_count += len(boxes)
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().squeeze().tolist())
                img = self.mosaic_rectangle_region(img, x1, y1, x2, y2)
        
        return img, plates_count, 0

    def process_image(self, img_path):
        """处理单张图片（已废弃，使用批处理方法）"""
        return self.process_images_batch([img_path])
    
    def process_images_batch(self, img_paths):
        """批量处理图片列表"""
        if not img_paths:
            return 0, 0, 0
            
        total_faces = 0
        total_plates = 0
        processed_count = 0
        
        # 批量读取图片
        images = []
        valid_paths = []
        
        for img_path in img_paths:
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                valid_paths.append(img_path)
            else:
                self.logger.error(f"无法读取图片: {img_path}")
        
        if not images:
            return 0, 0, 0
            
        # 批量处理人脸检测
        faces_results = []
        for img in images:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.face_detector.detect_faces(img_rgb)
            faces_results.append(faces)
            total_faces += len(faces)
        
        # 批量处理车牌检测
        plates_results = []
        for img in images:
            results = self.plate_detector(img)
            plates_count = 0
            for result in results:
                boxes = result.boxes
                plates_count += len(boxes)
            plates_results.append(plates_count)
            total_plates += plates_count
        
        # 批量应用马赛克并保存
        for i, (img, img_path, faces, plates_count) in enumerate(zip(images, valid_paths, faces_results, plates_results)):
            # 处理人脸
            for face in faces:
                center, axes = self.get_optimal_ellipse(face, img.shape)
                img = self.mosaic_ellipse_region(img, center, axes)
            
            # 处理车牌
            for result in self.plate_detector(img):
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().squeeze().tolist())
                    img = self.mosaic_rectangle_region(img, x1, y1, x2, y2)
            
            # 保存结果
            filename = os.path.basename(img_path)
            output_path = os.path.join(self.output_dir, filename)
            cv2.imwrite(output_path, img)
            processed_count += 1
        
        return processed_count, total_faces, total_plates

#def batch_process_images(input_dir, output_dir):
def batch_process_images(input_dir, output_dir, face_detector, plate_detector):
    """批量处理目录中的所有图片（使用预加载的模型，16张图片批处理）"""
    logger = logging.getLogger('VideoProcessor.batch_process_images')
    logger.info("批量处理图片开始...")
    
    # 直接使用预加载的模型
    processor = CombinedProcessor(face_detector, plate_detector, output_dir)
    
    # 获取图片
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, f'*.{ext}')))

    logger.info(f"找到 {len(image_paths)} 张图片待处理")
    if not image_paths:
        logger.info(f"在目录 '{input_dir}' 中未找到图片文件")
        return
    
    # 批量处理
    processed_count = 0
    total_faces = 0
    total_plates = 0
    start_time = time.time()
    
    # 按16张图片为一批进行处理
    batch_size = 16
    for i in range(0, len(image_paths), batch_size):
        batch_files = image_paths[i:i+batch_size]
        batch_processed, batch_faces, batch_plates = processor.process_images_batch(batch_files)
        
        processed_count += batch_processed
        total_faces += batch_faces
        total_plates += batch_plates
        logger.info(f"批次 {i//batch_size + 1} 处理完成: {batch_processed} 张图片")
    
    # 打印统计信息
    total_time = time.time() - start_time
    avg_time = total_time / len(image_paths) if image_paths else 0
    logger.info(f"\n处理完成! 共处理 {processed_count}/{len(image_paths)} 张图片")
    logger.info(f"人脸总数: {total_faces} | 车牌总数: {total_plates}")
    logger.info(f"总耗时: {total_time:.2f}秒 | 平均每张: {avg_time:.2f}秒")
    
    # 处理完成后打开输出目录（仅限Windows）
    if os.name == 'nt' and os.path.exists(output_dir):
        os.startfile(output_dir)

#def process_video_pipeline(input_video_path, output_video_path, temp_dir="temp_processing", fps=30):
def process_video_pipeline(input_video_path, output_video_path, face_detector, plate_detector, temp_dir="temp_processing", fps=60):
    """
    完整的视频处理流程：视频 -> 图片 -> 人脸车牌处理 -> 图片 -> 视频（使用16张批处理）
    :param input_video_path: 输入视频文件路径
    :param output_video_path: 输出视频文件路径
    :param face_detector: 预加载的人脸检测模型
    :param plate_detector: 预加载的车牌检测模型
    :param temp_dir: 临时处理目录
    :param fps: 视频帧率
    :return: 处理是否成功
    """

    logger = logging.getLogger('VideoProcessor.pipeline')
    logger.info(f"视频处理管道启动: {input_video_path} → {output_video_path}")
    
    start_time = time.time()
    
    # 创建临时目录
    frame_dir = os.path.join(temp_dir, "frames")
    processed_dir = os.path.join(temp_dir, "processed")
    for dir_path in [frame_dir, processed_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    logger.info(f"临时目录结构创建完成: {frame_dir}, {processed_dir}")
    
    # 视频转图片
    extract_start = time.time()
    frame_count = convert_video_to_frames(input_video_path, frame_dir)
    if frame_count == 0:
        logger.error("错误: 视频拆解失败")
        return False
    extract_time = time.time() - extract_start
    logger.info(f"视频转图片完成 | 拆解 {frame_count} 帧 | 耗时: {extract_time:.2f}秒")
    
    # 处理图片（使用16张批处理）
    process_start = time.time()
    batch_process_images(frame_dir, processed_dir, face_detector, plate_detector)
    process_time = time.time() - process_start
    logger.info(f"图片批量处理完成 | 耗时: {process_time:.2f}秒 | 平均每帧: {process_time/max(1, frame_count):.4f}秒")
    
    # 图片转视频
    compile_start = time.time()
    success = create_video(processed_dir, output_video_path, fps)
    compile_time = time.time() - compile_start
    if not success:
        logger.error("错误: 视频合成失败")
        return False
    logger.info(f"图片转视频完成 | 耗时: {compile_time:.2f}秒 | 输出: {output_video_path}")
    
    # 清理临时文件
    shutil.rmtree(temp_dir)
    
    total_time = time.time() - start_time
    logger.info(f"视频处理完成: {os.path.basename(input_video_path)} → {os.path.basename(output_video_path)} | "
                f"总耗时: {total_time:.2f}秒 | "
                f"抽帧: {extract_time:.2f}s | "
                f"处理: {process_time:.2f}s | "
                f"合成: {compile_time:.2f}s")
    
    return True

# 新函数：处理单个视频文件（使用预加载的模型，16张批处理）
def process_single_video(video_path, output_videos_dir, face_detector, plate_detector, temp_base_dir, cleanup=True):
    """
    处理单个视频的完整流程（使用预加载的模型，16张批处理优化）
    """
    logger = logging.getLogger('VideoProcessor.single_video')
    video_filename = os.path.basename(video_path)
    video_name, video_ext = os.path.splitext(video_filename)

    # 获取视频原始格式
    original_format = video_ext.lstrip('.').lower() if video_ext else ''
    logger.info(f"处理视频: {video_filename}")
    
    # 如果无法确定格式，跳过处理
    if not original_format:
        logger.warning(f"无法确定视频格式: {video_filename}，将直接复制而不处理")
        return False
    
    # 为每个视频创建唯一的输出文件名
    output_filename = f"{video_name}_processed.{original_format}"
    output_video_path = os.path.join(output_videos_dir, output_filename)
    
    # 为每个视频创建唯一的临时目录
    video_temp_dir = os.path.join(temp_base_dir, f"temp_{video_name}")
    
    logger.info(f"\n===== 开始处理视频: {video_filename} =====")
    logger.info(f"输入路径: {video_path}")
    logger.info(f"输出路径: {output_video_path}")
    logger.info(f"临时目录: {video_temp_dir}")
    
    start_time = time.time()
    success = process_video_pipeline(
        input_video_path=video_path,
        output_video_path=output_video_path,
        face_detector=face_detector,
        plate_detector=plate_detector,
        temp_dir=video_temp_dir,
        fps=60
    )
    
    if success:
        total_time = time.time() - start_time
        logger.info(f"视频处理成功! | 耗时: {total_time:.2f}秒")
        
        # 清理临时文件
        if cleanup and os.path.exists(video_temp_dir):
            logger.info(f"清理临时文件: {video_temp_dir}")
            shutil.rmtree(video_temp_dir)
    else:
        logger.error("视频处理失败")
    
    return success

# 不需要处理的视频直接复制到输出目录中
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
        logger.info(f"复制视频 {video_filename} 失败: {e}")
        return False

#新增加载配置文件函数
def load_config(config_file='config.ini'):
    """加载配置文件并返回解析结果"""
    config = configparser.ConfigParser()
    config.read(config_file)
    
    if 'PATHS' not in config:
        raise ValueError(f"配置文件中缺少 [PATHS] 部分: {config_file}")
    
    paths = config['PATHS']
    required_keys = [
        'model_weights', 
        'record_dir',       # 新增
        'input_videos_dir', 
        'output_videos_dir', 
        'temp_directory_base',
        'final_record'      # 新增
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
    else:
        video_formats = ['h265', 'hevc', '265', 'mp4', 'mov', 'avi']
        cleanup_temp = True
        copy_unprocessed = True
    
    return {
        'model_weights': paths['model_weights'],
        'record_dir': paths['record_dir'],           # 新增
        'input_videos_dir': paths['input_videos_dir'],
        'output_videos_dir': paths['output_videos_dir'],
        'temp_directory_base': paths['temp_directory_base'],
        'final_record': paths['final_record'],       # 新增
        'video_formats': video_formats,
        'cleanup_temp': cleanup_temp,
        'copy_unprocessed': copy_unprocessed
    }

# mf4格式的文件特殊处理
def process_mf4(file_path, output_dir):
    """
    特殊处理 .mf4 文件
    """
    filename = os.path.basename(file_path)
    logger.info(f"调用成功,处理 .mf4 文件: {filename}")

    return True

if __name__ == "__main__":
#    # 配置参数
#    input_video = "/home/24181214123/yolo/original_video/camera_front_narrow.h265"  # 输入视频文件
#    output_video = "/home/24181214123/yolo/output_video.h265"  # 输出视频文件
#    temp_directory = "/home/24181214123/yolo/temp_processing"  # 临时处理目录

    
#    all_files = []
#    for root, _, files in os.walk(input_videos_dir):
#        for file in files:
#            all_files.append(os.path.join(root, file))
#    
#    if not all_files:
#        logger.info("在指定目录中没有找到任何文件")
#        exit(0)
#    
#    logger.info(f"找到 {len(all_files)} 个文件")
#    
##     # 执行完整处理流程
##    start_time = time.time()
##    success = process_video_pipeline(input_video, output_video, temp_directory)
#
#    
##    if success:
##        total_time = time.time() - start_time
##        logger.info(f"\n处理完成! 输出视频: {output_video}")
##        logger.info(f"总耗时: {total_time:.2f}秒")
#
#    # 开始处理文件
#    total_start_time = time.time()
#    success_count = 0
#    copy_count = 0
#    skip_count = 0
#    mf4_count = 0
#    
#    for i, file_path in enumerate(all_files, 1):
#        filename = os.path.basename(file_path)
#        logger.info(f"\n=== 处理文件 ({i}/{len(all_files)}): {filename} ===")
#        
#        # 获取文件扩展名（小写，不含点）
#        _, file_ext = os.path.splitext(filename)
#        file_ext = file_ext.lstrip('.').lower() if file_ext else ''
#        
#        # 检查是否是要处理的视频文件
#        if file_ext == 'mf4':
#            # 特殊处理.mf4文件
#            if process_mf4(file_path, output_videos_dir):
#                mf4_count += 1
#                logger.info(f".mf4 文件处理成功: {filename}")
#            else:
#                logger.info(f".mf4 文件处理失败: {filename}")
#                skip_count += 1
#        elif file_ext in video_formats:
#            # 符合格式，进行马赛克处理
#            success = process_single_video(
#                video_path=file_path,
#                output_videos_dir=output_videos_dir,
#                plate_model_path=plate_model_path,
#                temp_base_dir=temp_directory_base,
#                cleanup=cleanup_temp
#            )
#            if success:
#                success_count += 1
#                logger.info(f"视频处理成功: {filename}")
#            else:
#                logger.info(f"视频处理失败: {filename}")
#                skip_count += 1
#        elif copy_unprocessed:
#            # 不符合格式但配置了复制
#            if copy_unprocessed_video(file_path, output_videos_dir):
#                copy_count += 1
#                logger.info(f"已复制未处理文件: {filename}")
#            else:
#                logger.info(f"复制文件失败: {filename}")
#                skip_count += 1
#        else:
#            # 不符合格式且未配置复制
#            logger.info(f"跳过不符合格式的文件: {filename}")
#            skip_count += 1
#
#    # 打印总体统计信息
#    total_time = time.time() - total_start_time
#    logger.info(f"\n===== 所有文件处理完成! =====")
#    logger.info(f"总文件数: {len(all_files)}")
#    logger.info(f"特殊处理 .mf4 文件数: {mf4_count}")
#    logger.info(f"成功处理视频数: {success_count}")
#    logger.info(f"复制未处理文件数: {copy_count}")
#    logger.info(f"跳过文件数: {skip_count}")
#    logger.info(f"总耗时: {total_time:.2f}秒 | 平均每个文件: {total_time / max(1, len(all_files)):.2f}秒")
#
#        
#    # 处理完成后打开输出目录（仅限Windows）
#    if os.name == 'nt' and os.path.exists(output_video):
#        output_dir = os.path.dirname(output_video) or '.'
#        os.startfile(output_dir)
#    else:
#        logger.info("视频处理失败")

     # 初始化日志器 - 这是最重要的修改
    logger = setup_logger('video_processing.log')
    logger.info("===== 程序启动 =====")
    
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
        plate_model_path = config['model_weights']
        record_dir = config['record_dir']   #新增record文件路径
        input_videos_dir = config['input_videos_dir']
        output_videos_dir = config['output_videos_dir']
        temp_directory_base = config['temp_directory_base']
        final_record = config['final_record']  #新增打包路径
        video_formats = config['video_formats']
        cleanup_temp = config['cleanup_temp']
        copy_unprocessed = config['copy_unprocessed']
        
        logger.info("配置参数:")
        logger.info(f"模型权重: {plate_model_path}")
        logger.info(f"record输入: {record_dir}")
        logger.info(f"输入目录: {input_videos_dir}")
        logger.info(f"输出目录: {output_videos_dir}")
        logger.info(f"临时目录: {temp_directory_base}")
        logger.info(f"record打包: {final_record}")
        logger.info(f"支持格式: {', '.join(video_formats)}")
        
        # 在主函数中初始化模型，避免重复加载
        logger.info("开始初始化检测模型...")
        start_init_time = time.time()
        
        # 初始化MTCNN人脸检测模型
        logger.info("正在加载MTCNN人脸检测模型...")
        face_detector = MTCNN()
        logger.info("MTCNN人脸检测模型加载完成")
        
        # 初始化YOLOv8车牌检测模型
        logger.info("正在加载YOLOv8车牌检测模型...")
        plate_detector = YOLO(plate_model_path)
        logger.info("YOLOv8车牌检测模型加载完成")
        
        init_time = time.time() - start_init_time
        logger.info(f"模型初始化完成，总耗时: {init_time:.2f}秒")
        
        # 确保目录存在
        os.makedirs(output_videos_dir, exist_ok=True)
        logger.info(f"输出目录已创建/确认: {output_videos_dir}")
        os.makedirs(temp_directory_base, exist_ok=True)
        logger.info(f"临时根目录已创建/确认: {temp_directory_base}")
        
        #解包record文件，获取摄像头数据
        logging.info("开始解包数据...")
        camera_count, timestamps = extract_camera_data(record_dir, input_videos_dir)
        logging.info(f"解包完成: {camera_count} 个摄像头通道")

        
        # 开始文件处理
        logger.info(f"在目录 {input_videos_dir} 中查找文件...")
        all_files = []
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
                    face_detector, plate_detector, temp_directory_base, cleanup_temp
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

        #record文件打包
        logging.info("开始重新打包record文件...")
        repack_record(
            original_record=record_dir,
            blurred_dir=output_videos_dir,
            hevc_dir=input_videos_dir,
            output_record=final_record
            )
        logging.info(f"打包完成: {final_record}")
        
        # 最终统计信息
        logger.info("\n===== 处理完成! 最终统计 =====")
        logger.info(f"总文件数: {file_count}")
        logger.info(f"成功处理视频: {success_count}")
        logger.info(f"处理特殊格式 (.mf4): {mf4_count}")
        logger.info(f"复制文件: {copy_count}")
        logger.info(f"跳过文件: {skip_count}")
        
        # 打开日志文件位置
        if os.name == 'nt':
            log_dir = os.path.abspath(os.getcwd())
            logger.info(f"日志文件位于: {log_dir}/video_processing.log")
            os.startfile(log_dir)
        
        logger.info("程序正常结束")
        
    except Exception as e:
        import traceback
        print("="*50)
        print("发生未捕获的异常:")
        traceback.print_exc()
        print("="*50)
        logger.exception("程序发生致命错误:")  # 确保logger已初始化
        sys.exit(1)

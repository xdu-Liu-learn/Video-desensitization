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
        """检测人脸并应用马赛克"""
        self.logger.debug("开始检测人脸...")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#        faces = self.face_detector.detect_faces(img_rgb)
        
        start_time = time.time()
        faces = self.face_detector.detect_faces(img_rgb)
        face_detection_time = time.time() - start_time
        
        faces_count = len(faces)
        self.logger.info(f"检测到 {faces_count} 张人脸 | 耗时: {face_detection_time:.4f}s")
        
        for face in faces:
            center, axes = self.get_optimal_ellipse(face, img.shape)
            self.logger.debug(f"应用椭圆马赛克 - 位置: {center}, 轴长: {axes}")
            img = self.mosaic_ellipse_region(img, center, axes)
        
        return img, len(faces), face_detection_time

    def detect_plates(self, img):
        """检测车牌并应用马赛克"""
        # 使用YOLOv8检测车牌
        self.logger.debug("开始检测车牌...")
        start_time = time.time()
        results = self.plate_detector(img)
        car_detection_time = time.time() - start_time

        
        plates_count = 0
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().squeeze().tolist())
                img = self.mosaic_rectangle_region(img, x1, y1, x2, y2)
                plates_count += 1
                self.logger.debug(f"检测到车牌 #{plates_count} - 坐标: ({x1},{y1})-({x2},{y2})")

        self.logger.info(f"检测到 {plates_count} 个车牌 | 耗时: {car_detection_time:.4f}s")
        return img, plates_count, car_detection_time

    def process_image(self, img_path):
        """处理单张图片"""
        self.logger.debug(f"开始处理图片: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
#            logger.info(f"无法读取图片: {img_path}")
            self.logger.error(f"无法读取图片: {img_path}")
            return False
        
        original_img = img.copy()
        
# 处理人脸（接收耗时变量）
        img, faces_count, face_time = self.detect_faces(img)  # 添加第三个返回值
    
    # 处理车牌（接收耗时变量）
        img, plates_count, plate_time = self.detect_plates(img)  # 添加第三个返回值
        
        # 保存结果
        filename = os.path.basename(img_path)
        output_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(output_path, img)
        
        self.logger.info(f"已处理: {filename} | 人脸: {faces_count} | 车牌: {plates_count}")
        self.logger.info(f"图片处理完成 | 文件名: {filename} | 人脸: {faces_count} | 车牌: {plates_count} | 总耗时: {face_time+plate_time:.4f}s")
        return True

#def batch_process_images(input_dir, output_dir):
def batch_process_images(input_dir, output_dir, plate_model_path):  # 修改函数参数
    """批量处理目录中的所有图片"""
    logger = logging.getLogger('VideoProcessor.batch_process_images')
    logger.info("批量处理图片开始...")
    
    # 初始化模型
#    logger.info("初始化模型...")
#    face_detector = MTCNN()
##    plate_detector = YOLO('/home/24181214123/yolo/best.pt')  # 替换为你的模型路径
#    plate_detector = YOLO(plate_model_path)  # 使用传入的模型路径

    logger.info("初始化人脸检测模型 (MTCNN)...")
    try:
        face_detector = MTCNN()
        logger.info("人脸检测模型初始化成功")
    except Exception as e:
        logger.exception("人脸检测模型初始化失败")
        return
    
    logger.info(f"初始化车牌检测模型 (YOLO): {plate_model_path}")
    try:
        plate_detector = YOLO(plate_model_path)
        logger.info("车牌检测模型初始化成功")
    except Exception as e:
        logger.exception("车牌检测模型初始化失败")
        return 
    
    # 创建处理器实例
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
    start_time = time.time()
    
    for i, img_path in enumerate(image_paths, 1):
        logger.info(f"\n处理图片 ({i}/{len(image_paths)}): {img_path}")
        if processor.process_image(img_path):
            processed_count += 1
    
    # 打印统计信息
    total_time = time.time() - start_time
    avg_time = total_time / len(image_paths) if image_paths else 0
    logger.info(f"\n处理完成! 共处理 {processed_count}/{len(image_paths)} 张图片")
    logger.info(f"总耗时: {total_time:.2f}秒 | 平均每张: {avg_time:.2f}秒")
    
    # 处理完成后打开输出目录（仅限Windows）
    if os.name == 'nt' and os.path.exists(output_dir):
        os.startfile(output_dir)

#def process_video_pipeline(input_video_path, output_video_path, temp_dir="temp_processing", fps=30):
def process_video_pipeline(input_video_path, output_video_path, plate_model_path, temp_dir="temp_processing", fps=60):  # 修改函数参数
    """
    完整的视频处理流程：视频 -> 图片 -> 人脸车牌处理 -> 图片 -> 视频
    :param input_video_path: 输入视频文件路径
    :param output_video_path: 输出视频文件路径
    :param plate_model_path: YOLO模型路径
    :param temp_dir: 临时处理目录
    :param fps: 视频帧率
    :return: 处理是否成功
    """

#    os.makedirs(temp_dir, exist_ok=True)
#    # 创建临时目录结构
#    frame_dir = os.path.join(temp_dir, "frames")
#    processed_dir = os.path.join(temp_dir, "processed")
#    os.makedirs(frame_dir, exist_ok=True)
#    os.makedirs(processed_dir, exist_ok=True)

    logger = logging.getLogger('VideoProcessor.pipeline')
    logger.info(f"视频处理管道启动: {input_video_path} → {output_video_path}")
    
    os.makedirs(temp_dir, exist_ok=True)
    frame_dir = os.path.join(temp_dir, "frames")
    processed_dir = os.path.join(temp_dir, "processed")
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    logger.info(f"临时目录结构创建完成: {frame_dir}, {processed_dir}")
    
    logger.info(f"步骤 1/4: 视频拆解为图片帧 ({input_video_path})")
    # 转换视频为图片帧
    frame_count = convert_video_to_frames(input_video_path, frame_dir)
    if frame_count == 0:
        logger.error("错误: 视频拆解失败")
        return False
    logger.info(f"成功拆解 {frame_count} 帧图片到 {frame_dir}")
    
    logger.info(f"步骤 2/4: 处理图片中的人脸和车牌")
    # 处理图片帧（人脸和车牌打码）
#    batch_process_images(frame_dir, processed_dir)
    start_time = time.time()
    batch_process_images(frame_dir, processed_dir, plate_model_path)  # 传递模型路径
    process_time = time.time() - start_time
    logger.info(f"图片处理完成 | 耗时: {process_time:.2f}秒 | 平均每帧: {process_time/max(1, frame_count):.4f}秒")
    
    logger.info(f"步骤 3/4: 将处理后的图片合成为视频")
    # 创建处理后的视频
#    success = create_h265_video(processed_dir, output_video_path, fps)
    # 替换原来的 create_h265_video 函数为新的通用 create_video 函数
    start_time = time.time()
    success = create_video(processed_dir, output_video_path, fps)
    video_time = time.time() - start_time
    if not success:
        logger.info("错误: 视频合成失败")
        return False
    logger.info(f"视频合成完成 | 耗时: {video_time:.2f}秒 | 输出: {output_video_path}")
    
    logger.info(f"步骤 4/4: 清理临时文件")
    # 清理临时文件（可选）
    if os.path.exists(temp_dir):
        logger.info(f"步骤 4/4: 清理临时文件 ({temp_dir})")
        shutil.rmtree(temp_dir)
        logger.info("临时文件清理完成")
    
    logger.info("视频处理管道完成!")
    
    return True

# 新函数：处理单个视频文件
def process_single_video(video_path, output_videos_dir, plate_model_path, temp_base_dir, cleanup=True):
    """
    处理单个视频的完整流程
    """
    logger = logging.getLogger('VideoProcessor.single_video')
    video_filename = os.path.basename(video_path)
    video_name, video_ext = os.path.splitext(video_filename)

    # 获取视频原始格式
    original_format = video_ext.lstrip('.').lower() if video_ext else ''
    logger.info(f"处理视频: {video_filename}")
    # 如果无法确定格式，跳过处理
    if not original_format:
        logger.info(f"警告: 无法确定视频格式: {video_filename}，将直接复制而不处理")
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
        plate_model_path=plate_model_path,
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
        logger.info("视频处理失败")
    
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
                    plate_model_path, temp_directory_base, cleanup_temp
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

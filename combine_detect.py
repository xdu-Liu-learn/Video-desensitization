from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
import os
import glob
import time
from ultralytics import YOLO
from ultralytics.yolo.utils import ops
import torch
from video2picture import batch_convert_videos, convert_video_to_frames  # 导入视频转图片函数
from picture2video import create_h265_video  # 导入图片转视频函数

class CombinedProcessor:
    def __init__(self, face_detector, plate_detector, output_dir):
        self.face_detector = face_detector
        self.plate_detector = plate_detector
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
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
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.face_detector.detect_faces(img_rgb)
        
        for face in faces:
            center, axes = self.get_optimal_ellipse(face, img.shape)
            img = self.mosaic_ellipse_region(img, center, axes)
        
        return img, len(faces)

    def detect_plates(self, img):
        """检测车牌并应用马赛克"""
        # 使用YOLOv8检测车牌
        results = self.plate_detector(img)
        
        plates_count = 0
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().squeeze().tolist())
                img = self.mosaic_rectangle_region(img, x1, y1, x2, y2)
                plates_count += 1
        
        return img, plates_count

    def process_image(self, img_path):
        """处理单张图片"""
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            return False
        
        original_img = img.copy()
        
        # 处理人脸
        img, faces_count = self.detect_faces(img)
        
        # 处理车牌
        img, plates_count = self.detect_plates(img)
        
        # 保存结果
        filename = os.path.basename(img_path)
        output_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(output_path, img)
        
        print(f"已处理: {filename} | 人脸: {faces_count} | 车牌: {plates_count}")
        return True

def batch_process_images(input_dir, output_dir):
    """批量处理目录中的所有图片"""
    # 初始化模型
    print("初始化模型...")
    face_detector = MTCNN()
    plate_detector = YOLO('/home/24181214123/yolo/best.pt')  # 替换为你的模型路径
    
    # 创建处理器实例
    processor = CombinedProcessor(face_detector, plate_detector, output_dir)
    
    # 获取图片
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, f'*.{ext}')))
    
    if not image_paths:
        print(f"在目录 '{input_dir}' 中未找到图片文件")
        return
    
    print(f"找到 {len(image_paths)} 张图片待处理")
    
    # 批量处理
    processed_count = 0
    start_time = time.time()
    
    for i, img_path in enumerate(image_paths, 1):
        print(f"\n处理图片 ({i}/{len(image_paths)}): {img_path}")
        if processor.process_image(img_path):
            processed_count += 1
    
    # 打印统计信息
    total_time = time.time() - start_time
    avg_time = total_time / len(image_paths) if image_paths else 0
    print(f"\n处理完成! 共处理 {processed_count}/{len(image_paths)} 张图片")
    print(f"总耗时: {total_time:.2f}秒 | 平均每张: {avg_time:.2f}秒")
    
    # 处理完成后打开输出目录（仅限Windows）
    if os.name == 'nt' and os.path.exists(output_dir):
        os.startfile(output_dir)

def process_video_pipeline(input_video_path, output_video_path, temp_dir="temp_processing", fps=30):
    """
    完整的视频处理流程：视频 -> 图片 -> 人脸车牌处理 -> 图片 -> 视频
    :param input_video_path: 输入视频文件路径
    :param output_video_path: 输出视频文件路径
    :param temp_dir: 临时处理目录
    :param fps: 视频帧率
    :return: 处理是否成功
    """
    # 创建临时目录结构
    frame_dir = os.path.join(temp_dir, "frames")
    processed_dir = os.path.join(temp_dir, "processed")
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"步骤 1/4: 视频拆解为图片帧 ({input_video_path})")
    # 转换视频为图片帧
    frame_count = convert_video_to_frames(input_video_path, frame_dir)
    if frame_count == 0:
        print("错误: 视频拆解失败")
        return False
    print(f"成功拆解 {frame_count} 帧图片到 {frame_dir}")
    
    print(f"步骤 2/4: 处理图片中的人脸和车牌")
    # 处理图片帧（人脸和车牌打码）
    batch_process_images(frame_dir, processed_dir)
    print(f"图片处理完成，结果保存在 {processed_dir}")
    
    print(f"步骤 3/4: 将处理后的图片合成为视频")
    # 创建处理后的视频
    success = create_h265_video(processed_dir, output_video_path, fps)
    if not success:
        print("错误: 视频合成失败")
        return False
    
    print(f"步骤 4/4: 清理临时文件")
    # 清理临时文件（可选）
    shutil.rmtree(temp_dir)
    print(f"临时文件 {temp_dir} 已清理")
    
    return True


if __name__ == "__main__":
    # 配置参数
    input_video = "/home/24181214123/yolo/original_video/camera_front_narrow.h265"  # 输入视频文件
    output_video = "/home/24181214123/yolo/output_video.h265"  # 输出视频文件
    temp_directory = "/home/24181214123/yolo/temp_processing"  # 临时处理目录
    
     # 执行完整处理流程
    start_time = time.time()
    success = process_video_pipeline(input_video, output_video, temp_directory)
    
    if success:
        total_time = time.time() - start_time
        print(f"\n处理完成! 输出视频: {output_video}")
        print(f"总耗时: {total_time:.2f}秒")
        
        # 处理完成后打开输出目录（仅限Windows）
        if os.name == 'nt' and os.path.exists(output_video):
            output_dir = os.path.dirname(output_video) or '.'
            os.startfile(output_dir)
    else:
        print("视频处理失败")
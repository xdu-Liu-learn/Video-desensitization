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
from skimage import transform as trans
from dataloader import VideoFrameDataset

class MTCNN:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 获取P模型
        self.pnet = torch.jit.load(os.path.join(model_path, 'PNet.pth'))
        self.pnet.to(self.device)
        self.softmax_p = torch.nn.Softmax(dim=0)
        self.pnet.eval()

        # 获取R模型
        self.rnet = torch.jit.load(os.path.join(model_path, 'RNet.pth'))
        self.rnet.to(self.device)
        self.softmax_r = torch.nn.Softmax(dim=-1)
        self.rnet.eval()

        # 获取O模型
        self.onet = torch.jit.load(os.path.join(model_path, 'ONet.pth'))
        self.onet.to(self.device)
        self.softmax_o = torch.nn.Softmax(dim=-1)
        self.onet.eval()

    # 使用PNet模型预测
    def predict_pnet(self,infer_data):
        infer_data = torch.tensor(infer_data, dtype=torch.float32, device=self.device)
        infer_data = torch.unsqueeze(infer_data, dim=0)
        cls_prob, bbox_pred, _ = self.pnet(infer_data)
        cls_prob = torch.squeeze(cls_prob)
        cls_prob = self.softmax_p(cls_prob)
        bbox_pred = torch.squeeze(bbox_pred)
        return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()

    # 使用RNet模型预测
    def predict_rnet(self,infer_data):
        infer_data = torch.tensor(infer_data, dtype=torch.float32, device=self.device)
        cls_prob, bbox_pred, _ = self.rnet(infer_data)
        cls_prob = self.softmax_r(cls_prob)
        return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()

    # 使用ONet模型预测
    def predict_onet(self,infer_data):
        infer_data = torch.tensor(infer_data, dtype=torch.float32, device=self.device)
        cls_prob, bbox_pred, landmark_pred = self.onet(infer_data)
        cls_prob = self.softmax_o(cls_prob)
        return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy(), landmark_pred.detach().cpu().numpy()

    # 获取PNet网络输出结果
    def detect_pnet(self,im, min_face_size, scale_factor, thresh):
        net_size = 12
        current_scale = float(net_size) / min_face_size
        im_resized = self.processed_image(im, current_scale)
        _, current_height, current_width = im_resized.shape
        all_boxes = list()
        
        while min(current_height, current_width) > net_size:
            cls_cls_map, reg = self.predict_pnet(im_resized)
            boxes = self.generate_bbox(cls_cls_map[1, :, :], reg, current_scale, thresh)
            current_scale *= scale_factor
            im_resized = self.processed_image(im, current_scale)
            _, current_height, current_width = im_resized.shape

            if boxes.size == 0:
                continue
            
            keep = self.py_nms(boxes[:, :5], 0.5, mode='Union')
            boxes = boxes[keep]
            all_boxes.append(boxes)
        
        if len(all_boxes) == 0:
            return None
        
        all_boxes = np.vstack(all_boxes)
        keep = self.py_nms(all_boxes[:, 0:5], 0.7, mode='Union')
        all_boxes = all_boxes[keep]
        
        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
        
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T

        return boxes_c

    # 获取RNet网络输出结果
    def detect_rnet(self,im, dets, thresh):
        h, w, c = im.shape
        dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        delete_size = np.ones_like(tmpw) * 20
        ones = np.ones_like(tmpw)
        zeros = np.zeros_like(tmpw)
        num_boxes = np.sum(np.where((np.minimum(tmpw, tmph) >= delete_size), ones, zeros))
        cropped_ims = np.zeros((int(num_boxes), 3, 24, 24), dtype=np.float32)
        
        for i in range(int(num_boxes)):
            if tmph[i] < 20 or tmpw[i] < 20:
                continue
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            try:
                tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
                img = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_LINEAR)
                img = img.transpose((2, 0, 1))
                img = (img - 127.5) / 128
                cropped_ims[i, :, :, :] = img
            except:
                continue
        
        cls_scores, reg = self.predict_rnet(cropped_ims)
        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > thresh)[0]
        
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
        else:
            return None

        keep = self.py_nms(boxes, 0.4, mode='Union')
        boxes = boxes[keep]
        boxes_c = self.calibrate_box(boxes, reg[keep])
        return boxes_c

    # 获取ONet模型预测结果
    def detect_onet(self,im, dets, thresh):
        h, w, c = im.shape
        dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 3, 48, 48), dtype=np.float32)
        
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            img = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_LINEAR)
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 128
            cropped_ims[i, :, :, :] = img
        
        cls_scores, reg, landmark = self.predict_onet(cropped_ims)
        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > thresh)[0]
        
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None

        w = boxes[:, 2] - boxes[:, 0] + 1
        h = boxes[:, 3] - boxes[:, 1] + 1
        
        landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
        landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
        boxes_c = self.calibrate_box(boxes, reg)

        keep = self.py_nms(boxes_c, 0.6, mode='Minimum')
        boxes_c = boxes_c[keep]
        landmark = landmark[keep]
        return boxes_c, landmark

    def infer_image_path(self, image_path):
        im = cv2.imread(image_path)
        boxes_c = self.detect_pnet(im, 20, 0.79, 0.9)
        if boxes_c is None:
            return None, None
        
        boxes_c = self.detect_rnet(im, boxes_c, 0.6)
        if boxes_c is None:
            return None, None
        
        boxes_c, landmark = self.detect_onet(im, boxes_c, 0.7)
        if boxes_c is None:
            return None, None

        return boxes_c, landmark

    # 对齐
    @staticmethod
    def estimate_norm(lmk):
        assert lmk.shape == (5, 2)
        tform = trans.SimilarityTransform()
        src = np.array([[38.2946, 51.6963],
                        [73.5318, 51.5014],
                        [56.0252, 71.7366],
                        [41.5493, 92.3655],
                        [70.7299, 92.2041]], dtype=np.float32)
        tform.estimate(lmk, src)
        M = tform.params[0:2, :]
        return M

    def norm_crop(self, img, landmark, image_size=112):
        M = self.estimate_norm(landmark)
        warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
        return warped

    def infer_image(self, im):
        if isinstance(im, str):
            im = cv2.imread(im)
            
        boxes_c = self.detect_pnet(im, 20, 0.79, 0.95)  # 提高PNet阈值
        if boxes_c is None:
            return None, None
        
        boxes_c = self.detect_rnet(im, boxes_c, 0.8)  # 提高RNet阈值
        if boxes_c is None:
            return None, None
        
        boxes_c, landmarks = self.detect_onet(im, boxes_c, 0.9)  # 提高ONet阈值
        if boxes_c is None:
            return None, None
            
        faces = []
        for i, box in enumerate(boxes_c):
            x1, y1, x2, y2, score = box.astype(int)
            face = {
                'box': [x1, y1, x2-x1, y2-y1],
                'confidence': score,
                'keypoints': {}
            }
            
            # 处理关键点
            if landmarks is not None and i < len(landmarks):
                landmark = landmarks[i]
                face['keypoints'] = {
                    'left_eye': (landmark[0], landmark[1]),
                    'right_eye': (landmark[2], landmark[3]),
                    'nose': (landmark[4], landmark[5]),
                    'mouth_left': (landmark[6], landmark[7]),
                    'mouth_right': (landmark[8], landmark[9])
                }
                
            faces.append(face)

        return faces, boxes_c

    # 工具函数
    @staticmethod
    def generate_bbox(cls_map, reg, scale, thresh):
        stride = 2
        cellsize = 12

        cls_map = np.transpose(cls_map)
        dx1 = np.transpose(reg[0, :, :])
        dy1 = np.transpose(reg[1, :, :])
        dx2 = np.transpose(reg[2, :, :])
        dy2 = np.transpose(reg[3, :, :])

        (y, x) = np.where(cls_map >= thresh)
        if y.size == 0:
            return np.array([])
        
        score = np.array([cls_map[i, j] for i, j in zip(y, x)])
        regx1 = np.array([dx1[i, j] for i, j in zip(y, x)])
        regy1 = np.array([dy1[i, j] for i, j in zip(y, x)])
        regx2 = np.array([dx2[i, j] for i, j in zip(y, x)])
        regy2 = np.array([dy2[i, j] for i, j in zip(y, x)])
        
        x1 = np.round((x * stride + 1) / scale)
        y1 = np.round((y * stride + 1) / scale)
        x2 = np.round((x * stride + 1 + cellsize - 1) / scale)
        y2 = np.round((y * stride + 1 + cellsize - 1) / scale)
        
        bbox = np.vstack([x1, y1, x2, y2, score, regx1, regy1, regx2, regy2])
        return bbox.T

    @staticmethod
    def py_nms(dets, thresh, mode='Union'):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            if mode == 'Union':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            else:
                ovr = inter / np.minimum(areas[i], areas[order[1:]])

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    @staticmethod
    def convert_to_square(bbox):
        square_bbox = bbox.copy()
        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        max_side = np.maximum(h, w)
        
        square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
        square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
        square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
        square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
        
        return square_bbox

    @staticmethod
    def pad(bboxes, w, h):
        tmpw = (bboxes[:, 2] - bboxes[:, 0] + 1).astype(np.int32)
        tmph = (bboxes[:, 3] - bboxes[:, 1] + 1).astype(np.int32)
        num_boxes = bboxes.shape[0]

        dx = np.zeros((num_boxes,))
        dy = np.zeros((num_boxes,))
        edx = tmpw - 1
        edy = tmph - 1

        x = bboxes[:, 0].astype(np.int32)
        y = bboxes[:, 1].astype(np.int32)
        x2 = bboxes[:, 2].astype(np.int32)
        y2 = bboxes[:, 3].astype(np.int32)

        # 处理边界
        for i in range(num_boxes):
            if x[i] < 0:
                dx[i] = -x[i]
                x[i] = 0
            if y[i] < 0:
                dy[i] = -y[i]
                y[i] = 0
            if x2[i] >= w:
                edx[i] = tmpw[i] - 1 - (x2[i] - w + 1)
                x2[i] = w - 1
            if y2[i] >= h:
                edy[i] = tmph[i] - 1 - (y2[i] - h + 1)
                y2[i] = h - 1

        return dy, edy, dx, edx, y, y2, x, x2, tmpw, tmph

    @staticmethod
    def calibrate_box(bbox, reg):
        w = bbox[:, 2] - bbox[:, 0] + 1
        h = bbox[:, 3] - bbox[:, 1] + 1
        cx = bbox[:, 0] + w * 0.5
        cy = bbox[:, 1] + h * 0.5

        cx1 = cx + reg[:, 0] * w
        cy1 = cy + reg[:, 1] * h
        cx2 = cx + reg[:, 2] * w
        cy2 = cy + reg[:, 3] * h

        bbox[:, 0] = cx1 - w * 0.5
        bbox[:, 1] = cy1 - h * 0.5
        bbox[:, 2] = cx2 - w * 0.5 + w - 1
        bbox[:, 3] = cy2 - h * 0.5 + h - 1
        
        return bbox

    @staticmethod
    def processed_image(img, scale):
        h, w, c = img.shape
        new_h = int(h * scale)
        new_w = int(w * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img_resized = img_resized.transpose((2, 0, 1))
        img_resized = (img_resized - 127.5) / 128
        return img_resized

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
    
class CombinedProcessor:
    def __init__(self, face_detector, plate_detector, output_dir, debug_mode=False):
        self.face_detector = face_detector
        self.plate_detector = plate_detector
        self.output_dir = output_dir
        self.debug_mode = debug_mode
        self.logger = logging.getLogger('VideoProcessor.CombinedProcessor')
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"输出目录已创建: {output_dir}")
        
        # 检查GPU使用情况
        self.check_gpu_usage()
        
        # 将模型迁移到GPU（如果可用）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.logger.info(f"将模型迁移到GPU设备: {self.device}")
        
    def check_gpu_usage(self):
        """检查模型是否使用GPU运行"""
        # 检查YOLOv8是否使用GPU
        if torch.cuda.is_available():
            yolo_device = next(self.plate_detector.model.parameters()).device
            self.logger.info(f"YOLOv8车牌检测模型运行在: {yolo_device}")
        else:
            self.logger.info("YOLOv8车牌检测模型运行在: CPU")
        
        # 检查MTCNN设备
        mtcnn_device = self.face_detector.device
        self.logger.info(f"MTCNN人脸检测模型运行在: {mtcnn_device}")
        
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
                dx = (px - center[0]) / axes[0] if axes[0] > 0 else 0
                dy = (py - center[1]) / axes[1] if axes[1] > 0 else 0
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
        x, y = max(0, center[0] - axes[0]), max(0, center[1] - axes[1])
        w, h = min(2 * axes[0], img.shape[1] - x), min(2 * axes[1], img.shape[0] - y)
        
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
        start_time = time.time()
        faces, _ = self.face_detector.infer_image(img)
        face_detection_time = time.time() - start_time
        
        faces_count = len(faces) if faces is not None else 0
        
        if faces is not None and faces_count > 0:
            for face in faces:
                center, axes = self.get_optimal_ellipse(face, img.shape)
                img = self.mosaic_ellipse_region(img, center, axes)
        
        return img, faces_count, face_detection_time

    def detect_plates(self, img):
        """检测车牌并应用马赛克"""
        start_time = time.time()
        results = self.plate_detector(img, device=self.device, verbose=False, conf=0.5)  # 设置置信度阈值为0.5
        car_detection_time = time.time() - start_time

        plates_count = 0
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf.cpu().numpy().squeeze())
                if confidence >= 0.5:  # 只处理置信度>=0.5的检测结果
                    x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().squeeze().tolist())
                    img = self.mosaic_rectangle_region(img, x1, y1, x2, y2)
                    plates_count += 1

        return img, plates_count, car_detection_time

    def process_images_batch_dataloader(self, batch_images, batch_paths):
        """使用DataLoader批量处理图片"""
        if len(batch_images) == 0:
            return 0, 0, 0
            
        batch_start_time = time.time()
        
        total_processed = 0
        total_faces_processed = 0
        total_plates_processed = 0
        
        # 处理批次中的每张图片
        for i, (image, img_path) in enumerate(zip(batch_images, batch_paths)):
            try:
                # 确保图像是numpy数组
                if isinstance(image, torch.Tensor):
                    image = image.numpy()
                
                # 确保数据类型正确
                if image.dtype != np.uint8:
                    image = image.astype(np.uint8)
                
                # 检测并处理人脸
                processed_img, faces_count, _ = self.detect_faces(image)
                total_faces_processed += faces_count
                
                # 检测并处理车牌
                processed_img, plates_count, _ = self.detect_plates(processed_img)
                total_plates_processed += plates_count
                
                # 保存处理后的图片
                filename = os.path.basename(str(img_path))
                output_path = os.path.join(self.output_dir, filename)
                
                # 转换回BGR格式保存
                processed_img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, processed_img_bgr)
                
                total_processed += 1
                
            except Exception as e:
                self.logger.error(f"处理图片 {img_path} 时出错: {str(e)}")
                continue
        
        batch_time = time.time() - batch_start_time
        
        # 批次级别的统计信息
        if total_faces_processed > 0 or total_plates_processed > 0:
            self.logger.info(f"批次处理完成: {total_processed} 张图片, "
                           f"检测到 {total_faces_processed} 个人脸, {total_plates_processed} 个车牌 "
                           f"| 耗时: {batch_time:.2f}s")
        else:
            self.logger.debug(f"批次处理完成: {total_processed} 张图片, 无检测目标 | 耗时: {batch_time:.2f}s")
            
        return total_processed, total_faces_processed, total_plates_processed

    def process_images_batch(self, image_paths):
        """批量处理多张图片（16张批处理优化）- 已废弃，使用DataLoader版本"""
        if not image_paths:
            return 0, 0, 0
            
        self.logger.debug(f"开始批量处理 {len(image_paths)} 张图片")
        
        total_processed = 0
        total_faces_processed = 0
        total_plates_processed = 0
        
        # 处理每个图片路径
        for img_path in image_paths:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    self.logger.error(f"无法读取图片: {img_path}")
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 检测并处理人脸
                processed_img, faces_count, _ = self.detect_faces(img_rgb)
                total_faces_processed += faces_count
                
                # 检测并处理车牌
                processed_img, plates_count, _ = self.detect_plates(processed_img)
                total_plates_processed += plates_count
                
                # 保存处理后的图片
                filename = os.path.basename(img_path)
                output_path = os.path.join(self.output_dir, filename)
                processed_img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, processed_img_bgr)
                
                total_processed += 1
                
            except Exception as e:
                self.logger.error(f"处理图片 {img_path} 时出错: {str(e)}")
                continue
        
        self.logger.info(f"批处理完成: {total_processed} 张图片, "
                        f"人脸 {total_faces_processed} 个, 车牌 {total_plates_processed} 个")
        return total_processed, total_faces_processed, total_plates_processed

def batch_process_images(input_dir, output_dir, face_detector, plate_detector, batch_size=16):
    """批量处理目录中的所有图片（使用数据集和DataLoader优化）"""
    logger = logging.getLogger('VideoProcessor.batch_process_images')
    
    # 创建数据集实例
    dataset = VideoFrameDataset(input_dir)
    
    if len(dataset) == 0:
        logger.info(f"在目录 '{input_dir}' 中未找到图片文件")
        return 0, 0, 0
    
    # 创建DataLoader
    dataloader = dataset.get_dataloader(batch_size=batch_size, num_workers=0, shuffle=False)
    
    # 初始化处理器
    processor = CombinedProcessor(face_detector, plate_detector, output_dir, debug_mode=False)
    
    total_images = len(dataset)
    total_processed = 0
    total_faces = 0
    total_plates = 0
    
    logger.info(f"开始批量处理 {total_images} 张图片（每批{batch_size}张）")
    batch_start_time = time.time()
    
    # 使用DataLoader进行批次处理
    processed_batches = 0
    for batch_idx, (batch_images, batch_paths) in enumerate(dataloader):
        batch_start_idx = time.time()
        
        # 处理当前批次
        batch_processed, batch_faces, batch_plates = processor.process_images_batch_dataloader(
            batch_images, batch_paths
        )
        
        total_processed += batch_processed
        total_faces += batch_faces
        total_plates += batch_plates
        processed_batches += 1
        
        # 显示进度（每5个批次显示一次）
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(dataloader):
            progress = (batch_idx + 1) / len(dataloader) * 100
            logger.info(f"处理进度: {progress:.1f}% ({batch_idx + 1}/{len(dataloader)} 批次)")
    
    # 打印批处理统计信息
    batch_total_time = time.time() - batch_start_time
    avg_time_per_image = batch_total_time / total_processed if total_processed > 0 else 0
    
    logger.info(f"批处理完成! 共处理 {total_processed}/{total_images} 张图片")
    logger.info(f"批处理总耗时: {batch_total_time:.2f}秒 | 平均: {avg_time_per_image:.3f}秒/张")
    if total_faces > 0 or total_plates > 0:
        logger.info(f"检测结果: {total_faces} 个人脸, {total_plates} 个车牌")
    
    return total_processed, total_faces, total_plates

def convert_video_to_frames(video_path, output_dir, interval=1):
    """将视频转换为图片帧"""
    logger = logging.getLogger('VideoProcessor.convert_video_to_frames')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频文件: {video_path}")
        return 0
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    logger.info(f"视频信息: {os.path.basename(video_path)} - "
               f"帧率: {fps:.1f}, 总帧数: {total_frames}, 时长: {duration:.1f}秒")
    
    frame_count = 0
    saved_count = 0
    
    # 读取帧并保存
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 按间隔保存帧
        if frame_count % interval == 0:
            frame_filename = f"frame_{frame_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
        frame_count += 1
        
        # 进度提示
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            logger.debug(f"抽帧进度: {progress:.1f}% ({frame_count}/{total_frames})")
    
    cap.release()
    logger.info(f"抽帧完成，共保存 {saved_count} 帧到 {output_dir}")
    return saved_count

def create_video(frame_dir, output_path, fps=30):
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
        'model_weights', 
        'mtcnn_model_path',
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
        'model_weights': paths['model_weights'],
        'mtcnn_model_path': paths['mtcnn_model_path'],
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

# 模拟recordDeal模块（实际使用时替换为真实模块）
class MockRecordDeal:
    @staticmethod
    def read_record2h265_all(input_dir, output_dir):
        logger = logging.getLogger('VideoProcessor.MockRecordDeal')
        logger.info(f"模拟解包record文件: {input_dir} -> {output_dir}")
        os.makedirs(os.path.join(output_dir, "hevcs"), exist_ok=True)
        return True
        
    @staticmethod
    def write_allH265_record_all(input_dir, processed_dir, output_dir):
        logger = logging.getLogger('VideoProcessor.MockRecordDeal')
        logger.info(f"模拟重新打包record文件: {input_dir} + {processed_dir} -> {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        return True

if __name__ == "__main__":
    # 初始化日志器
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
        mtcnn_model_path = config['mtcnn_model_path']
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
        logger.info(f"MTCNN模型路径: {mtcnn_model_path}")
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
        # 使用模拟的recordDeal，实际使用时替换为真实模块
        recordDeal = MockRecordDeal()
        result1 = recordDeal.read_record2h265_all(record_dir, output_h265_dir)
        logger.info(f"解包完成")
        
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
        logger.info("正在加载MTCNN人脸检测模型...")
        logger.info(f"MTCNN模型路径: {mtcnn_model_path}")
        face_detector = MTCNN(model_path=mtcnn_model_path)
        device = face_detector.device
        print(f"face模型运行设备: {device}")
        logger.info("MTCNN人脸检测模型加载完成")
        
        # 初始化YOLOv8车牌检测模型
        logger.info("正在加载YOLOv8车牌检测模型...")
        plate_detector = YOLO(plate_model_path).cuda()
        
        # 检查YOLOv8使用的设备
        if torch.cuda.is_available():
            yolo_device = next(plate_detector.model.parameters()).device
            logger.info(f"YOLOv8车牌检测模型是否运行在GPU: {yolo_device}")
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
        result2 = recordDeal.write_allH265_record_all(record_dir, output_videos_dir, record_output_dir)
        logger.info(f"打包完成")
        
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

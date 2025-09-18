import time
import cv2
import numpy as np
import torch
import torch.nn as nn

from .retinaface import RetinaFace
from .utils.anchors import Anchors
from .utils.config import cfg_mnet, cfg_re50
from .utils.utils import letterbox_image, preprocess_input
from .utils.utils_bbox import decode, decode_landm, non_max_suppression, retinaface_correct_boxes


class Retinaface(object):
    _defaults = {
        "model_path": '/home/liuwj/Retinaface_resnet50.pth',
        "backbone": 'resnet50',
        "confidence": 0.5,
        "nms_iou": 0.45,
        "input_shape": [1280, 1280, 3],
        "letterbox_image": True,
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        return cls._defaults.get(n, f"Unrecognized attribute name '{n}'")

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # 设置配置
        self.cfg = cfg_mnet if self.backbone == "mobilenet" else cfg_re50

        # 初始化设备
        self.device = torch.device('cuda' if self.cuda and torch.cuda.is_available() else 'cpu')

        # 生成 anchors（仅在 letterbox 模式下一次性生成）
        if self.letterbox_image:
            self.anchors = Anchors(self.cfg, image_size=[self.input_shape[0], self.input_shape[1]]).get_anchors()
            self.anchors = self.anchors.to(self.device)
        else:
            self.anchors = None  # 动态生成

        # 构建模型
        self.generate()

    def generate(self):
        """加载模型权重"""
        self.net = RetinaFace(cfg=self.cfg, mode='eval').eval()
        self.net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
        if self.cuda and torch.cuda.is_available():
            self.net = nn.DataParallel(self.net).cuda()
        else:
            self.net = self.net.to(self.device)

        print(f'Model {self.model_path} loaded.')

    # ---------------------------------------------------#
    #   前处理：图像标准化 + 转 Tensor
    # ---------------------------------------------------#
    def preprocess(self, images):
        """
        输入: list of np.ndarray (H, W, 3), RGB
        输出: torch.Tensor (B, C, H, W), normalized
        """
        input_tensors = []
        image_shapes = []

        for img in images:
            h, w = img.shape[:2]
            image_shapes.append([h, w])

            if self.letterbox_image:
                img = letterbox_image(img, [self.input_shape[1], self.input_shape[0]])
            else:
                raise ValueError("Batch inference requires letterbox_image=True for shape alignment.")

            img = preprocess_input(img)
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            input_tensors.append(img)

        # 堆叠成 batch
        batch_tensor = torch.from_numpy(np.stack(input_tensors, axis=0)).float().to(self.device)
        return batch_tensor, torch.tensor(image_shapes, dtype=torch.float32).to(self.device)

    # ---------------------------------------------------#
    #   后处理：解码 + NMS + 坐标还原
    # ---------------------------------------------------#
    def postprocess(self, loc, conf, landms, image_shapes, batch_size):
        """
        批量后处理
        """
        with torch.no_grad():
            # 解码
            device = loc.device
            batch_anchors = self.anchors.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, 4)
            boxes = decode(loc, batch_anchors, self.cfg['variance'])  # (B, N, 4)
            conf = conf[:, :, 1:2]  # (B, N, 1)
            landms = decode_landm(landms, batch_anchors, self.cfg['variance'])  # (B, N, 10)

            # 拼接
            detections = torch.cat([boxes, conf, landms], dim=-1)  # (B, N, 15)

            # NMS
            results = non_max_suppression(detections, self.confidence, self.nms_iou)

            # 校正坐标（从 letterbox 映射回原图）
            if self.letterbox_image:
                results = retinaface_correct_boxes(results, self.input_shape[:2], image_shapes)

        return results  # list of (M_i, 15)

    # ---------------------------------------------------#
    #   主检测接口：批量处理图像
    # ---------------------------------------------------#
    def detect_images(self, images):    
        """
        输入: list of np.ndarray (H, W, 3), RGB
        输出: list of tuples (image, boxes) where boxes are lists of [x1, y1, x2, y2]
        """
        if not isinstance(images, list):
            images = [images]
        batch_size = len(images)

        old_images = [img.copy() for img in images]
        input_tensor, image_shapes = self.preprocess(images)

        with torch.no_grad():
            loc, conf, landms = self.net(input_tensor)
            results = self.postprocess(loc, conf, landms, image_shapes, batch_size)

        final_images_boxes = []
        for i, result in enumerate(results):
            img = old_images[i]
            boxes = []
            if len(result) > 0:
                result = result.cpu().numpy()
                h, w = image_shapes[i].cpu().numpy()

                scale_box = np.array([w, h, w, h])
                result[:, :4] *= scale_box
                boxes = result[:, :4].tolist()

            final_images_boxes.append((img, boxes))

        return final_images_boxes

    # ---------------------------------------------------#
    #   FPS 测试（单图）
    # ---------------------------------------------------#
    def get_FPS(self, image, test_interval=100):
        image = np.array(image, dtype=np.float32)
        im_height, im_width = image.shape[:2]

        if self.letterbox_image:
            image_resized = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors().to(self.device)
            image_resized = image

        image_tensor = torch.from_numpy(
            preprocess_input(image_resized).transpose(2, 0, 1)
        ).unsqueeze(0).float().to(self.device)

        # Warm-up
        with torch.no_grad():
            for _ in range(10):
                self.net(image_tensor)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                self.net(image_tensor)
        t2 = time.time()

        return (t2 - t1) / test_interval

    # ---------------------------------------------------#
    #   获取检测结果（用于 mAP 计算）
    # ---------------------------------------------------#
    def get_map_txt(self, image):
        image = np.array(image, dtype=np.float32)
        im_height, im_width = image.shape[:2]

        if self.letterbox_image:
            image_resized = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors().to(self.device)
            image_resized = image

        image_tensor = torch.from_numpy(
            preprocess_input(image_resized).transpose(2, 0, 1)
        ).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            loc, conf, landms = self.net(image_tensor)
            loc = loc.squeeze(0)
            conf = conf.squeeze(0)[:, 1:2]
            landms = landms.squeeze(0)

            if self.letterbox_image:
                anchors = self.anchors
            else:
                anchors = anchors

            boxes = decode(loc, anchors, self.cfg['variance'])
            landms = decode_landm(landms, anchors, self.cfg['variance'])
            detections = torch.cat([boxes, conf, landms], dim=-1)
            detections = non_max_suppression(detections, self.confidence, self.nms_iou)

            if len(detections) == 0:
                return np.array([])

            if self.letterbox_image:
                detections = retinaface_correct_boxes(
                    detections,
                    self.input_shape[:2],
                    np.array([im_height, im_width])
                )

            scale_box = np.array([im_width, im_height, im_width, im_height])
            scale_landmarks = np.tile([im_width, im_height], 5)
            detections[:, :4] *= scale_box
            detections[:, 5:] *= scale_landmarks

        return detections.cpu().numpy()

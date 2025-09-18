import numpy as np
import torch
from torchvision.ops import batched_nms


#---------------------------------------------------------------#
#   将输出调整为相对于原图的大小（支持批量）
#   input_shape: [H, W] 模型输入尺寸
#   image_shapes: (B, 2) 或 (B, [h, w])，每张图原始尺寸
#   result: list of tensors 或 (B, N, 15) tensor -> 返回 list[(N1,15), (N2,15)...]
#---------------------------------------------------------------#
def retinaface_correct_boxes(batch_result, input_shape, image_shapes):
    """
    batch_result: list of (N, 15) tensors
    input_shape: [H, W]
    image_shapes: (B, 2) numpy array or list
    """
    corrected_results = []
    input_shape = torch.tensor(input_shape, dtype=torch.float32).cuda()

    for i, result in enumerate(batch_result):
        if len(result) == 0:
            corrected_results.append(result)
            continue

        img_h, img_w = image_shapes[i]
        image_shape = torch.tensor([img_h, img_w], dtype=torch.float32, device=result.device)

        new_shape = image_shape * torch.min(input_shape / image_shape)
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape

        scale_box = torch.tensor([scale[1], scale[0], scale[1], scale[0]], device=result.device)
        scale_land = torch.tensor([scale[1], scale[0]] * 5, device=result.device)
        offset_box = torch.tensor([offset[1], offset[0], offset[1], offset[0]], device=result.device)
        offset_land = torch.tensor([offset[1], offset[0]] * 5, device=result.device)

        result[:, :4] = (result[:, :4] - offset_box) * scale_box
        result[:, 5:] = (result[:, 5:] - offset_land) * scale_land

        corrected_results.append(result)

    return corrected_results


#-----------------------------#
#   中心解码，宽高解码
#-----------------------------#
def decode(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :, :2] + loc[:, :, :2] * variances[0] * priors[:, :, 2:],  # cx, cy
        priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1]),           # w, h
    ), dim=-1)

    # 转换为 (x1, y1, x2, y2)
    boxes[:, :, :2] -= boxes[:, :, 2:] / 2  # cx,cy -> x1,y1
    boxes[:, :, 2:] += boxes[:, :, :2]      # w,h -> x2,y2

    return boxes

#-----------------------------#
#   关键点解码
#-----------------------------#
def decode_landm(landm, priors, variances):
    """
    解码头部关键点
    landm: (B, N, 10)
    priors: (N, 4)
    """
    if priors.dim() == 2:
        priors = priors.unsqueeze(0) # (1, N, 4)
    landms = torch.cat((
        priors[:, :, :2] + landm[:, :, :2] * variances[0] * priors[:, :, 2:],
        priors[:, :, :2] + landm[:, :, 2:4] * variances[0] * priors[:, :, 2:],
        priors[:, :, :2] + landm[:, :, 4:6] * variances[0] * priors[:, :, 2:],
        priors[:, :, :2] + landm[:, :, 6:8] * variances[0] * priors[:, :, 2:],
        priors[:, :, :2] + landm[:, :, 8:10] * variances[0] * priors[:, :, 2:]
    ), dim=-1)
    return landms

def iou(b1,b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    
    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)
    
    area_b1 = (b1_x2-b1_x1)*(b1_y2-b1_y1)
    area_b2 = (b2_x2-b2_x1)*(b2_y2-b2_y1)
    
    iou = inter_area/np.maximum((area_b1 + area_b2 - inter_area), 1e-6)
    return iou

#   非极大抑制（支持批量）
#   输入: (B, N, 15) 的检测结果
#   输出: list of (M_i, 15) tensors
#---------------------------------------------------------------#
def non_max_suppression(detection, conf_thres=0.5, nms_thres=0.3):
    """
    detection: (B, N, 15) Tensor
    返回: list of (M_i, 15) tensors
    """
    if detection.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {detection.dim()}D")

    batch_size = detection.shape[0]
    output = []

    for b in range(batch_size):
        boxes_scores = detection[b]  # (N, 15)
        mask = boxes_scores[:, 4] >= conf_thres
        boxes_scores = boxes_scores[mask]
        if len(boxes_scores) == 0:
            output.append(torch.empty((0, 15), device=detection.device))
            continue

        keep = batched_nms(
            boxes=boxes_scores[:, :4],
            scores=boxes_scores[:, 4],
            idxs=torch.zeros(len(boxes_scores), device=boxes_scores.device),  # 单类
            iou_threshold=nms_thres
        )
        output.append(boxes_scores[keep])

    return output

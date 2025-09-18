import torch
import cv2
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor
from face import Retinaface

# -------------------------------
# 配置参数
# -------------------------------
model_path = '/mnt/d/P_WorkSpace/Video-desensitization/Retinaface_resnet50.pth'
image_folder = '/mnt/d/P_WorkSpace/Video-desensitization/temp_processing/temp_camera_right_back/frames'
output_folder = '/mnt/d/P_WorkSpace/Video-desensitization/output_pictures'
batch_size = 32          # 根据显存调整（建议从 32 开始测试）
num_workers = 6           # 图像加载线程数（CPU 核心数附近）
use_amp = True            # 启用自动混合精度（如果模型支持）

os.makedirs(output_folder, exist_ok=True)

# -------------------------------
# 1. 加载模型
# -------------------------------
print("Loading RetinaFace model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

face_detect = Retinaface(
    model_path=model_path,
    backbone="resnet50",
    input_shape=[640, 640, 3],
    confidence=0.5,
    nms_iou=0.4,
    letterbox_image=True,
    cuda=True  # 自动使用 GPU
)

# 确保模型在正确的设备上
face_detect.net.to(device)
face_detect.net.eval()

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

# -------------------------------
# 主推理流程
# -------------------------------
start_time = time.time()

# 获取所有图片路径
image_paths = [
    os.path.join(image_folder, fname) for fname in os.listdir(image_folder)
    if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
]
print(f"找到 {len(image_paths)} 张待处理图片。")

middle_time = time.time()
print(f"模型加载耗时: {middle_time - start_time:.2f} 秒")

# 使用线程池异步保存
executor = ThreadPoolExecutor(max_workers=num_workers)
save_futures = []

# 按 batch 处理
for i in range(0, len(image_paths), batch_size):
    batch_files = image_paths[i:i + batch_size]
    print(f"正在处理第 {i // batch_size + 1} 批次（{len(batch_files)} 张）...")

    # 多线程加载图像
    with ThreadPoolExecutor(max_workers=num_workers) as loader:
        batch_images = list(loader.map(load_image_rgb, batch_files))

    # 批量推理（核心优化点：一次性传入 list[np.ndarray]）
    infer_start = time.time()
    try:
        # 直接调用 detect_images，它内部会调用 preprocess -> inference -> postprocess -> draw
        processed_images = face_detect.detect_images(batch_images)  # 返回 list[PIL.Image 或 np.ndarray]
    except Exception as e:
        print(f"批次推理失败: {e}")
        continue
    infer_end = time.time()
    print(f"批次推理耗时: {infer_end - infer_start:.2f} 秒 ({len(batch_files)} 张)")

    # 转换输出格式（确保是 numpy array）
    for img_path, result_img in zip(batch_files, processed_images):
        output_path = os.path.join(output_folder, f"processed_{os.path.basename(img_path)}")
        
        # 如果返回的是 PIL.Image，转换为 numpy
        if hasattr(result_img, 'convert'):
            result_np = np.array(result_img.convert('RGB'))
        else:
            result_np = result_img  # 假设已经是 HWC uint8 numpy array

        # 提交异步保存任务
        future = executor.submit(save_output_image, result_np, output_path)
        save_futures.append(future)

# 等待所有保存完成
for future in save_futures:
    future.result()

total_time = time.time() - start_time
avg_time_per_image = total_time / len(image_paths)

print(f"\n全部完成！")
print(f"总耗时: {total_time:.2f} 秒")
print(f"平均每张: {avg_time_per_image:.3f} 秒")
print(f"总体吞吐量: {len(image_paths) / total_time:.2f} 张/秒")

# 关闭线程池
executor.shutdown()

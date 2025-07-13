import os
import pickle
import logging
import sys
from tqdm import tqdm
from collections import defaultdict
from cyber_record.record import Record
from google.protobuf.descriptor_pb2 import FileDescriptorSet

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def is_camera_channel(channel_name):
    """检查通道是否为摄像头通道"""
    return 'camera' in channel_name and 'compressed/image' in channel_name

def get_proto_descriptor(msg):
    """获取消息的protobuf描述符"""
    try:
        # 尝试获取序列化的描述符
        if hasattr(msg.DESCRIPTOR, 'serialized_pb'):
            return msg.DESCRIPTOR.serialized_pb
        if hasattr(msg.DESCRIPTOR, 'serialized_blob'):
            return msg.DESCRIPTOR.serialized_blob
    except AttributeError:
        pass
    
    # 回退方案：使用FileDescriptorSet
    try:
        fds = FileDescriptorSet()
        msg.DESCRIPTOR.file.CopyToProto(fds.file.add())
        return fds.SerializeToString()
    except Exception:
        logging.warning(f"无法获取消息描述符: {type(msg).__name__}")
        return None

def extract_camera_data(record_path, output_dir):
    """高效解包摄像头数据到指定目录"""
    os.makedirs(output_dir, exist_ok=True)
    camera_channels = {}
    camera_timestamps = defaultdict(list)  # 记录每个通道的时间戳
    
    with Record(record_path, mode='r') as record:
        # 获取所有通道信息
        channels = record.get_channel_cache()
        
        # 识别摄像头通道
        for channel in channels:
            if is_camera_channel(channel.name):
                safe_name = channel.name.replace('/', '__')
                h265_path = os.path.join(output_dir, f"{safe_name}.h265")
                
                # 打开文件句柄
                h265_file = open(h265_path, 'wb')
                
                camera_channels[channel.name] = {
                    'h265_file': h265_file,
                    'count': 0,
                    'safe_name': safe_name
                }
        
        # 处理所有消息
        for topic, msg, t in tqdm(record.read_messages(), desc="提取消息"):
            # 处理摄像头消息
            if topic in camera_channels:
                try:
                    # 获取图像数据
                    if hasattr(msg, 'data'):
                        img_data = msg.data
                    else:
                        img_data = msg.SerializeToString()
                    
                    # 写入帧数据
                    h265_file = camera_channels[topic]['h265_file']
                    h265_file.write(img_data)
                    camera_channels[topic]['count'] += 1
                    
                    # 记录时间戳
                    camera_timestamps[topic].append(t)
                except Exception as e:
                    logging.error(f"处理摄像头数据失败 ({topic}): {e}")
    
    # 关闭所有文件句柄并记录结果
    for topic, data in camera_channels.items():
        data['h265_file'].close()
        logging.info(f"提取摄像头通道 {topic}: {data['count']} 帧")
        if camera_timestamps[topic]:
            logging.debug(f"时间戳范围: min={min(camera_timestamps[topic])} max={max(camera_timestamps[topic])}")
    
    # 保存时间戳信息
    timestamps_path = os.path.join(output_dir, 'timestamps.pkl')
    with open(timestamps_path, 'wb') as f:
        pickle.dump(dict(camera_timestamps), f)
    
    return len(camera_channels), camera_timestamps

def split_h265_frames(data):
    """更高效的H.265帧分割实现"""
    if not data:
        return []
    
    frames = []
    start = 0
    state = 0  # 状态机：0=初始，1=收到1个0，2=收到2个0，3=收到3个0
    n = len(data)
    
    for i in range(n):
        byte = data[i]
        
        # 状态转移
        if state < 3:
            state = (state + 1) if byte == 0 else 0
        else:  # state == 3 (已收到3个0)
            if byte == 0:
                pass  # 保持状态3
            elif byte == 1:
                # 找到起始码 (00 00 00 01)
                if i - 3 > start:  # 非起始位置
                    frames.append(data[start:i-3])
                start = i - 3
                state = 0
            else:
                state = 0
    
    # 添加最后一帧
    if start < n:
        frames.append(data[start:n])
    
    return frames

def repack_record(original_record, blurred_dir, hevc_dir, output_record):
    """高效重新打包record文件"""
    # 1. 加载时间戳信息
    timestamps_path = os.path.join(hevc_dir, 'timestamps.pkl')
    if not os.path.exists(timestamps_path):
        raise FileNotFoundError(f"时间戳信息文件不存在: {timestamps_path}")
    
    with open(timestamps_path, 'rb') as f:
        original_timestamps = pickle.load(f)
    
    # 2. 准备脱敏视频数据
    video_data = {}
    for file in os.listdir(blurred_dir):
        if file.endswith('.h265'):
            safe_name = file[:-5]  # 移除 .h265 扩展名
            channel_name = safe_name.replace('__', '/')
            
            # 检查通道是否有时间戳数据
            if channel_name not in original_timestamps:
                logging.warning(f"找不到通道 {channel_name} 的时间戳数据")
                continue
            
            # 读取脱敏视频文件
            file_path = os.path.join(blurred_dir, file)
            try:
                with open(file_path, 'rb') as f:
                    video_bytes = f.read()
                
                # 分割视频为帧
                frames = split_h265_frames(video_bytes)
                timestamps = original_timestamps[channel_name]
                
                # 验证帧数量
                if len(frames) != len(timestamps):
                    logging.warning(
                        f"帧数量不匹配: {channel_name} "
                        f"脱敏帧数={len(frames)} 原始时间戳数={len(timestamps)}"
                    )
                    # 使用较小值避免索引越界
                    min_len = min(len(frames), len(timestamps))
                    frames = frames[:min_len]
                    timestamps = timestamps[:min_len]
                
                # 存储帧数据和时间戳
                video_data[channel_name] = list(zip(timestamps, frames))
                logging.info(f"加载脱敏视频: {channel_name} ({len(frames)} 帧)")
                
            except Exception as e:
                logging.error(f"处理视频文件失败 {file_path}: {e}")
    
    # 3. 创建新record并写入数据
    with Record(output_record, mode='w') as new_record:
        total_messages = 0
        camera_messages = 0
        other_messages = 0
        
        # 为每个通道创建帧指针
        frame_pointers = {channel: 0 for channel in video_data}
        
        # 4. 处理原始record中的消息
        with Record(original_record, mode='r') as orig_record:
            for topic, msg, t in tqdm(orig_record.read_messages(), desc="处理消息"):
                try:
                    if topic in video_data:
                        # 获取当前通道的帧指针
                        ptr = frame_pointers[topic]
                        frames = video_data[topic]
                        
                        # 检查是否有可用帧
                        if ptr < len(frames):
                            frame_ts, frame_data = frames[ptr]
                            
                            # 验证时间戳匹配
                            if frame_ts == t:
                                # 创建新消息对象
                                try:
                                    # 尝试使用原始消息类型创建新实例
                                    new_msg = type(msg)()
                                    
                                    # 优先使用data属性
                                    if hasattr(new_msg, 'data'):
                                        new_msg.data = frame_data
                                    else:
                                        # 尝试反序列化
                                        try:
                                            new_msg.ParseFromString(frame_data)
                                        except Exception:
                                            # 回退到复制原始消息结构
                                            new_msg.CopyFrom(msg)
                                            new_msg.ParseFromString(frame_data)
                                    
                                    new_record.write(topic, new_msg, t)
                                    camera_messages += 1
                                    frame_pointers[topic] += 1
                                except Exception as e:
                                    logging.warning(f"创建消息失败 ({topic}, t={t}): {e}, 使用原始消息")
                                    new_record.write(topic, msg, t)
                            else:
                                # 时间戳不匹配，使用原始消息
                                new_record.write(topic, msg, t)
                                logging.debug(f"时间戳不匹配: {topic} 期望={frame_ts} 实际={t}")
                        else:
                            # 没有更多帧可用
                            new_record.write(topic, msg, t)
                            logging.debug(f"无可用脱敏帧: {topic} @ {t}")
                    else:
                        # 非摄像头消息直接写入
                        new_record.write(topic, msg, t)
                        other_messages += 1
                    
                    total_messages += 1
                    
                    # 定期刷新写入缓冲区
                    if total_messages % 1000 == 0:
                        new_record._writer.flush()
                except Exception as e:
                    logging.error(f"处理消息失败 ({topic}, t={t}): {e}")
    
    # 5. 记录未使用的脱敏帧
    for channel, ptr in frame_pointers.items():
        frames = video_data[channel]
        unused = len(frames) - ptr
        if unused > 0:
            logging.warning(f"通道 {channel} 有 {unused}/{len(frames)} 个未使用的脱敏帧")
    
    logging.info(f"打包完成! 总计: {total_messages} 条消息")
    logging.info(f"摄像头消息: {camera_messages} (脱敏: {len(video_data)})")
    logging.info(f"其他消息: {other_messages}")
    return total_messages

def main():
    # 配置路径
    original_record = "e:\\样例\\样例\\2025-04-20_17-21-12.record.00002.17-22-22"
    hevc_dir = "hevcs3"
    blurred_dir = "E:\\face\\Blured"
    final_record = "output_final.record"
    
    # 步骤1: 解包摄像头数据
    logging.info("开始解包数据...")
    camera_count, timestamps = extract_camera_data(original_record, hevc_dir)
    logging.info(f"解包完成: {camera_count} 个摄像头通道")
    
    # 步骤2: 重新打包record
    logging.info("开始重新打包record文件...")
    repack_record(
        original_record=original_record,
        blurred_dir=blurred_dir,
        hevc_dir=hevc_dir,
        output_record=final_record
    )
    logging.info(f"打包完成: {final_record}")

if __name__ == "__main__":
    main()
    #截断分割
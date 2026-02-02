import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image

def initialize_camera():
    """
    初始化所有相机
    根据实际情况替换 serial_numbers 中的序列号。
    返回一个字典，键为相机型号，值为相机的管道对象。
    """
    # 创建上下文对象
    context = rs.context()
    connected_devices = [device.get_info(rs.camera_info.serial_number) for device in context.devices]
    connected_serials = [device.get_info(rs.camera_info.serial_number) for device in context.query_devices()]

    print("Connected camera serials:", connected_devices)
    
    # 映射相机型号到序列号（请替换为您的相机实际序列号）
    serial_numbers = {
        'L515': 'f0190751', # 'f0265239',f0265239  f0210138 # 替换为 L515 的序列号 f0265239
        'D435i': '332522070934',  # 替换为 D435i 的序列号 332522071841
        # 'D455': '', # 替换为 D455 的序列号
        # 'D435': ''  # 替换为 D435 的序列号
    }

    # 初始化每个相机的管道
    pipelines = {}
    align_objects = {}  # 存储对齐对象
    
    for model, serial in serial_numbers.items():
        print(f"Trying to initialize model: {model}, serial: {serial}")
        if serial not in connected_serials:
            print(f"Error: Serial {serial} not connected!")
            continue
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        
        # 为L515启用深度流
        if model == 'L515':
            cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
            # cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            # 创建对齐对象，将深度对齐到彩色
            align_objects[model] = rs.align(rs.stream.color)
        else:
            cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        pipeline.start(cfg)
        pipelines[model] = pipeline
        print(f"Initialized camera {model} with serial number {serial}.")
    
    # 返回管道和对齐对象
    return pipelines, align_objects

def get_L515_image(pipelines):
    """
    从 pipelines 中获取 L515 相机的图像，转换为 PIL Image 并调整为 (224, 224) 大小。
    """
    if 'L515' not in pipelines:
        print("L515 camera not found in pipelines.")
        return None
    pipeline = pipelines['L515']
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        print("No color frame captured from L515 camera.")
        return None
    color_image = np.asanyarray(color_frame.get_data())
    # 将BGR格式转换为RGB格式
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(color_image_rgb)
    # image = image.resize((224, 224)) # resize for openvla
    image = image.resize((640, 480)) # resize for pi0
    return image

def get_L515_depth(pipelines, align_objects=None, return_colormap=False, return_raw=False):
    """
    从 pipelines 中获取 L515 相机的深度图像。
    
    Args:
        pipelines: 相机管道字典
        align_objects: 对齐对象字典（如果提供，将返回对齐后的深度图）
        return_colormap: 是否返回彩色映射的深度图（用于可视化）
        return_raw: 是否返回原始深度数据（numpy数组，单位为毫米）
    
    Returns:
        如果 return_raw=True: 返回原始深度numpy数组
        如果 return_colormap=True: 返回彩色映射的PIL Image
        默认: 返回归一化的灰度PIL Image
    """
    if 'L515' not in pipelines:
        print("L515 camera not found in pipelines.")
        return None
        
    pipeline = pipelines['L515']
    frames = pipeline.wait_for_frames()
    
    # 如果提供了对齐对象，使用对齐后的深度帧
    if align_objects and 'L515' in align_objects:
        aligned_frames = align_objects['L515'].process(frames)
        depth_frame = aligned_frames.get_depth_frame()
    else:
        depth_frame = frames.get_depth_frame()
        
    if not depth_frame:
        print("No depth frame captured from L515 camera.")
        return None
    
    # 获取深度数据（单位：毫米）
    depth_image = np.asanyarray(depth_frame.get_data())
    
    if return_raw:
        # 返回原始深度数据
        return depth_image
    
    if return_colormap:
        # 创建彩色映射用于可视化
        # 将深度值归一化到0-255范围
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        # 转换为RGB格式的PIL Image
        depth_colormap_rgb = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
        depth_pil = Image.fromarray(depth_colormap_rgb)
        return depth_pil.resize((224, 224))
    
    # 默认：返回归一化的灰度深度图
    # 设置有效深度范围（毫米）
    min_depth = 200  # 0.2米
    max_depth = 4000  # 4米
    
    # 裁剪深度值到有效范围
    depth_clipped = np.clip(depth_image, min_depth, max_depth)
    
    # 归一化到0-255
    depth_normalized = ((depth_clipped - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    
    # 转换为PIL Image
    depth_pil = Image.fromarray(depth_normalized)
    return depth_pil #.resize((224, 224))

def get_D435_image(pipelines):
    """
    从 pipelines 中获取 D435 相机的图像，转换为 PIL Image 并调整为 (224, 224) 大小。
    """
    if 'D435i' not in pipelines:
        print("D435i camera not found in pipelines.")
        return None
    pipeline = pipelines['D435i']
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        print("No color frame captured from D435i camera.")
        return None
    color_image = np.asanyarray(color_frame.get_data())
    # 将BGR格式转换为RGB格式
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(color_image_rgb)
    # image = image.resize((224, 224)) # size for openvla
    image = image.resize((640, 480)) # resize for pi0

    return image

def stop_camera(pipelines):
    """
    停止所有相机管道。
    """
    if isinstance(pipelines, dict):
        for pipeline in pipelines.values():
            pipeline.stop()
    else:
        pipelines.stop()
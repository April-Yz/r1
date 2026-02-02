import pyrealsense2 as rs
import numpy as np
import cv2
import time
import json
import os
"""
python collect_hand.py --task your_task_name
python collect_hand.py --task pour
"""

class DataRecorder:
    def __init__(self, task_name="pour"):
        # 1. 配置 Pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 采集分辨率（更大视角）
        self.capture_width, self.capture_height = 1280, 720
        # 保存分辨率
        self.width, self.height = 640, 360
        self.fps = 30
        self.config.enable_stream(rs.stream.depth, self.capture_width, self.capture_height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.capture_width, self.capture_height, rs.format.bgr8, self.fps)
        
        # 录制状态标志
        self.is_recording = False
        self.record_idx = 0  # 录制计数，用于文件名递增
        self.out_color = None
        self.out_depth = None
        self.task_name = task_name

    def save_intrinsics(self, profile, folder="."):
        """保存相机内参到 JSON"""
        depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intrinsics = depth_stream.get_intrinsics()
        params = {
            "width": intrinsics.width, "height": intrinsics.height,
            "fx": intrinsics.fx, "fy": intrinsics.fy,
            "ppx": intrinsics.ppx, "ppy": intrinsics.ppy,
            "model": str(intrinsics.model), "coeffs": intrinsics.coeffs
        }
        filename = os.path.join(folder, f"params_{self.record_idx}.json")
        with open(filename, 'w') as f:
            json.dump(params, f, indent=4)
        print(f"[Info] 参数已保存: {filename}")

    def start_recording(self, profile):
        """初始化视频写入器"""
        self.record_idx += 1
        print(f"\n>>> 开始录制 #{self.record_idx} ...")
        # hand/{task_name} 文件夹路径
        hand_dir = os.path.join("hand", self.task_name)
        if not os.path.exists(hand_dir):
            os.makedirs(hand_dir)
        # mp4编码
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # 简单命名
        color_name = os.path.join(hand_dir, f"rgb_{self.record_idx-1}.mp4")
        depth_name = os.path.join(hand_dir, f"depth_{self.record_idx-1}.mp4")
        # 创建写入对象
        self.out_color = cv2.VideoWriter(color_name, fourcc, self.fps, (self.width, self.height))
        self.out_depth = cv2.VideoWriter(depth_name, fourcc, self.fps, (self.width, self.height))
        self.is_recording = True
        # 保存相机参数到hand/{task_name}目录
        self.save_intrinsics(profile, folder=hand_dir)

    def stop_recording(self):
        """释放资源"""
        if self.is_recording:
            print(f"<<< 停止录制 #{self.record_idx}")
            self.out_color.release()
            self.out_depth.release()
            self.is_recording = False

    def run(self):
        # 启动相机
        profile = self.pipeline.start(self.config)
        # 创建对齐对象 (将深度图对齐到 RGB)
        align = rs.align(rs.stream.color)
        try:
            print("相机已启动。按 '空格' 开始/停止录制，按 'ESC' 退出。")
            while True:
                # 获取帧
                frames = self.pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                # 转换为 numpy 数组
                color_image_full = np.asanyarray(color_frame.get_data())
                depth_image_full = np.asanyarray(depth_frame.get_data())
                # resize到保存分辨率
                color_image = cv2.resize(color_image_full, (self.width, self.height))
                depth_image = cv2.resize(depth_image_full, (self.width, self.height))
                # 深度图伪彩色处理 (用于显示和保存可视视频)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                # --- 录制逻辑 ---
                if self.is_recording:
                    # 写入视频帧
                    self.out_color.write(color_image)
                    self.out_depth.write(depth_colormap)
                    # 在画面上添加 "Recording" 红点提示
                    cv2.circle(color_image, (30, 30), 10, (0, 0, 255), -1)
                # --- 显示画面 ---
                cv2.imshow('RealSense Color', color_image)
                cv2.imshow('RealSense Depth', depth_colormap)
                # --- 键盘控制 ---
                key = cv2.waitKey(1) & 0xFF
                # 按下 ESC (27) 或 q (113) 退出
                if key == 27 or key == 113:
                    break
                # 按下 空格 (32) 切换录制状态
                elif key == 32:
                    if not self.is_recording:
                        self.start_recording(profile)
                    else:
                        self.stop_recording()
        finally:
            self.stop_recording() # 确保退出前保存文件
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RealSense数据采集")
    parser.add_argument('--task', type=str, default="pour", help="任务名称，决定保存路径 hand/{task}/")
    args = parser.parse_args()
    recorder = DataRecorder(task_name=args.task)
    recorder.run()
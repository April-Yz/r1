# import pyrealsense2 as rs
# import numpy as np
# import cv2
# import json
# import os
# import argparse

# class DataRecorder:
#     def __init__(self, task_name="pour"):
#         # 1. 配置 Pipeline
#         self.pipeline = rs.pipeline()
#         self.config = rs.config()
        
#         # 设定原生分辨率
#         self.width, self.height = 1280, 720
#         self.fps = 30
        
#         # 显式配置流
#         self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
#         self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        
#         # 录制状态
#         self.is_recording = False
#         self.record_idx = 0
#         self.out_color = None
#         self.out_depth_vis = None # 仅用于预览展示的视频
#         self.task_name = task_name

#     def save_intrinsics(self, profile, folder):
#         """保存当前流的原始相机内参"""
#         # 获取 color 流的内参（因为我们做了 align 对齐）
#         color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
#         intrinsics = color_stream.get_intrinsics()
        
#         params = {
#             "width": intrinsics.width,
#             "height": intrinsics.height,
#             "fx": intrinsics.fx,
#             "fy": intrinsics.fy,
#             "ppx": intrinsics.ppx,
#             "ppy": intrinsics.ppy,
#             "model": str(intrinsics.model),
#             "coeffs": intrinsics.coeffs
#         }
        
#         filename = os.path.join(folder, f"params_{self.record_idx}.json")
#         with open(filename, 'w') as f:
#             json.dump(params, f, indent=4)
#         print(f"[Info] 相机内参已同步保存: {filename}")

#     def start_recording(self, profile):
#         self.record_idx += 1
#         hand_dir = os.path.join("hand", self.task_name)
#         if not os.path.exists(hand_dir):
#             os.makedirs(hand_dir)

#         print(f"\n>>> 正在录制序列 #{self.record_idx}...")
        
#         # 视频编码设置
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         color_path = os.path.join(hand_dir, f"rgb_{self.record_idx-1}.mp4")
#         depth_vis_path = os.path.join(hand_dir, f"depth_vis_{self.record_idx-1}.mp4")
        
#         self.out_color = cv2.VideoWriter(color_path, fourcc, self.fps, (self.width, self.height))
#         self.out_depth_vis = cv2.VideoWriter(depth_vis_path, fourcc, self.fps, (self.width, self.height))
        
#         self.save_intrinsics(profile, hand_dir)
#         self.is_recording = True

#     def stop_recording(self):
#         if self.is_recording:
#             print(f"<<< 录制停止，已保存至 #{self.record_idx-1}")
#             self.out_color.release()
#             self.out_depth_vis.release()
#             self.is_recording = False

#     def run(self):
#         # 启动相机
#         profile = self.pipeline.start(self.config)
#         # 对齐器：将深度图对齐到彩色图视角
#         align = rs.align(rs.stream.color)
        
#         try:
#             print(f"--- 任务: {self.task_name} ---")
#             print("控制说明: [空格]: 开始/停止录制 | [ESC/Q]: 退出程序")
            
#             while True:
#                 frames = self.pipeline.wait_for_frames()
#                 aligned_frames = align.process(frames)
                
#                 color_frame = aligned_frames.get_color_frame()
#                 depth_frame = aligned_frames.get_depth_frame()
                
#                 if not color_frame or not depth_frame:
#                     continue

#                 # 直接转换为 numpy 数组 (不进行 resize)
#                 color_image = np.asanyarray(color_frame.get_data())
#                 depth_data = np.asanyarray(depth_frame.get_data()) # 原始 16bit 深度值

#                 # 生成用于显示的彩色深度图
#                 depth_colormap = cv2.applyColorMap(
#                     cv2.convertScaleAbs(depth_data, alpha=0.03), 
#                     cv2.COLORMAP_JET
#                 )

#                 if self.is_recording:
#                     # 1. 写入 RGB 视频
#                     self.out_color.write(color_image)
#                     # 2. 写入 8bit 伪彩色深度视频（仅供查看）
#                     self.out_depth_vis.write(depth_colormap)
                    
#                     # 提示文字
#                     cv2.putText(color_image, "RECORDING", (50, 50), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#                 # 显示
#                 # 拼接显示以节省窗口空间
#                 display_img = np.hstack((cv2.resize(color_image, (640, 360)), 
#                                         cv2.resize(depth_colormap, (640, 360))))
#                 cv2.imshow('RealSense (RGB | Depth)', display_img)

#                 key = cv2.waitKey(1) & 0xFF
#                 if key == 27 or key == ord('q'):
#                     break
#                 elif key == 32: # Space
#                     if not self.is_recording:
#                         self.start_recording(profile)
#                     else:
#                         self.stop_recording()

#         finally:
#             self.stop_recording()
#             self.pipeline.stop()
#             cv2.destroyAllWindows()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--task', type=str, default="pour")
#     args = parser.parse_args()
    
#     recorder = DataRecorder(task_name=args.task)
#     recorder.run()
import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
import argparse
import re

class DataRecorder:
    def __init__(self, task_name="pour"):
        self.task_name = task_name
        self.save_dir = os.path.join("hand", self.task_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 1. 自动计算起始索引 (find max index in folder)
        self.record_idx = self._get_next_index()
        
        # 2. 配置 Pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.width, self.height = 1280, 720
        self.fps = 30
        
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        
        self.is_recording = False
        self.out_color = None
        self.out_depth_vis = None
        self.current_depth_dir = None
        self.current_frame_idx = 0

    def _get_next_index(self):
        """扫描文件夹，返回当前最大的索引 + 1"""
        files = os.listdir(self.save_dir)
        # 寻找形如 rgb_0.mp4, params_0.json 中的数字
        indices = [int(re.findall(r'\d+', f)[0]) for f in files if re.findall(r'\d+', f)]
        if not indices:
            return 0
        return max(indices) + 1

    def save_intrinsics(self, profile, current_id):
        """保存当前 ID 对应的相机内参"""
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_stream.get_intrinsics()
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        
        params = {
            "id": current_id,
            "width": intrinsics.width, "height": intrinsics.height,
            "fx": intrinsics.fx, "fy": intrinsics.fy,
            "ppx": intrinsics.ppx, "ppy": intrinsics.ppy,
            "model": str(intrinsics.model), "coeffs": intrinsics.coeffs,
            "depth_scale_m": depth_scale,
            "depth_aligned_to_color": True,
            "depth_storage": f"depth_{current_id}/%06d.png"
        }
        
        filename = os.path.join(self.save_dir, f"params_{current_id}.json")
        with open(filename, 'w') as f:
            json.dump(params, f, indent=4)
        print(f"[Info] 内参已保存: {filename}")

    def start_recording(self, profile):
        # 使用当前确定的 record_idx
        curr_id = self.record_idx
        print(f"\n>>> 正在录制序列 #{curr_id}...")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        color_path = os.path.join(self.save_dir, f"rgb_{curr_id}.mp4")
        depth_vis_path = os.path.join(self.save_dir, f"depth_vis_{curr_id}.mp4")
        self.current_depth_dir = os.path.join(self.save_dir, f"depth_{curr_id}")
        os.makedirs(self.current_depth_dir, exist_ok=True)
        self.current_frame_idx = 0
        
        # 确保保存分辨率与相机输出完全一致
        self.out_color = cv2.VideoWriter(color_path, fourcc, self.fps, (self.width, self.height))
        self.out_depth_vis = cv2.VideoWriter(depth_vis_path, fourcc, self.fps, (self.width, self.height))
        
        self.save_intrinsics(profile, curr_id)
        self.is_recording = True

    def stop_recording(self):
        if self.is_recording:
            print(f"<<< 录制停止，已保存 ID: {self.record_idx}")
            self.out_color.release()
            self.out_depth_vis.release()
            self.is_recording = False
            self.current_depth_dir = None
            self.current_frame_idx = 0
            # 停止后，索引自增，准备下一次录制
            self.record_idx += 1

    def run(self):
        profile = self.pipeline.start(self.config)
        align = rs.align(rs.stream.color)
        
        try:
            print(f"--- 任务: {self.task_name} | 下一个起始 ID: {self.record_idx} ---")
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_data = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_data, alpha=0.03), cv2.COLORMAP_JET)

                if self.is_recording:
                    if self.current_frame_idx == 0:  # 录制第一帧时打印一次
                        ds = profile.get_device().first_depth_sensor().get_depth_scale()
                        print("depth dtype:", depth_data.dtype, "min:", depth_data.min(), "max:", depth_data.max(), "scale:", ds)
                    self.out_color.write(color_image)
                    self.out_depth_vis.write(depth_colormap)
                    depth_path = os.path.join(self.current_depth_dir, f"{self.current_frame_idx:06d}.png")
                    cv2.imwrite(depth_path, depth_data.astype(np.uint16))
                    self.current_frame_idx += 1
                    cv2.circle(color_image, (40, 40), 15, (0, 0, 255), -1)

                # 仅在显示预览时 resize，不影响保存的数据
                show_rgb = cv2.resize(color_image, (640, 360))
                show_depth = cv2.resize(depth_colormap, (640, 360))
                cv2.imshow('Recorder (Space: Record | Esc: Quit)', np.hstack((show_rgb, show_depth)))

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break
                elif key == 32: # Space
                    if not self.is_recording:
                        self.start_recording(profile)
                    else:
                        self.stop_recording()
        finally:
            self.stop_recording()
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="pour")
    args = parser.parse_args()
    DataRecorder(task_name=args.task).run()

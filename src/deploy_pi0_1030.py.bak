"""
PI0 Robot Controller - Optimized Version
Author: Wenkai (Optimized)
"""

import time
import numpy as np
import torch
import os
from PIL import Image
from scipy.spatial.transform import Rotation as R
from pyrealsense_image import initialize_camera, get_L515_image, get_D435_image, stop_camera
from openpi_client import websocket_client_policy

# Optional video writer
try:
    import imageio
    _HAS_IMAGEIO = True
except Exception:
    imageio = None
    _HAS_IMAGEIO = False


def make_unique_file_path(save_dir, base_name, ext='mp4'):
    """Return a non-existing file path in save_dir with pattern
    base_name_{i}.ext.

    Ensures save_dir exists. Starts from index 0 and increments until a free
    name is found.
    """
    os.makedirs(save_dir, exist_ok=True)
    i = 0
    while True:
        fname = f"{base_name}_{i}.{ext}"
        full = os.path.join(save_dir, fname)
        if not os.path.exists(full):
            return full
        i += 1


class PI0RobotController:
    """Simplified PI0 Robot Controller"""
    
    # Preset positions and task prompts
    # input must follow [x, y, z, qw, qx, qy, qz, gripper]
    
    TASK_PROMPTS = {
        'place_apple_on_the_plate': "Put apple on the plate",
        "place_carrot_on_the_plate": "Put carrot on the plate",
        "place_bread_on_the_plate": "Put bread on the plate",
        "place_croissant_on_the_plate": "Put croissant on the plate",
        "place_red_cube_on_yellow_cube": "Put red cube on yellow cube",
        "place_eggplant_in_the_bowl": "Put eggplant in the bowl",
        "pick_up_plastic_bottle": "Pick up plastic bottle",
        "place_banana_near_the_plate": "Move banana near the plate",
        "place_mango_in_the_basket": "Put mango in the basket",
        "place_mango_in_the_plate": "Put mango in the yellow plate",
        "place_orange_in_the_plate": "Put orange in the plate"
        
    }

    POSITION_PRESETS = {
        "place_apple_on_the_plate": (0.337484, -0.085453, 0.297076, 0.00, 0.00, 1.0, 0.0, 1.0),
        "place_carrot_on_the_plate": (0.3157, 0.058, 0.322154, 0.00, 0.00, 1.0, 0.0, 1.0),
        "place_bread_on_the_plate": (0.309957, 0.051831, 0.323434, 0.00, 0.00, 1.0, 0.0, 1.0),
        "place_croissant_on_the_plate": (0.29112300276756287, 0.26573899388313293, 0.2070660024881363, 0.00, 0.00, 1.0, 0.0, 1.0),
        "place_blue_cube_on_green_cube": (0.379047, 0.000818, 0.235766, 0.00, 0.00, 1.0, 0.0, 1.0),
        "place_eggplant_in_the_bowl": (0.404717, -0.026984, 0.271975, 0.00, 0.00, 1.0, 0.0, 1.0),
        "pick_up_plastic_bottle": (0.388340, 0.035733, 0.292288, 0.00, 0.00, 1.0, 0.0, 1.0),
        "place_banana_near_the_plate": (0.29112300276756287, 0.26573899388313293, 0.2070660024881363, 0.00, 0.00, 1.0, 0.0, 1.0),
        "place_mango_in_the_basket": (0.337484, -0.035453, 0.297076, 0.00, 0.00, 1.0, 0.0, 1.0),
        "place_mango_in_the_plate": (0.337484, -0.085453, 0.297076, 0.00, 0.00, 1.0, 0.0, 1.0),
        "place_orange_in_the_plate": (0.337484, -0.085453, 0.297076, 0.00, 0.00, 1.0, 0.0, 1.0),
    }
    
    def __init__(self, controller, websocket_host="0.0.0.0", websocket_port=8000, 
                 save_video=True, video_path=None, video_fps=10):
        self.controller = controller
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pipelines, self.align_objects = initialize_camera()
        self._warm_up_cameras()
        
        # Set numpy print precision
        np.set_printoptions(precision=4, suppress=True)
        
        # Initialize PI0 client
        self.client = websocket_client_policy.WebsocketClientPolicy(websocket_host, websocket_port)
        print(f"Initialized PI0 client with WebSocket at {websocket_host}:{websocket_port}")
        
        # Video saving configuration
        self.save_video = save_video
        self.video_path = video_path
        self.video_fps = video_fps
        self.main_frames = []  # collected frames (as numpy arrays RGB)
    
    def _warm_up_cameras(self, warm_up_frames=10):
        """Warm up cameras"""
        print("[Client] Warming up cameras...")
        for _ in range(warm_up_frames):
            try:
                get_L515_image(self.pipelines)
                get_D435_image(self.pipelines)
            except Exception as e:
                print(f"[Client] Warm-up warning: {e}")
        print("[Client] Camera warm-up completed")
    
    def initialize_camera_1(self, warm_up_frames=10):
        """Initialize cameras and warm up"""
        try:
            self.pipelines = initialize_camera()
            if not self.pipelines:
                print("Failed to initialize cameras")
                return False
                
            print("Camera pipeline initialized.")
            
            # Camera warm-up with error checking
            for i in range(warm_up_frames):
                main_img = get_L515_image(self.pipelines)
                wrist_img = get_D435_image(self.pipelines)
                if main_img is None or wrist_img is None:
                    print(f"Warning: Failed to get images during warm-up (frame {i})")
                else:
                    print(f"Warm-up frame {i+1}/{warm_up_frames} successful")
                    
            return True
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            return False
    
    def capture_images(self, step=None, save_dir=None):
        """Capture images from RealSense cameras"""
        try:
            # L515 for main view
            main_image = get_L515_image(self.pipelines)
            # D435i for wrist view
            wrist_image = get_D435_image(self.pipelines)
            
            if main_image is None or wrist_image is None:
                print("[Client] Failed to capture images")
                return None, None
            
            # Collect main_image frames for video
            if self.save_video:
                try:
                    # main_image is a PIL Image; convert to RGB numpy array
                    main_np = np.array(main_image.convert("RGB"))
                    self.main_frames.append(main_np)
                    print(f"[Client] Collected {len(self.main_frames)} frames for video")
                except Exception as e:
                    print(f"[Client] Failed to append frame for video: {e}")
            
            # Save images if requested
            if step is not None and save_dir:
                os.makedirs(save_dir, exist_ok=True)
                main_path = os.path.join(save_dir, f"main_{step}.png")
                wrist_path = os.path.join(save_dir, f"wrist_{step}.png")
                main_image.save(main_path)
                wrist_image.save(wrist_path)
                print(f"[Client] Captured images - Main: {main_image.size}, Wrist: {wrist_image.size}")
            
            return main_image, wrist_image
            
        except Exception as e:
            print(f"[Client] Failed to capture images: {e}")
            return None, None
    
    def _convert_to_bgr(self, image):
        """Convert image to BGR format"""
        rgb_image = image.convert("RGB")
        channels = rgb_image.split()[::-1]
        return Image.merge("RGB", channels)
    
    def quaternion_to_rpy(self, quaternion):
        """Convert quaternion to roll-pitch-yaw angles"""
        return R.from_quat(quaternion).as_euler('xyz', degrees=False)
    
    def save_video_file(self, task_name, save_dir="."):
        """Save collected main camera frames as video"""
        if not self.save_video or len(self.main_frames) == 0:
            return
        
        # Choose a non-overwriting path
        if self.video_path:
            # User provided a path: use its directory and base name
            provided_dir = os.path.dirname(self.video_path) or save_dir
            provided_base = os.path.splitext(os.path.basename(self.video_path))[0]
            provided_ext = os.path.splitext(self.video_path)[1].lstrip('.') or 'mp4'
            video_path = make_unique_file_path(provided_dir, provided_base, provided_ext)
        else:
            # Default: use task name with numeric suffix
            video_path = make_unique_file_path(save_dir, task_name, 'mp4')
        
        print(f"[Client] Saving video to: {video_path} ({len(self.main_frames)} frames)")
        
        try:
            if _HAS_IMAGEIO:
                # imageio expects frames with dtype=uint8
                writer = imageio.get_writer(video_path, fps=self.video_fps)
                for fr in self.main_frames:
                    # ensure uint8
                    fr_uint8 = fr.astype('uint8') if fr.dtype != np.uint8 else fr
                    writer.append_data(fr_uint8)
                writer.close()
                print("[Client] Video saved successfully (imageio) in:", video_path)
            else:
                # fallback: save frames as png sequence
                png_dir = os.path.splitext(video_path)[0] + '_frames'
                os.makedirs(png_dir, exist_ok=True)
                for i, fr in enumerate(self.main_frames):
                    img = Image.fromarray(fr.astype('uint8'))
                    img.save(os.path.join(png_dir, f'frame_{i:05d}.png'))
                print(f"[Client] imageio not available; saved frames to: {png_dir}")
        except Exception as e:
            print(f"[Client] Failed to save video/frames: {e}")
    
    def preset_position(self, task_name):
        """Set robot to preset position"""
        pos = (0.337484, -0.035453, 0.297076, 0.00, 0.00, 1.0, 0.0, 1.0)
        self.controller.execute_eef(pos, task_name)
        return True
        if task_name not in self.POSITION_PRESETS:
            print(f"Preset {task_name} not found")
            return False
        
        self.controller.execute_eef(self.POSITION_PRESETS[task_name], task_name)
        return True
    
    def prepare_inference_data(self, main_image, wrist_image, current_state, prompt):
        """Prepare data for inference"""
        return {
            "observation/image": np.asarray(main_image, dtype=np.uint8),
            "observation/wrist_image": np.asarray(wrist_image, dtype=np.uint8),
            "observation/state": current_state,
            "prompt": prompt,
        }
    
    def process_action(self, action_step):
        """Process absolute action step"""
        # Extract absolute position
        new_position = [round(action_step[i], 5) for i in range(3)]
        
        # Extract absolute orientation (assuming RPY format in action)
        new_rpy = [round(action_step[i], 5) for i in range(3, 6)]
        
        # Process gripper command
        gripper_command = action_step[6]
        gripper_position = round(gripper_command, 5)
        
        # Override with fixed orientation values if needed
        # new_rpy = [3.1415926, 0.0, 3.1415926]
        
        return new_position, new_rpy, gripper_position
    
    def execute_action_chunk(self, all_actions, chunk_size, merge_step, step, 
                             task_name):
        """Execute a chunk of actions"""
        n_steps = min(len(all_actions), chunk_size)
        
        for step_idx in range(0, n_steps, merge_step):
            # For absolute actions, we take the last action in the merge window
            # or could take the mean, but taking last is more common
            merged_chunk = all_actions[step_idx:step_idx + merge_step]
            if len(merged_chunk) > 0:
                # Use the last action in the chunk (most recent)
                action_step = merged_chunk[-1]
            else:
                continue
            
            # Process and execute action
            new_pos, new_rpy, grip = self.process_action(action_step)
            
            final_action = new_pos + new_rpy + [grip]
            print(f"Executing action: {np.array(final_action)}")
            self.controller.execute_eef(final_action, task_name)
        
        # time.sleep(0.1)  # brief pause after chunk execution
        # Return the last executed action components
        return new_pos, new_rpy, grip
    
    def run_control_loop(self, task_name="place_apple_on_the_plate", n_iterations=200, 
                        chunk_size=15, merge_step=2, loop_interval=0.1):
        """Main control loop"""
        # # Initialize camera if needed
        # if self.pipelines is None:
        #     if not self.initialize_camera_1():
        #         print("Failed to initialize cameras. Exiting...")
        #         return False

        self.preset_position('place_apple_on_the_plate')

        # Get task prompt
        #prompt = self.TASK_PROMPTS.get(task_name, self.TASK_PROMPTS["place_banana_near_the_plate"])
        prompt = task_name
         
        print(f"Task: {task_name}, Prompt: {prompt}")
        
        # Initialize robot state
        execute_action = self.POSITION_PRESETS['place_apple_on_the_plate'] ## edited to fixed initial position
        init_pose = list(execute_action[:3])
        # init_quat_exe = execute_action[3:7]
        init_quat_infer = [execute_action[4], execute_action[5], execute_action[6], execute_action[3]]
        
        # print(f"Initial pose: {init_pose}, Quaternion: {init_quat_infer}")
        
        # init_rpy_exe = self.quaternion_to_rpy(init_quat_exe)
        init_rpy_infer = self.quaternion_to_rpy(init_quat_infer) 
        init_gripper = execute_action[-1]
        
        print(f"Initial state - Pose: {init_pose}, RPY: {init_rpy_infer}, Gripper: {init_gripper}")

        step = 0

        try:
            while step < n_iterations:
                print(f"\n--- Step {step} ---")
                start_time = time.time()
                
                # Capture images
                main_image, wrist_image = self.capture_images(step)

                if main_image is None or wrist_image is None:
                    print("Failed to capture camera images, skipping step")
                    time.sleep(loop_interval)
                    continue
                
                # Prepare current state
                current_state = np.concatenate((init_pose, init_rpy_infer, [init_gripper]))
                
                # Prepare inference data
                element = self.prepare_inference_data(main_image, wrist_image, current_state, prompt)

                print("current_state (input to model):", current_state)

                # Perform inference
                inf_time = time.time()
                action = self.client.infer(element)
                # print(f"Inference time: {time.time() - inf_time:.4f}s")
                
                # Process actions
                all_actions = np.asarray(action["actions"])
                print(f"Received {len(all_actions)} actions from model")
                # breakpoint()
                actions_to_execute = all_actions[:chunk_size]
                print(f"Actions to execute: {actions_to_execute}")
                
                # # Log absolute positions
                # absolute_actions = np.zeros_like(actions_to_execute)
                # absolute_actions[:,0:6] = current_state[0:6] + actions_to_execute[0:6]
                # absolute_actions[:,6] = actions_to_execute[:,6]
                # print("Actions to execute (absolute positions):")
                # for i, abs_action in enumerate(absolute_actions):
                #     print(f"  Step {i+1}: {abs_action}")
                
                # Execute action chunk
                last_pose, last_rpy, last_gripper = self.execute_action_chunk(
                    actions_to_execute, chunk_size, merge_step, step, task_name
                )
                
                # Since actions are absolute, we update the state to the last executed position
                init_pose, init_rpy_infer, init_gripper = (
                    last_pose, last_rpy, last_gripper
                )
                
                # Control loop timing
                elapsed_time = time.time() - start_time
                # print(f"Total step time: {elapsed_time:.4f}s")
                
                sleep_time = loop_interval - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    print("Warning: Control loop running slower than desired frequency")
                
                step += 1
                
        except KeyboardInterrupt:
            print("\nControl loop interrupted by user")
        except Exception as e:
            import traceback
            print(f"An unexpected error occurred: {e}")
            traceback.print_exc()
        finally:
            print("Control loop ended")
            
            # Save video if frames were collected
            if self.save_video and len(self.main_frames) > 0:
                print(f"[Client] Saving collected main camera frames as video...")
                self.save_video_file(task_name, save_dir="/home/luka/1008/openpi/videos")
            
            # Return to initial position
            self.controller.execute_eef(execute_action, "reset")


def main():
    """Main function to initialize and run the robot controller"""
    from controller_eef import A1ArmController
    
    # Initialize controller and robot system
    controller = A1ArmController()
    robot_system = PI0RobotController(
        controller,
        save_video=True,
        video_path=None,  # Auto-generate unique filename
        video_fps=10
    )
    
    # Run control loop
    robot_system.run_control_loop(
        task_name="place_orange_in_the_plate", # 
        n_iterations=40,
        chunk_size=10,
        merge_step=1,
        loop_interval=0.1
    )


if __name__ == "__main__":
    main()
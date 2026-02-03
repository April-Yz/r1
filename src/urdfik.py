#!/usr/bin/env python3
"""
Curobo-based IK solver for R1 robot
Adapted for real-time control with trajectory tracking
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from curobo.geom.types import Pose as CuroboPose
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel


class URDFInverseKinematics:
    def __init__(self, 
                 urdf_file="/home/pine/yzj/R1_urdf/galaxea_sim/assets/r1/robot.urdf",
                 base_link="base_link",
                 ee_link="left_gripper_link"):  # or "right_gripper_link"
        
        self.tensor_args = TensorDeviceType()
        self.robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, self.tensor_args)
        
        self.ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            None,
            rotation_threshold=0.05,
            position_threshold=0.005,
            # 使用单个种子从当前关节角开始优化（提高跟踪性能）
            num_seeds=1,          
            self_collision_check=False, 
            self_collision_opt=False,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
        )

        self.ik_solver = IKSolver(self.ik_config)
        
        # 初始化 FK 模型
        self.fk_model = CudaRobotModel(self.robot_cfg.kinematics)
    
    
    # CRITICAL CHANGE 2: Add current_joints argument
    def solve_ik(self, target_position, target_orientation, current_joints=None):
        """
        Calculates IK with validation and normalization.
        :param current_joints: List or Array of current joint angles (in radians).
                               Required for smooth motion tracking.
        """
        # 1. Normalize Quaternion
        quat = np.array(target_orientation)
        norm = np.linalg.norm(quat)
        if norm > 0:
            quat = quat / norm
        else:
            # rospy.logerr("Invalid quaternion: Norm is zero")
            print("[IK] Invalid quaternion: Norm is zero")
            return None

        # Convert Target to Tensor
        target_position_tensor = torch.tensor(list(target_position), 
                                              device=self.tensor_args.device, 
                                              dtype=torch.float32)
        target_orientation_tensor = torch.tensor(list(quat), 
                                                 device=self.tensor_args.device, 
                                                 dtype=torch.float32)
        
        goal = CuroboPose(target_position_tensor, target_orientation_tensor)

        # CRITICAL CHANGE 3: Prepare the seed (initial state)
        # If we have current joints, we format them as the seed for the solver.
        # Shape must be [batch_size, num_dof], here batch is 1.
        seed_tensor = None
        if current_joints is not None:
            # Ensure it is a tensor with shape (1, DOF)
            seed_tensor = torch.tensor(current_joints, 
                                     device=self.tensor_args.device, 
                                     dtype=torch.float32).view(1, -1)

        # CRITICAL CHANGE 4: Pass the seed to solve_batch
        # This tells the solver: "Start searching FROM here"
        if seed_tensor is not None:
            result = self.ik_solver.solve_batch(goal, seed_config=seed_tensor)
        else:
            # Fallback to random initialization if no current state provided (dangerous for tracking)
            result = self.ik_solver.solve_batch(goal)
            
        torch.cuda.synchronize()

        # 2. Check Success Status
        is_success = result.success.cpu().numpy().all()
        
        # Retry logic (Retaining your original logic, but be careful with this loop in real-time control)
        original_pos_thresh = self.ik_solver.position_threshold
        original_rot_thresh = self.ik_solver.rotation_threshold

        while not is_success:
            # rospy.logwarn(f"IK convergence iteration: Pos Error: {result.position_error.cpu().numpy()[0,0]:.3f} m")
            
            self.ik_solver.position_threshold *= 5
            self.ik_solver.rotation_threshold *= 2
            
            # Pass the seed again during retry!
            if seed_tensor is not None:
                result = self.ik_solver.solve_batch(goal, seed_config=seed_tensor)
            else:
                result = self.ik_solver.solve_batch(goal)
                
            is_success = result.success.cpu().numpy().all()
            
            # Break if thresholds get absurdly high to prevent infinite loops
            if self.ik_solver.position_threshold > 0.1: 
                # rospy.logerr("IK Failed to converge even with relaxed thresholds.")
                print(f"[IK] Failed to converge (pos_err={result.position_error.cpu().numpy()[0,0]:.4f}m)")
                break

        # Reset thresholds for next run
        self.ik_solver.position_threshold = original_pos_thresh
        self.ik_solver.rotation_threshold = original_rot_thresh

        if is_success:
            # print(f"IK Converged! Final Pos Error: {result.position_error.cpu().numpy()[0,0]:.3f} m")
            return result
        else:
            return None
    
    def forward_kinematics(self, joint_angles):
        """计算正运动学（FK）
        
        Args:
            joint_angles: 关节角度数组 (numpy array or list)
            
        Returns:
            position: 3D 位置 [x, y, z] (numpy array)
            quaternion: 四元数 [w, x, y, z] (numpy array)
            euler: 欧拉角 [roll, pitch, yaw] in 'zyx' order (numpy array)
        """
        # 将关节角度转换为 tensor
        joint_tensor = torch.tensor(joint_angles, 
                                    device=self.tensor_args.device,
                                    dtype=torch.float32).view(1, -1)
        
        # 使用 CudaRobotModel 计算 FK
        kin_state = self.fk_model.get_state(joint_tensor)
        
        # 提取末端位置和姿态
        position = kin_state.ee_position.cpu().numpy()[0]  # [x, y, z]
        quaternion_wxyz = kin_state.ee_quaternion.cpu().numpy()[0]  # [w, x, y, z]
        
        # 将四元数转换为欧拉角 (zyx order)
        quat_xyzw = [quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]]
        r = R.from_quat(quat_xyzw)
        euler = r.as_euler('zyx')
        
        return position, quaternion_wxyz, euler

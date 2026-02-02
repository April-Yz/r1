#!/usr/bin/env python
import rospy
import time
import torch
import numpy as np
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from signal_arm.msg import gripper_position_control
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuroboPose
from curobo.types.robot import RobotConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

'''
author: Wenkai for class
'''

class URDFInverseKinematics:
    def __init__(self, 
                 urdf_file="/home/luka/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf",
                 base_link="base_link",
                 ee_link="arm_seg6"):
        """
        :param urdf_file: URDF文件路径
        :param base_link: 机器人基坐标系名称
        :param ee_link: 末端执行器坐标系名称
        """
        self.tensor_args = TensorDeviceType()
        
        self.robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, self.tensor_args)
        
        self.ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            None,  # 无 world config
            rotation_threshold=0.08,    # 旋转误差阈值
            position_threshold=0.01,   # 位置误差阈值
            num_seeds=10,               # 采样初始解个数
            self_collision_check=False, 
            self_collision_opt=False,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
        )

        self.ik_solver = IKSolver(self.ik_config)
    
    def solve_ik(self, target_position, target_orientation):

        target_position_tensor = torch.tensor(list(target_position), 
                                              device=self.tensor_args.device, 
                                              dtype=torch.float32)
        target_orientation_tensor = torch.tensor(list(target_orientation), 
                                                 device=self.tensor_args.device, 
                                                 dtype=torch.float32)
        
        goal = CuroboPose(target_position_tensor, target_orientation_tensor)

        start_time = time.time()
        result = self.ik_solver.solve_batch(goal)
        # print("goal:",goal)
        elapsed_time = (time.time() - start_time) * 1000  # ms
        # print(f"🌟 IK 求解耗时: {elapsed_time:.2f} ms")
        torch.cuda.synchronize()

        # print("Solved joint pose:", result.js_solution.position.cpu().numpy().flatten())
        return result

class A1ArmController:
    def __init__(self, 
                 urdf_path="/home/luka/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf",
                 base_link="base_link",
                 ee_link="arm_seg6"):

        if not rospy.core.is_initialized():
            rospy.init_node('a1_arm_controller', anonymous=True)
            
        self._ik_solver = URDFInverseKinematics(urdf_path, base_link, ee_link)
        
        self._current_gripper_position = None
        self._current_joint_position = None
        self._current_end_effector_pose = None

        self._joint_pub = rospy.Publisher('/arm_joint_target_position', JointState, queue_size=10)
        self._gripper_pub = rospy.Publisher('/gripper_position_control_host', gripper_position_control, queue_size=10)
        
        rospy.Subscriber('/gripper_stroke_host', JointState, self._gripper_state_callback)
        rospy.Subscriber('/joint_states_host', JointState, self._joint_state_callback)
        rospy.Subscriber("/end_effector_pose", PoseStamped, self._end_effector_pose_callback)

        self._urdf_path = urdf_path
        
        self._wait_for_initial_state()

    def _wait_for_initial_state(self):
        while (self._current_gripper_position is None or 
               self._current_joint_position is None) and not rospy.is_shutdown():
            rospy.sleep(0.1)

    def _gripper_state_callback(self, msg):
        if msg.position:
            self._current_gripper_position = msg.position[0]

    def _joint_state_callback(self, msg):
        if msg.position:
            self._current_joint_position = msg.position

    def _end_effector_pose_callback(self, msg):
        if msg.pose:
            self._current_end_effector_pose = msg.pose

    def execute_eef(self, action, task):
        """
        :param action:
            - 7 params: [x, y, z, roll, pitch, yaw, gripper]  list[float]
            - 8 params: [x, y, z, w, x, y, z, gripper]  list[float]
        :return: success (bool)
        """
        try:
            if len(action) not in [7, 8]:
                rospy.logerr("action must be 7 or 8 elements")
                return False

            if len(action) == 7:
                current_pos = self._current_end_effector_pose.position
                current_rotation = self._current_end_effector_pose.orientation
                current_orientation = [current_rotation.w, current_rotation.x, current_rotation.y, current_rotation.z]
                target_pos = action[:3]
                target_pos[0] = action[0]
                target_pos[1] = action[1]
                target_pos[2] = action[2]
                if target_pos[2] > 0.32:
                    print("Warning: target z too high, clipping to 0.33")
                    target_pos[2] = 0.32
                #     print("Warning: target z too low, clipping to 0.10")
                #     target_pos[2] = 0.09
                # elif target_pos[2] > 0.33:
                #     target_pos[2] = 0.33
                # elif target_pos[0] < 0.11:
                #     print("Warning: target x too low, clipping to 0.11")
                #     target_pos[0] = 0.11
                # target_pos[2] -= 0.007  # to avoid collision with table
                # elif target_pos[1] > 0.40:
                #     print("Warning: target y too high, clipping to 0.40")
                #     target_pos[1] = 0.40
                # current_rotation = euler_from_quaternion(current_orientation)
                # delta_r, delta_p, delta_y = action[3:6]
                # target_rotation = [current_rotation[0] + delta_r,
                #                    current_rotation[1],
                #                    current_rotation[2]]
                # # target_pos = action[:3]
                # print("!!!!!!!!!!!!!!!", target_rotation[0])
                # target_orientation = quaternion_from_euler(target_rotation[0],
                #                                             target_rotation[1],
                #                                             target_rotation[2])
                # fix orientation
                target_orientation = [0.00, 0.00, 1.0, 0.0]
                target_rotation = euler_from_quaternion(target_orientation)
                
                # for debug dhy
                # if action[6] < 0.5:
                #     action[6] = 0.0
                # else:
                #     action[6] = 1.0
                # action[6] = 1.0 - action[6]
                print("Inverted gripper action:", action[6])
                gripper_target = min(100, max(0, action[6] * 100))
                if 35 < gripper_target < 60:
                    gripper_target = min(35, gripper_target)
                elif gripper_target <= 35:
                    gripper_target = 20
                # if gripper_target > 80:
                #     gripper_target = 100
            else:
                target_pos = action[:3]
                target_orientation = action[3:7]
                target_rotation = euler_from_quaternion(target_orientation)

                gripper_target = min(100, max(0, action[7] * 100))

            joint_solution = self._ik_solver.solve_ik(
                target_position=target_pos, 
                target_orientation=target_orientation
            )
            # if gripper_target < 80 and task == "pick_bowl":
            #     gripper_target = 30
            # if gripper_target < 80 and task == "stack_cup":
            #     gripper_target = 0
            # if gripper_target < 80 and task == "place_fruit_bowl":
            #     gripper_target = 50
            # if gripper_target < 80 and task == "place_bread_plate":
            #     gripper_target = 10
            # if gripper_target < 80 and task == "put_vege_bowl":
            #     gripper_target = 30
            print("gripper_target", gripper_target)
            target_joint = joint_solution.js_solution.position.cpu().numpy().flatten()
            self._inter_and_pub_motion(target_joint, gripper_target)
            
            current_pos = np.array([self._current_end_effector_pose.position.x,
                                    self._current_end_effector_pose.position.y,
                                    self._current_end_effector_pose.position.z,
                                    target_rotation[0],
                                    target_rotation[1],
                                    target_rotation[2],
                                    gripper_target / 100.0
                                    ])

            return current_pos

        except Exception as e:
            rospy.logerr(f"ERROR in execution: {e}")
            return False

    def _inter_and_pub_motion(self, target_joint, gripper_target, steps=1, hz=20):

        if len(target_joint) != 6:
            rospy.logerr("joint target must be 6 elements")
            return

        rate = rospy.Rate(hz)
        
        joint_state = JointState()
        joint_state.header.frame_id = 'world'
        joint_state.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        joint_state.position = list(self._current_joint_position)

        step_increment = [
            (target - current) / steps for target, current in zip(target_joint, joint_state.position)
        ]

        start_time = time.time()
        for _ in range(steps):
            if rospy.is_shutdown():
                break
            joint_state.header.stamp = rospy.Time.now()
            joint_state.position = [
                current + increment for current, increment in zip(joint_state.position, step_increment)
            ]
            self._joint_pub.publish(joint_state)
            rate.sleep()
        # print(f"关节插值时间: {(time.time() - start_time) * 1000:.2f} ms")

        gripper_start = self._current_gripper_position
        gripper_steps = 3
        gripper_increment = (gripper_target - gripper_start) / gripper_steps
        gripper_msg = gripper_position_control()
        gripper_msg.header = Header(frame_id="")

        start_time = time.time()
        for i in range(gripper_steps + 1):
            if rospy.is_shutdown():
                break
            gripper_msg.header.stamp = rospy.Time.now()
            gripper_msg.gripper_stroke = gripper_start + gripper_increment * i
            self._gripper_pub.publish(gripper_msg)
            rate.sleep()
        # print(f"夹爪插值时间: {(time.time() - start_time) * 1000:.2f} ms")
        
if __name__ == "__main__":
    ik_solver = URDFInverseKinematics()

    # 测试 IK
    target_position = [0.5, 0.0, 0.3]  # 目标位置
    target_orientation = [0.5, 0.5, 0.5, 0.5]  # 目标姿态（四元数）
    result = ik_solver.solve_ik(target_position, target_orientation)
    result_qpos = result.js_solution.position.cpu().numpy().flatten()
    print("IK 求解结果关节角度:", result_qpos)
    
    current_joint = [0.38, 1.60, -0.92, -1.66, 0.65, 1.96]
    FK = ik_solver.robot_cfg.forward_kinematics(current_joint)
    print("FK 结果:", FK.position.cpu().numpy().flatten())
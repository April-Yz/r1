rosbag record --lz4 -O /home/pine/yzj/pour/r1_data_$(date +%Y%m%d_%H%M%S).bag \
  /left/camera/color/image_raw/compressed \
  /right/camera/color/image_raw/compressed \
  /left/camera/depth/image_rect_raw \
  /right/camera/depth/image_rect_raw \
  /hdas/camera_head/depth/depth_registered \
  /hdas/camera_head/rgb/image_rect_color/compressed \
  /hdas/feedback_arm_left_low \
  /hdas/feedback_arm_right_low \
  /hdas/feedback_gripper_left_low \
  /hdas/feedback_gripper_right_low \
  /motion_target/target_joint_state_arm_left_low \
  /motion_target/target_joint_state_arm_right_low \
  /motion_control/position_control_gripper_left_low \
  /motion_control/position_control_gripper_right_low
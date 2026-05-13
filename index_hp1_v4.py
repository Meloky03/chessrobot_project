#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
index_hp1_v4.py
===============
تحكم يدوي بذراع Franka Panda لمشروع روبوت الشطرنج.

هذا الملف بيركز فقط على:
- تهيئة MoveIt + TF + gripper client
- حلقة التحكم اليدوي (input loop)
- تنفيذ حركة الذراع لمربعات اللوحة والـgraveyard/promotion

أي كود خاص بالمعايرة أو بناء المواقع أو الـprobing موجود في
`board_calibration.py`.
"""

import sys

import rospy
import moveit_commander
import actionlib
import numpy as np

from franka_gripper.msg import MoveAction, MoveGoal

from board_calibration import (
    BoardCalibration,
    ForceMonitor,
    init_tf,
    move_to_pose,
    REFERENCE_FRAME,
    V_SLOW, A_SLOW, SAFE_H,
    OPEN_WIDTH, CLOSE_WIDTH, GRIPPER_SPEED,
)


# =====================================================================
# --- تحكم القابض ---
# =====================================================================
def gripper_control(move_client, action_type):
    goal = MoveGoal()
    if action_type == "open":
        goal.width = float(OPEN_WIDTH)
        rospy.loginfo(f"Opening to: {OPEN_WIDTH}m")
    else:
        goal.width = float(CLOSE_WIDTH)
        rospy.loginfo(f"Closing to: {CLOSE_WIDTH}m")
    goal.speed = float(GRIPPER_SPEED)
    move_client.send_goal(goal)
    move_client.wait_for_result()


# =====================================================================
# --- حلقة التحكم اليدوي ---
# =====================================================================
def manual_control():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('manual_robot_control_globals', anonymous=True)

    # --- TF listener لتحويل get_current_pose() لـworld frame ---
    init_tf()

    # --- MoveIt commander ---
    arm = moveit_commander.MoveGroupCommander(
        "panda1_manipulator",
        robot_description="/panda1/robot_description",
        ns="/panda1")

    rospy.loginfo(f"Setting pose reference frame to: {REFERENCE_FRAME}")
    arm.set_pose_reference_frame(REFERENCE_FRAME)
    rospy.loginfo(f"  planning frame (base) : {arm.get_planning_frame()}")
    rospy.loginfo(f"  pose reference frame  : {arm.get_pose_reference_frame()}")

    # --- gripper action client ---
    gripper_client = actionlib.SimpleActionClient(
        '/panda1/franka_gripper/move', MoveAction)
    gripper_client.wait_for_server()

    # --- force monitor ---
    force_monitor = ForceMonitor()
    try:
        force_monitor.wait_for_data(timeout=5.0)
    except RuntimeError as e:
        rospy.logwarn(f"ForceMonitor: {e}")

    # --- board state + محاولة تحميل آخر معايرة ---
    board = BoardCalibration()
    board.load()

    print(f"--- Manual Control (Open: {OPEN_WIDTH}, Close: {CLOSE_WIDTH}) ---")
    print(f"Home Coords: X={board.hx:.3f}, Y={board.hy:.3f}, Z={board.hz:.3f}")
    print(f"Board theta: {np.degrees(board.board_theta):+.3f} deg")
    print("Commands: a1..h8 | pq pr pb pn | x0..x23 | open | close | ready")
    print("          home | calibrate | save | load | show | test | exit")

    while not rospy.is_shutdown():
        cmd = input("Enter Target/Action: ").strip().lower()
        if cmd == 'exit':
            break
        if cmd == '':
            continue

        # --- أوامر القابض ---
        if cmd == 'open':
            gripper_control(gripper_client, "open"); continue
        if cmd == 'close':
            gripper_control(gripper_client, "close"); continue

        # --- ready / home ---
        if cmd == 'ready':
            arm.set_named_target('ready'); arm.go(wait=True); continue
        if cmd == 'home':
            move_to_pose(arm, board.hx, board.hy, board.hz, V_SLOW, A_SLOW)
            continue

        # --- المعايرة + حفظ/تحميل/عرض/اختبار ---
        if cmd in ('calibrate', 'calib'):
            ok = board.calibrate(arm, force_monitor, gripper_client)
            if ok:
                ans = input("Save calibration to YAML? [y/N]: ").strip().lower()
                if ans == 'y':
                    board.save()
            continue

        if cmd == 'save':
            board.save(); continue
        if cmd == 'load':
            board.load(); continue
        if cmd == 'show':
            board.show(); continue
        if cmd == 'test':
            board.test(arm); continue

        # --- مربع على اللوحة / promotion / graveyard ---
        target_pos = None
        if cmd in board.mirrored_squares:
            target_pos = board.mirrored_squares[cmd]
        elif (cmd.startswith('p') and len(cmd) == 2
              and cmd[1] in board.promotion_positions):
            target_pos = board.promotion_positions[cmd[1]]
        elif cmd.startswith('x'):
            try:
                idx = int(cmd[1:])
                if 0 <= idx < len(board.graveyard_positions):
                    target_pos = board.graveyard_positions[idx]
            except ValueError:
                pass

        if target_pos is not None:
            move_to_pose(arm, target_pos[0], target_pos[1], SAFE_H,
                         V_SLOW, A_SLOW)
        else:
            print("Invalid Input!")


if __name__ == '__main__':
    try:
        manual_control()
    except rospy.ROSInterruptException:
        pass

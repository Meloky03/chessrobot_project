#!/usr/bin/env python3
import rospy
import moveit_commander
import numpy as np
import sys
import copy
import actionlib
import copy, math
import moveit_msgs.msg
from geometry_msgs.msg import Pose, Quaternion
from tf.transformations import quaternion_from_euler
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal

# --- ناخذ مواقع اللوحة والزاوية والـhome من ملف المعايرة المحفوظ ---
from board_calibration_p1 import (
    BoardCalibration,
    init_tf,
    REFERENCE_FRAME,
)

travel_h = 0.18 
safe_h=0.1
pick_h = 0.047
vc=1
ac=1
vg=0.6
ag=0.6

# حالة اللوحة: تنشأ فارغة هنا وتُعبّأ من YAML عبر board.load() في main()
# بعدها: board.square_positions, board.mirrored_squares,
#         board.hx/hy/hz, board.board_theta
board = BoardCalibration()



def init_gripper_clients():
    """Create action clients for Franka gripper."""
    grasp_client = actionlib.SimpleActionClient('/panda1/franka_gripper/grasp', GraspAction)
    move_client  = actionlib.SimpleActionClient('/panda1/franka_gripper/move',  MoveAction)
    grasp_client.wait_for_server()
    move_client.wait_for_server()
    return grasp_client, move_client

def gripper_open(move_client, width=0.035, speed=0.1):
    """Open gripper to a given width (meters)."""
    goal = MoveGoal()
    goal.width = float(width)
    goal.speed = float(speed)
    move_client.send_goal(goal)
    move_client.wait_for_result()
    return move_client.get_result()
    
def gripper_close_full(move_client, speed=0.1):
    """
    Fully close the gripper (maximum closing).
    """
    goal = MoveGoal()
    goal.width = 0.0
    goal.speed = float(speed)
    move_client.send_goal(goal)
    move_client.wait_for_result()
    return move_client.get_result()




def gripper_grasp(grasp_client, width=0.015, force=8, speed=0.1,
                  eps_inner=0.005, eps_outer=0.005):
    """Close gripper with force control (Grasp)."""
    goal = GraspGoal()
    goal.width = float(width)
    goal.speed = float(speed)
    goal.force = float(force)
    goal.epsilon.inner = float(eps_inner)
    goal.epsilon.outer = float(eps_outer)

    grasp_client.send_goal(goal)
    grasp_client.wait_for_result()
    return grasp_client.get_result()
    
    







def move_to_pose(move_group, x, y, z, vf,af, cartesian=True, yaw=0.0):
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)

    q = quaternion_from_euler(np.pi, 0.0, float(yaw))
    pose.orientation = Quaternion(*q)

    if cartesian:
        waypoints = [copy.deepcopy(pose)]

        
        plan, fraction = move_group.compute_cartesian_path(
            waypoints,
            0.05,   # eef_step
            False      # jump_threshold
        )

        if fraction > 0.9:
            current_state = move_group.get_current_state()
            plan = move_group.retime_trajectory(
                current_state,
                plan,
                velocity_scaling_factor=vf,
                acceleration_scaling_factor=af,
                algorithm="iterative_time_parameterization"
            )
            move_group.execute(plan, wait=True)   
        else:
            move_group.set_pose_target(pose)
            move_group.go(wait=True)
    else:
        move_group.set_pose_target(pose)
        move_group.go(wait=True)

    move_group.stop()
    move_group.clear_pose_targets()




def arc_move(move_group,
                       sx, sy, ex, ey,
                       pick_h,
                       h_travel,           
                       arc_extra=0.2,
                       vf=1, af=1,
                       N_up=10, N_curve=40, N_down=10,
                       eef_step=0.01,
                       yaw=0.0):

    import copy, rospy
    import numpy as np
    from geometry_msgs.msg import Pose, Quaternion
    from tf.transformations import quaternion_from_euler

   
    q = quaternion_from_euler(np.pi, 0.0, float(yaw))
    ori = Quaternion(*q)

    sx, sy, ex, ey = float(sx), float(sy), float(ex), float(ey)
    pick_h = float(pick_h)
    h_travel = float(h_travel)
    arc_extra = float(arc_extra)

    def lerp(a, b, t):
        return a + (b - a) * t

    def bezier(p0, p1, p2, p3, t):
        u = 1.0 - t
        return (u*u*u)*p0 + 3*(u*u)*t*p1 + 3*u*(t*t)*p2 + (t*t*t)*p3

    waypoints = []

   
    cur = move_group.get_current_pose().pose
    cur.orientation = ori
    waypoints.append(copy.deepcopy(cur))

    
   
    z0 = float(cur.position.z)
    for i in range(max(2, int(N_up))):
        t = i / float(N_up - 1)
        p = Pose()
        p.position.x = sx
        p.position.y = sy
        p.position.z = lerp(z0, h_travel, t)
        p.orientation = ori
        waypoints.append(copy.deepcopy(p))

 
    P0 = (sx, sy, h_travel)
    P1 = (sx, sy, h_travel + arc_extra)
    P2 = (ex, ey, h_travel + arc_extra)
    P3 = (ex, ey, h_travel)

    for i in range(max(2, int(N_curve))):
        t = i / float(N_curve - 1)
        p = Pose()
        p.position.x = float(bezier(P0[0], P1[0], P2[0], P3[0], t))
        p.position.y = float(bezier(P0[1], P1[1], P2[1], P3[1], t))
        p.position.z = float(bezier(P0[2], P1[2], P2[2], P3[2], t))
        p.orientation = ori
       
        if i == 0:
            continue
        waypoints.append(copy.deepcopy(p))

  
    for i in range(max(2, int(N_down))):
        t = i / float(N_down - 1)
        p = Pose()
        p.position.x = ex
        p.position.y = ey
        p.position.z = lerp(h_travel, pick_h, t)
        p.orientation = ori
        if i == 0:
            continue
        waypoints.append(copy.deepcopy(p))

   
    move_group.set_start_state_to_current_state()
    plan, fraction = move_group.compute_cartesian_path(
        waypoints,
        float(eef_step),
        False
    )

    if fraction < 0.9:
        rospy.logwarn(f"Connected path incomplete: fraction={fraction:.2f}")
        return False

    current_state = move_group.get_current_state()
    plan = move_group.retime_trajectory(
        current_state,
        plan,
        velocity_scaling_factor=float(vf),
        acceleration_scaling_factor=float(af),
        algorithm="iterative_time_parameterization"
    )

    move_group.execute(plan, wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    return True




def move_piece(move_group, grasp_client, move_client, move_text):

    start_square = move_text[:2].lower()
    end_square = move_text[2:].lower()

    # المواقع ياتون من board (محمّل من YAML). نستخدم mirrored_squares
    # للحفاظ على نفس سلوك السكربت القديم.
    sq_map = board.mirrored_squares if board.mirrored_squares else board.square_positions
    sx, sy, _ = sq_map[start_square]
    ex, ey, _ = sq_map[end_square]

    # home + board yaw من ملف المعايرة
    hx, hy, hz = board.hx, board.hy, board.hz
    yaw = float(board.board_theta)

    rospy.loginfo(f"Executing move: {start_square} -> {end_square}")
    gripper_close_full(move_client)



    move_to_pose(move_group, hx, hy, safe_h, vc, ac, cartesian=True, yaw=yaw)
    arc_move(move_group, hx, hy, sx, sy, safe_h, travel_h, yaw=yaw)

    gripper_open(move_client)
    move_to_pose(move_group, sx, sy, pick_h, vg, ag, cartesian=True, yaw=yaw)

    gripper_grasp(grasp_client)

    rospy.sleep(0.1)


 
    arc_move(move_group, sx, sy, ex, ey, pick_h, travel_h, yaw=yaw)



   #move_to_pose(move_group, ex, ey, pick_h, vg, ag, cartesian=True, yaw=yaw)
    gripper_open(move_client)
    rospy.sleep(0.5)


    move_to_pose(move_group, ex, ey, safe_h, vg, ag, cartesian=True, yaw=yaw)
    gripper_close_full(move_client)
    arc_move(move_group, ex, ey, hx, hy, safe_h, travel_h, yaw=yaw)
    move_group.clear_pose_targets()
    #move_group.go(wait=True)
    #move_group.stop()
    
   

# ---------- Main ----------
def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('chessrobot_p2', anonymous=False)

    # TF listener لازم عشان get_current_pose يتحول الى REFERENCE_FRAME
    init_tf()

    # التعديل الجوهري هنا:
    arm = moveit_commander.MoveGroupCommander(
    "panda1_manipulator",
    robot_description="/panda1/robot_description",
    ns="/panda1"
)
    # نفس الإطار المرجعي المستخدم في المعايرة (world)
    arm.set_pose_reference_frame(REFERENCE_FRAME)
    arm.set_max_velocity_scaling_factor(1)
    arm.set_max_acceleration_scaling_factor(1)

    # تحميل المعايرة المحفوظة (corner, theta, e_h, e_N) وبناء المربعات/home
    if not board.load():
        rospy.logerr("No saved calibration found. Run board calibration first.")
        moveit_commander.roscpp_shutdown()
        return

    rospy.loginfo(f"Loaded calibration:")
    rospy.loginfo(f"  home  = ({board.hx:+.4f}, {board.hy:+.4f}, {board.hz:+.4f})")
    rospy.loginfo(f"  theta = {np.degrees(board.board_theta):+.3f} deg")

    grasp_client, move_client = init_gripper_clients()
    
    rospy.sleep(1)
    rospy.loginfo("Robot Ready. Enter moves (e.g., a2a4):")
    
    while not rospy.is_shutdown():
        try:
            move_text = input("Enter move or exit: ").strip()
            if move_text.lower() == 'exit': break
            move_piece(arm, grasp_client, move_client, move_text)
        except Exception as e:
            rospy.logerr(f"Error: {e}")
    
    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    main()

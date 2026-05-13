#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
index_hp1_v4.py
===============
تحكم يدوي بذراع Franka Panda لمشروع روبوت الشطرنج.

التغييرات عن v3
--------------
* انفصلت كل منطق المواقع (squares / promotion / graveyard / home) إلى
  موديول chess_layout.py. هذا الملف الآن مسؤول فقط عن الحركة والمعايرة.
* اللون يُستقبل من topic (/chess/color). عند تغيّر اللون، تتعاد بنية المواقع
  تلقائياً بدون إعادة المعايرة (الكاليبريشن نفسه مستقل عن اللون).
* capture / promotion / home كلها تُوضع دائماً على يسار الروبوت، والـindex
  يبدأ من الصف الأقرب للروبوت - يعني ثابتة بالنسبة للروبوت مهما كان اللون.
* الهوم = مركز المربع الأيسر الأقرب للروبوت + إزاحة home_offset لليسار.
"""

import os
import sys
import copy
import threading

import rospy
import moveit_commander
import numpy as np
import yaml
import actionlib
import tf2_ros
import tf2_geometry_msgs  # لازم للـPoseStamped.transform()

from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from tf.transformations import quaternion_from_euler
from franka_gripper.msg import MoveAction, MoveGoal
from franka_msgs.msg import FrankaState

# Local module: all layout math + color subscriber
from chess_layout import (
    BoardGeometry, LayoutConfig, ChessLayout, ColorSubscriber,
)

# =====================================================================
# --- الاطار المرجعي ---
# =====================================================================
REFERENCE_FRAME = "world"

# TF globals (تتملا في manual_control)
tf_buffer = None
tf_listener = None

# =====================================================================
# --- متغيرات التحكم العالمية ---
# =====================================================================
OPEN_WIDTH   = 0.04
CLOSE_WIDTH  = 0.025
GRIPPER_SPEED = 0.1

v_slow, a_slow = 0.7, 0.7
safe_h = 0.12

# --- ثوابت الهندسة للوحة (مرّرة للـChessLayout) ---
BOARD = BoardGeometry(square_size=0.045, board_outer_size=0.360)

# --- الاعدادات القابلة للتعديل لمواقع الـAux ---
LAYOUT_CFG = LayoutConfig(
    aux_side_gap=0.01,
    capture_cols=3,
    capture_rows=8,
    home_offset=0.10,   # <-- المتطلب الرابع: مسافة الهوم عن أقرب مربع يسار
    home_z=safe_h,
    piece_z=0.0,
)

# --- اعدادات الـprobing ---
PROBE_Z            = 0.06
PROBE_SPEED_SCALE  = 0.02
PROBE_ACCEL_SCALE  = 0.02
PROBE_MAX_TRAVEL   = 0.4
PROBE_RETREAT      = 0.020
PROBE_POST_LIFT    = 0.010
PROBE_TRANSIT_Z    = safe_h + 0.080
PROBE_FORCE_THRESH   = 4
PROBE_FORCE_THRESH_S = 4
PROBE_BIAS_SAMPLES = 50
PROBE_BIAS_RATE_HZ = 100
PROBE_CONTACT_CONSEC  = 3
PROBE_MIN_TRAVEL      = 0.005

# --- ازاحة طرف القابض ---
GRIPPER_TIP_OFFSET = (0.018 / 2) - 0.00

# --- نقاط بدء الـprobing (في world) ---
W1_START_XY = (0.50,  0.20)
W2_START_XY = (0.52,  0.20)
W_PROBE_DIR = (0.0, -1.0)
W_PROBE_YAW = np.pi / 2.0

S1_START_XY = (0.30, -0.20)
S2_START_XY = (0.30, -0.22)
S_PROBE_DIR = (+1.0, 0.0)
S_PROBE_YAW = 0.0

CALIB_FILE = os.path.expanduser("~/board_calibration.yaml")

# قيم افتراضية للـcorner / axes (تستخدم قبل المعايرة أو بعد load)
origin_x, origin_y, origin_z = 0.3981098174137588, -6.41297277950631e-05, 0.0
board_corner = np.array([
    origin_x + 7.5 * BOARD.square_size + BOARD.margin,
    origin_y - 7.5 * BOARD.square_size - BOARD.margin,
])
e_h_axis = np.array([0.0,  1.0])
e_N_axis = np.array([-1.0, 0.0])
board_theta = 0.0

# Layout الحالي (يتعاد بناؤه عند المعايرة أو تغير اللون)
_layout: ChessLayout = None
_layout_lock = threading.Lock()


# =====================================================================
# --- Layout management ---
# =====================================================================
def rebuild_layout(color: str):
    """يعيد بناء ChessLayout بناءً على اللون + الكاليبريشن الحالي."""
    global _layout
    new_layout = ChessLayout(
        color=color,
        corner=board_corner,
        e_h=e_h_axis,
        e_N=e_N_axis,
        board=BOARD,
        config=LAYOUT_CFG,
    )
    with _layout_lock:
        _layout = new_layout
    rospy.loginfo(f"[layout] rebuilt for color={color!r}")
    rospy.loginfo(
        f"  a1 (logical) -> {new_layout.squares['a1']}, "
        f"home -> {new_layout.home}")


def get_layout() -> ChessLayout:
    with _layout_lock:
        return _layout


# =====================================================================
# --- مراقب القوة ---
# =====================================================================
class ForceMonitor:
    def __init__(self, topic="/panda1/franka_state_controller/franka_states"):
        self._lock = threading.Lock()
        self._force_xyz = np.zeros(3)
        self._bias_xyz  = np.zeros(3)
        self._got_msg   = False
        self._sub = rospy.Subscriber(topic, FrankaState, self._cb, queue_size=1)

    def _cb(self, msg):
        f = msg.O_F_ext_hat_K
        with self._lock:
            self._force_xyz = np.array([f[0], f[1], f[2]], dtype=float)
            self._got_msg = True

    def wait_for_data(self, timeout=5.0):
        start = rospy.Time.now()
        rate = rospy.Rate(50)
        while not self._got_msg and not rospy.is_shutdown():
            if (rospy.Time.now() - start).to_sec() > timeout:
                raise RuntimeError("No FrankaState message received (topic?)")
            rate.sleep()

    def zero_bias(self, n_samples=None, rate_hz=None):
        if n_samples is None: n_samples = PROBE_BIAS_SAMPLES
        if rate_hz   is None: rate_hz   = PROBE_BIAS_RATE_HZ
        rate = rospy.Rate(rate_hz)
        samples = []
        for _ in range(n_samples):
            with self._lock:
                samples.append(self._force_xyz.copy())
            rate.sleep()
        self._bias_xyz = np.mean(samples, axis=0)
        rospy.loginfo(f"  [ForceMonitor] bias = "
                      f"({self._bias_xyz[0]:+.2f}, {self._bias_xyz[1]:+.2f}, "
                      f"{self._bias_xyz[2]:+.2f}) N")

    def get_xy_magnitude(self):
        with self._lock:
            f = self._force_xyz - self._bias_xyz
        return float(np.hypot(f[0], f[1]))


# =====================================================================
# --- دوال الحركة ---
# =====================================================================
def _rot_2d(v, angle):
    """دوران متجه 2D بزاوية angle (CCW) حول نقطة الأصل."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def get_current_pose_in_ref(move_group, timeout=1.0):
    """ترجع الـcurrent pose في REFERENCE_FRAME."""
    ps_planning = move_group.get_current_pose()
    if tf_buffer is None:
        return ps_planning.pose
    ps_planning.header.stamp = rospy.Time(0)
    try:
        ps_ref = tf_buffer.transform(ps_planning, REFERENCE_FRAME,
                                     rospy.Duration(timeout))
        return ps_ref.pose
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"get_current_pose_in_ref: TF transform failed: {e}")
        return ps_planning.pose


def _make_down_pose(x, y, z, yaw):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = float(x), float(y), float(z)
    q = quaternion_from_euler(np.pi, 0.0, float(yaw))
    pose.orientation = Quaternion(*q)
    return pose


def move_to_pose(move_group, x, y, z, vf, af, yaw=None):
    if yaw is None:
        yaw = board_theta
    pose = _make_down_pose(x, y, z, yaw)
    waypoints = [copy.deepcopy(pose)]
    plan, fraction = move_group.compute_cartesian_path(waypoints, 0.05, False)
    if fraction > 0.9:
        plan = move_group.retime_trajectory(
            move_group.get_current_state(), plan, vf, af,
            "iterative_time_parameterization")
        move_group.execute(plan, wait=True)
    move_group.stop()
    move_group.clear_pose_targets()


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


def gripper_full_close(move_client, speed=None):
    goal = MoveGoal()
    goal.width = 0.0
    goal.speed = float(GRIPPER_SPEED if speed is None else speed)
    move_client.send_goal(goal)
    move_client.wait_for_result()


# =====================================================================
# --- محرك اللمس ---
# =====================================================================
def probe_linear(move_group, force_monitor, direction_xy,
                 max_travel=PROBE_MAX_TRAVEL,
                 force_threshold=PROBE_FORCE_THRESH):
    cur = get_current_pose_in_ref(move_group)
    start = np.array([cur.position.x, cur.position.y])
    z = cur.position.z

    d = np.asarray(direction_xy, dtype=float)
    d = d / np.linalg.norm(d)
    target_xy = start + d * max_travel

    target_pose = copy.deepcopy(cur)
    target_pose.position.x = float(target_xy[0])
    target_pose.position.y = float(target_xy[1])
    target_pose.position.z = float(z)

    plan, fraction = move_group.compute_cartesian_path(
        [target_pose], 0.005, False)
    if fraction < 0.9:
        rospy.logerr(f"  [probe] cartesian plan failed: fraction={fraction:.2f}")
        return None

    plan = move_group.retime_trajectory(
        move_group.get_current_state(), plan,
        PROBE_SPEED_SCALE, PROBE_ACCEL_SCALE,
        "iterative_time_parameterization")

    rospy.loginfo("  [probe] zeroing force bias ...")
    force_monitor.zero_bias()

    rospy.loginfo(f"  [probe] moving: dir=({d[0]:+.2f},{d[1]:+.2f}), "
                  f"max={max_travel*1000:.0f}mm, thresh={force_threshold:.1f}N")
    move_group.execute(plan, wait=False)

    rate = rospy.Rate(200)
    contact = False
    consec = 0
    t0 = rospy.Time.now()
    timeout = 30.0
    while not rospy.is_shutdown() and (rospy.Time.now() - t0).to_sec() < timeout:
        fmag = force_monitor.get_xy_magnitude()
        cur_now = get_current_pose_in_ref(move_group)
        traveled = np.hypot(cur_now.position.x - start[0],
                            cur_now.position.y - start[1])
        if fmag > force_threshold and traveled > PROBE_MIN_TRAVEL:
            consec += 1
            if consec >= PROBE_CONTACT_CONSEC:
                move_group.stop()
                rospy.loginfo(f"  [probe] CONTACT! |F_xy|={fmag:.2f} N "
                              f"traveled={traveled*1000:.1f}mm")
                contact = True
                break
        else:
            consec = 0
        if np.hypot(target_xy[0] - cur_now.position.x,
                    target_xy[1] - cur_now.position.y) < 0.002:
            rospy.logwarn("  [probe] reached max_travel without contact")
            break
        rate.sleep()

    move_group.stop()
    move_group.clear_pose_targets()
    if not contact:
        return None

    rospy.sleep(0.25)
    final = get_current_pose_in_ref(move_group)
    return np.array([final.position.x, final.position.y])


def _do_probe(arm, force_monitor, gripper_client,
              name, start_xy, direction, probe_yaw,
              force_threshold=PROBE_FORCE_THRESH):
    rospy.loginfo(f"[Probe] {name}")
    if gripper_client is not None:
        rospy.loginfo("  [probe] full-closing gripper before probe")
        gripper_full_close(gripper_client)

    cur = get_current_pose_in_ref(arm)
    move_to_pose(arm, cur.position.x, cur.position.y, PROBE_TRANSIT_Z,
                 v_slow, a_slow, yaw=probe_yaw)

    rospy.loginfo(f"  move -> start ({start_xy[0]:+.4f}, {start_xy[1]:+.4f}) "
                  f"@ Z={PROBE_TRANSIT_Z:.3f} yaw={np.degrees(probe_yaw):+.2f}deg")
    move_to_pose(arm, start_xy[0], start_xy[1], PROBE_TRANSIT_Z,
                 v_slow, a_slow, yaw=probe_yaw)

    rospy.loginfo(f"  descend -> Z={PROBE_Z:.3f}")
    move_to_pose(arm, start_xy[0], start_xy[1], PROBE_Z,
                 v_slow, a_slow, yaw=probe_yaw)

    contact = probe_linear(arm, force_monitor, direction,
                           max_travel=PROBE_MAX_TRAVEL,
                           force_threshold=force_threshold)
    if contact is None:
        rospy.logerr(f"{name}: no contact detected within "
                     f"{PROBE_MAX_TRAVEL*1000:.0f}mm.")
        cur = get_current_pose_in_ref(arm)
        move_to_pose(arm, cur.position.x, cur.position.y, PROBE_TRANSIT_Z,
                     v_slow, a_slow, yaw=probe_yaw)
        return None

    rospy.loginfo(f"  contact @ ({contact[0]:+.4f}, {contact[1]:+.4f})")

    d_norm = np.asarray(direction, dtype=float)
    d_norm = d_norm / np.linalg.norm(d_norm)
    back_xy = contact - d_norm * PROBE_RETREAT
    rospy.loginfo(f"  retreat -> ({back_xy[0]:+.4f}, {back_xy[1]:+.4f})")
    move_to_pose(arm, back_xy[0], back_xy[1], PROBE_Z,
                 v_slow, a_slow, yaw=probe_yaw)

    lift_z = PROBE_Z + PROBE_POST_LIFT
    move_to_pose(arm, back_xy[0], back_xy[1], lift_z,
                 v_slow, a_slow, yaw=probe_yaw)

    move_to_pose(arm, back_xy[0], back_xy[1], PROBE_TRANSIT_Z,
                 v_slow, a_slow, yaw=probe_yaw)

    return contact


# =====================================================================
# --- معايرة اللوحة (Two-pass) ---
# =====================================================================
def calibrate_board(arm, force_monitor, gripper_client=None,
                    current_color: str = 'white'):
    global board_corner, board_theta, e_h_axis, e_N_axis

    rospy.loginfo("=" * 62)
    rospy.loginfo("Starting board calibration (two-pass)")
    rospy.loginfo(f"  probe height Z   : {PROBE_Z:.3f} m")
    rospy.loginfo(f"  transit height Z : {PROBE_TRANSIT_Z:.3f} m")
    rospy.loginfo(f"  post-contact lift: {PROBE_POST_LIFT*1000:.0f} mm")
    rospy.loginfo(f"  probe speed      : {PROBE_SPEED_SCALE*100:.0f}% of max")
    rospy.loginfo(f"  force threshold  : {PROBE_FORCE_THRESH:.1f} N")
    rospy.loginfo(f"  max travel       : {PROBE_MAX_TRAVEL*1000:.0f} mm")
    rospy.loginfo("=" * 62)

    ans = input("Make sure the workspace is CLEAR. Proceed? [y/N]: ").strip().lower()
    if ans != 'y':
        rospy.loginfo("Calibration aborted by user.")
        return False

    rospy.loginfo(">>> PASS 1: estimating board angle from W1 & W2")

    W1_p1 = _do_probe(arm, force_monitor, gripper_client,
                      "W1 (pass1)", W1_START_XY, W_PROBE_DIR, W_PROBE_YAW)
    if W1_p1 is None:
        return False
    W2_p1 = _do_probe(arm, force_monitor, gripper_client,
                      "W2 (pass1)", W2_START_XY, W_PROBE_DIR, W_PROBE_YAW)
    if W2_p1 is None:
        return False

    dW = W2_p1 - W1_p1
    theta_est = float(np.arctan2(dW[1], dW[0]))
    rospy.loginfo(f"  W2 - W1 = ({dW[0]:+.4f}, {dW[1]:+.4f})")
    rospy.loginfo(f"  Estimated board theta = {np.degrees(theta_est):+.3f} deg")

    rospy.loginfo(">>> PASS 2: re-probing with yaw/dir rotated by theta")

    W_dir_rot  = _rot_2d(np.array(W_PROBE_DIR, dtype=float), theta_est)
    S_dir_rot  = _rot_2d(np.array(S_PROBE_DIR, dtype=float), theta_est)
    W_yaw_corr = W_PROBE_YAW + theta_est
    S_yaw_corr = S_PROBE_YAW + theta_est

    probes_p2 = [
        ("W1 (pass2)", W1_START_XY, W_dir_rot, W_yaw_corr, PROBE_FORCE_THRESH),
        ("W2 (pass2)", W2_START_XY, W_dir_rot, W_yaw_corr, PROBE_FORCE_THRESH),
        ("S1 (pass2)", S1_START_XY, S_dir_rot, S_yaw_corr, PROBE_FORCE_THRESH_S),
        ("S2 (pass2)", S2_START_XY, S_dir_rot, S_yaw_corr, PROBE_FORCE_THRESH_S),
    ]

    contacts = []
    probe_dirs = []
    for name, start_xy, direction, probe_yaw, thresh in probes_p2:
        contact = _do_probe(arm, force_monitor, gripper_client,
                            name, start_xy, direction, probe_yaw,
                            force_threshold=thresh)
        if contact is None:
            rospy.logerr(f"{name}: aborting calibration.")
            return False
        contacts.append(contact)
        probe_dirs.append(direction / np.linalg.norm(direction))

    W1, W2, S1, S2 = contacts
    eN_meas = (W2 - W1) / np.linalg.norm(W2 - W1)
    eh_meas = (S2 - S1) / np.linalg.norm(S2 - S1)

    dot = float(np.dot(eh_meas, eN_meas))
    perp_err_deg = abs(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))) - 90.0)
    rospy.loginfo(f"  perpendicularity error = {perp_err_deg:.2f} deg")

    eN_as_eh = np.array([eN_meas[1], -eN_meas[0]])
    eh_avg = (eh_meas + eN_as_eh) / 2.0
    eh_avg = eh_avg / np.linalg.norm(eh_avg)
    eN_fix = np.array([-eh_avg[1], eh_avg[0]])

    rospy.loginfo(f"Applying gripper tip offset = "
                  f"{GRIPPER_TIP_OFFSET*1000:.1f} mm to contacts")
    contacts_corr = [c + d * GRIPPER_TIP_OFFSET
                     for c, d in zip(contacts, probe_dirs)]
    W1, W2, S1, S2 = contacts_corr

    eN_meas = (W2 - W1) / np.linalg.norm(W2 - W1)
    eh_meas = (S2 - S1) / np.linalg.norm(S2 - S1)
    eN_as_eh = np.array([eN_meas[1], -eN_meas[0]])
    eh_avg = (eh_meas + eN_as_eh) / 2.0
    eh_avg = eh_avg / np.linalg.norm(eh_avg)
    eN_fix = np.array([-eh_avg[1], eh_avg[0]])

    A = np.column_stack([eN_fix, -eh_avg])
    b = S1 - W1
    try:
        uv = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        rospy.logerr("Line intersection failed (axes nearly parallel).")
        return False
    corner_meas = W1 + uv[0] * eN_fix

    theta_meas = float(np.arctan2(eh_avg[0], -eh_avg[1]))

    board_corner = corner_meas
    e_h_axis     = eh_avg
    e_N_axis     = eN_fix
    board_theta  = theta_meas

    rospy.loginfo("=" * 62)
    rospy.loginfo("Calibration complete.")
    rospy.loginfo(f"  theta_final = {np.degrees(theta_meas):+.3f} deg")
    rospy.loginfo(f"  corner      = ({corner_meas[0]:+.4f}, {corner_meas[1]:+.4f})")
    rospy.loginfo("=" * 62)

    rebuild_layout(current_color)
    return True


# =====================================================================
# --- حفظ / تحميل المعايرة ---
# =====================================================================
def save_calibration(path=CALIB_FILE):
    data = {
        'frame_id':        REFERENCE_FRAME,
        'corner_x':        float(board_corner[0]),
        'corner_y':        float(board_corner[1]),
        'theta_rad':       float(board_theta),
        'e_h_x':           float(e_h_axis[0]),
        'e_h_y':           float(e_h_axis[1]),
        'e_N_x':           float(e_N_axis[0]),
        'e_N_y':           float(e_N_axis[1]),
        'square_size':     float(BOARD.square_size),
        'margin':          float(BOARD.margin),
        'board_outer_size':float(BOARD.board_outer_size),
    }
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False)
    rospy.loginfo(f"Calibration saved -> {path}")


def load_calibration(path=CALIB_FILE, current_color: str = 'white'):
    global board_corner, board_theta, e_h_axis, e_N_axis
    if not os.path.exists(path):
        rospy.logwarn(f"No calibration file at {path}")
        return False
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    file_frame = data.get('frame_id')
    if file_frame is not None and file_frame != REFERENCE_FRAME:
        rospy.logerr(f"Calibration frame mismatch: file={file_frame!r}, "
                     f"code={REFERENCE_FRAME!r}. Aborting load.")
        return False

    board_corner = np.array([data['corner_x'], data['corner_y']])
    board_theta  = float(data['theta_rad'])
    e_h_axis     = np.array([data['e_h_x'], data['e_h_y']])
    e_N_axis     = np.array([data['e_N_x'], data['e_N_y']])
    rebuild_layout(current_color)
    rospy.loginfo(f"Calibration loaded <- {path}")
    rospy.loginfo(f"  corner=({board_corner[0]:+.4f}, {board_corner[1]:+.4f}), "
                  f"theta={np.degrees(board_theta):+.3f} deg")
    return True


def show_calibration():
    print("--- Current calibration ---")
    print(f"  corner              : "
          f"({board_corner[0]:+.4f}, {board_corner[1]:+.4f}) m")
    print(f"  theta               : "
          f"{np.degrees(board_theta):+.3f} deg")
    print(f"  e_h                 : "
          f"({e_h_axis[0]:+.5f}, {e_h_axis[1]:+.5f})")
    print(f"  e_N                 : "
          f"({e_N_axis[0]:+.5f}, {e_N_axis[1]:+.5f})")
    layout = get_layout()
    if layout is not None:
        print(layout.describe())


def test_calibration(arm):
    layout = get_layout()
    if layout is None:
        rospy.logerr("No layout built yet.")
        return
    for sq in ('a1', 'h1', 'h8', 'a8'):
        p = layout.squares[sq]
        rospy.loginfo(f"  test -> {sq} @ ({p[0]:+.4f}, {p[1]:+.4f})")
        move_to_pose(arm, p[0], p[1], safe_h, v_slow, a_slow)
        rospy.sleep(0.3)
    rospy.loginfo("  test -> home")
    move_to_pose(arm, layout.home[0], layout.home[1], layout.home[2],
                 v_slow, a_slow)


# =====================================================================
# --- حلقة التحكم اليدوي ---
# =====================================================================
def manual_control():
    global tf_buffer, tf_listener

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('manual_robot_control_globals', anonymous=True)

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.loginfo(f"Waiting for TF: panda1_link0 <-> {REFERENCE_FRAME} ...")
    if tf_buffer.can_transform(REFERENCE_FRAME, "panda1_link0",
                               rospy.Time(0), rospy.Duration(5.0)):
        rospy.loginfo(f"  TF available: panda1_link0 -> {REFERENCE_FRAME}")
    else:
        rospy.logerr(f"  TF NOT available: {REFERENCE_FRAME} <-> panda1_link0")
        rospy.logerr("  Check static_transform_publisher in the launch file.")

    arm = moveit_commander.MoveGroupCommander(
        "panda1_manipulator",
        robot_description="/panda1/robot_description",
        ns="/panda1")
    arm.set_pose_reference_frame(REFERENCE_FRAME)
    rospy.loginfo(f"  planning frame (base): {arm.get_planning_frame()}")
    rospy.loginfo(f"  pose reference frame : {arm.get_pose_reference_frame()}")

    gripper_client = actionlib.SimpleActionClient(
        '/panda1/franka_gripper/move', MoveAction)
    gripper_client.wait_for_server()

    force_monitor = ForceMonitor()
    try:
        force_monitor.wait_for_data(timeout=5.0)
    except RuntimeError as e:
        rospy.logwarn(f"ForceMonitor: {e}")

    # --- لون القطع من topic ---
    color_sub = ColorSubscriber(topic='/chess/color', initial='white')
    rospy.loginfo("Waiting up to 2.0s for /chess/color ...")
    got = color_sub.wait_for_message(timeout=2.0)
    if not got:
        rospy.logwarn("No /chess/color message received, using default 'white'.")

    # rebuild layout ON color change
    def _on_color_change(new_color: str):
        rospy.loginfo(f"[color] changed -> {new_color!r}, rebuilding layout")
        rebuild_layout(new_color)
    color_sub.set_on_change(_on_color_change)

    # اول بناء للـlayout (باللون الحالي)
    rebuild_layout(color_sub.get())

    # تحميل الكاليبريشن المحفوظة (سيعيد بناء الـlayout باللون الحالي)
    load_calibration(current_color=color_sub.get())

    layout = get_layout()
    print(f"--- Manual Control (Open: {OPEN_WIDTH}, Close: {CLOSE_WIDTH}) ---")
    print(f"Color: {color_sub.get()}")
    print(f"Home Coords: X={layout.home[0]:.3f}, "
          f"Y={layout.home[1]:.3f}, Z={layout.home[2]:.3f}")
    print(f"Board theta: {np.degrees(board_theta):+.3f} deg")
    print("Commands: a1..h8 | pq pr pb pn | x0..x23 | open | close | ready")
    print("          home | calibrate | save | load | show | test | color | exit")

    while not rospy.is_shutdown():
        cmd = input("Enter Target/Action: ").strip().lower()
        if cmd == 'exit': break
        if cmd == '':    continue

        layout = get_layout()

        if cmd == 'open':
            gripper_control(gripper_client, "open"); continue
        if cmd == 'close':
            gripper_control(gripper_client, "close"); continue

        if cmd == 'ready':
            arm.set_named_target('ready'); arm.go(wait=True); continue

        if cmd == 'home':
            move_to_pose(arm, layout.home[0], layout.home[1], layout.home[2],
                         v_slow, a_slow)
            continue

        if cmd in ('calibrate', 'calib'):
            ok = calibrate_board(arm, force_monitor, gripper_client,
                                 current_color=color_sub.get())
            if ok:
                ans = input("Save calibration to YAML? [y/N]: ").strip().lower()
                if ans == 'y':
                    save_calibration()
            continue

        if cmd == 'save':
            save_calibration(); continue
        if cmd == 'load':
            load_calibration(current_color=color_sub.get()); continue
        if cmd == 'show':
            show_calibration(); continue
        if cmd == 'test':
            test_calibration(arm); continue
        if cmd == 'color':
            print(f"  current color = {color_sub.get()}, "
                  f"received_msg={color_sub.has_received()}")
            continue

        target_pos = None
        if cmd in layout.squares:
            target_pos = layout.squares[cmd]
        elif cmd.startswith('p') and len(cmd) == 2 and cmd[1] in layout.promotion:
            target_pos = layout.promotion[cmd[1]]
        elif cmd.startswith('x'):
            try:
                idx = int(cmd[1:])
                if 0 <= idx < len(layout.graveyard):
                    target_pos = layout.graveyard[idx]
            except ValueError:
                pass

        if target_pos is not None:
            move_to_pose(arm, target_pos[0], target_pos[1], safe_h,
                         v_slow, a_slow)
        else:
            print("Invalid Input!")


if __name__ == '__main__':
    try:
        manual_control()
    except rospy.ROSInterruptException:
        pass

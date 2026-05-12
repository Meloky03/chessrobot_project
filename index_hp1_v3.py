#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
manual_robot_control.py
=======================
تحكم يدوي بذراع Franka Panda لمشروع روبوت الشطرنج.
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
 
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Quaternion
from tf.transformations import quaternion_from_euler
from franka_gripper.msg import MoveAction, MoveGoal
from franka_msgs.msg import FrankaState
 
# =====================================================================
# --- متغيرات التحكم العالمية ---
# =====================================================================
OPEN_WIDTH = 0.04
CLOSE_WIDTH = 0.025
GRIPPER_SPEED = 0.1
 
v_slow, a_slow = 0.7, 0.7
safe_h = 0.12
 
# --- ثوابت الهندسة للوحة ---
BOARD_OUTER_SIZE = 0.360
PLAY_AREA_SIZE   = 0.360
MARGIN           = (BOARD_OUTER_SIZE - PLAY_AREA_SIZE)/2.0  # 0.030m
 
# --- اعدادات الـprobing ---
PROBE_Z            = 0.06
PROBE_SPEED_SCALE  = 0.02
PROBE_ACCEL_SCALE  = 0.02
PROBE_MAX_TRAVEL   = 0.4
PROBE_RETREAT      = 0.020
PROBE_POST_LIFT    = 0.010
PROBE_TRANSIT_Z    = safe_h + 0.080
PROBE_FORCE_THRESH = 5.0
PROBE_BIAS_SAMPLES = 50
PROBE_BIAS_RATE_HZ = 100
PROBE_CONTACT_CONSEC  = 3       # عدد العينات المتتالية فوق العتبة لتأكيد اللمس
PROBE_MIN_TRAVEL      = 0.005   # أقل مسافة (م) قبل قبول أي contact (لتجاهل spike البداية)
 
# --- ازاحة طرف القابض عن مركز TCP (نصف عرض الاصبع باتجاه اللمس) ---
# هذه المسافة بين مركز الجريبر وحافته الخارجية التي تلمس اللوحة.
# تُستخدم لتصحيح نقاط التلامس بعد اجراء المعايرة.
GRIPPER_TIP_OFFSET = 0.018/2
 
# --- نقاط بدء الـprobing ---
W1_START_XY = (0.50,  0.20)
W2_START_XY = (0.60,  0.20)
W_PROBE_DIR = (0.0, -1.0)
W_PROBE_YAW = np.pi / 2.0
 
S1_START_XY = (0.30, -0.20)
S2_START_XY = (0.30, -0.30)
S_PROBE_DIR = (+1.0, 0.0)
S_PROBE_YAW = 0.0
 
CALIB_FILE = os.path.expanduser("~/board_calibration.yaml")
 
origin_x, origin_y, origin_z = 0.3981098174137588, -6.41297277950631e-05, 0
square_size = 0.045
 
# =====================================================================
# --- حالة المعايرة ---
# =====================================================================
board_corner = np.array([
    origin_x + 7.5*square_size + MARGIN,
    origin_y - 7.5*square_size - MARGIN,
])
e_h_axis = np.array([0.0, 1.0])
e_N_axis = np.array([-1.0, 0.0])
board_theta = 0.0
 
square_positions    = {}
mirrored_squares    = {}
promotion_positions = {}
graveyard_positions = []
hx = hy = hz = 0.0
 
# =====================================================================
# --- بناء خرائط المواقع ---
# =====================================================================
 
def build_positions(corner=None, eh=None, eN=None):
    global square_positions, mirrored_squares, promotion_positions, graveyard_positions
    global hx, hy, hz
 
    if corner is None: corner = board_corner
    if eh    is None: eh    = e_h_axis
    if eN    is None: eN    = e_N_axis
 
    corner = np.asarray(corner, dtype=float)
    eh     = np.asarray(eh,     dtype=float)
    eN     = np.asarray(eN,     dtype=float)
 
    sq_pos = {}
    for col_idx, col in enumerate('abcdefgh'):
        for row in range(1, 9):
            u = MARGIN + (col_idx + 0.5) * square_size
            v = MARGIN + (row - 1 + 0.5) * square_size
            p = corner + u * eh + v * eN
            sq_pos[f"{col}{row}"] = (float(p[0]), float(p[1]), origin_z)
 
    u_promo = MARGIN - 0.01 - 3.5 * square_size
    v_by_letter = {
        'q': MARGIN + 7.5 * square_size,
        'r': MARGIN + 6.5 * square_size,
        'b': MARGIN + 5.5 * square_size,
        'n': MARGIN + 4.5 * square_size,
    }
    promo = {}
    for letter, v in v_by_letter.items():
        p = corner + u_promo * eh + v * eN
        promo[letter] = (float(p[0]), float(p[1]), origin_z)
 
    graves = []
    for col_g in range(3):
        u = MARGIN - 0.01 - (col_g + 0.5) * square_size
        for row_g in range(8):
            v = MARGIN + (7.5 - row_g) * square_size
            p = corner + u * eh + v * eN
            graves.append((float(p[0]), float(p[1]), origin_z))
 
    square_positions    = sq_pos
    promotion_positions = promo
    graveyard_positions = graves
 
    original_keys = list(square_positions.keys())
    reversed_keys = list(reversed(original_keys))
    mirrored_squares = {original_keys[i]: square_positions[reversed_keys[i]]
                        for i in range(len(original_keys))}
 
    base_hx, base_hy, _ = square_positions["h8"]
    hx = base_hx - 0.1
    hy = base_hy + 0.3
    hz = safe_h
 
build_positions()
 
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
        if rate_hz  is None: rate_hz  = PROBE_BIAS_RATE_HZ
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
    return np.array([c*v[0] - s*v[1], s*v[0] + c*v[1]])

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
    """
    يسكر القابض لأقصى حد ممكن (width=0) عشان لما يلامس اللوحة أثناء
    الـprobing ما ينفتح. يُنادى قبل كل probe.
    """
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
    cur = move_group.get_current_pose().pose
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
        cur_now = move_group.get_current_pose().pose
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
        if np.hypot(target_xy[0]-cur_now.position.x,
                    target_xy[1]-cur_now.position.y) < 0.002:
            rospy.logwarn("  [probe] reached max_travel without contact")
            break
        rate.sleep()
 
    move_group.stop()
    move_group.clear_pose_targets()
    if not contact:
        return None
 
    rospy.sleep(0.25)
    final = move_group.get_current_pose().pose
    return np.array([final.position.x, final.position.y])
 
# =====================================================================
# --- دالة مساعدة: تنفيذ probe كاملة (approach + touch + retreat) ---
# =====================================================================
def _do_probe(arm, force_monitor, gripper_client,
              name, start_xy, direction, probe_yaw):
    """
    تسكّر الجريبر، ترفع، تروح لـstart، تنزل، تلامس، بعدين تتراجع وترفع.
    ترجع نقطة التلامس (TCP) أو None لو فشلت.
    """
    rospy.loginfo(f"[Probe] {name}")

    if gripper_client is not None:
        rospy.loginfo("  [probe] full-closing gripper before probe")
        gripper_full_close(gripper_client)

    cur = arm.get_current_pose().pose
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
                           max_travel=PROBE_MAX_TRAVEL)
    if contact is None:
        rospy.logerr(f"{name}: no contact detected within "
                     f"{PROBE_MAX_TRAVEL*1000:.0f}mm.")
        cur = arm.get_current_pose().pose
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
def calibrate_board(arm, force_monitor, gripper_client=None):
    global board_corner, board_theta, e_h_axis, e_N_axis
 
    rospy.loginfo("="*62)
    rospy.loginfo("Starting board calibration (two-pass)")
    rospy.loginfo(f"  probe height Z   : {PROBE_Z:.3f} m")
    rospy.loginfo(f"  transit height Z : {PROBE_TRANSIT_Z:.3f} m")
    rospy.loginfo(f"  post-contact lift: {PROBE_POST_LIFT*1000:.0f} mm")
    rospy.loginfo(f"  probe speed      : {PROBE_SPEED_SCALE*100:.0f}% of max")
    rospy.loginfo(f"  force threshold  : {PROBE_FORCE_THRESH:.1f} N")
    rospy.loginfo(f"  max travel       : {PROBE_MAX_TRAVEL*1000:.0f} mm")
    rospy.loginfo("="*62)
 
    ans = input("Make sure the workspace is CLEAR. Proceed? [y/N]: ").strip().lower()
    if ans != 'y':
        rospy.loginfo("Calibration aborted by user.")
        return False

    # =================================================================
    # PASS 1: W1 + W2 فقط بـyaw الاصلي لتقدير زاوية اللوحة θ
    # =================================================================
    rospy.loginfo(">>> PASS 1: estimating board angle from W1 & W2")

    W1_p1 = _do_probe(arm, force_monitor, gripper_client,
                      "W1 (pass1)", W1_START_XY, W_PROBE_DIR, W_PROBE_YAW)
    if W1_p1 is None:
        return False

    W2_p1 = _do_probe(arm, force_monitor, gripper_client,
                      "W2 (pass1)", W2_START_XY, W_PROBE_DIR, W_PROBE_YAW)
    if W2_p1 is None:
        return False

    # تقدير θ: الحافة الغربية (W1->W2) موازية لمحور اللوحة e_N.
    # لما اللوحة مثالية (θ=0)، e_N = (-1,0) ، يعني W1->W2 بالـ+X.
    # الفرق (dW) بيعطينا اتجاه المحور المقاس، وبمقارنة مع +X نحصل على θ.
    dW = W2_p1 - W1_p1
    theta_est = float(np.arctan2(dW[1], dW[0]))
    rospy.loginfo(f"  W2 - W1 = ({dW[0]:+.4f}, {dW[1]:+.4f})")
    rospy.loginfo(f"  Estimated board theta = {np.degrees(theta_est):+.3f} deg")

    # =================================================================
    # PASS 2: الرجوع لـW1 start وإعادة كل الخطوات بـyaw مصحّح بـθ
    # كده الجريبر يلف من yaw الحالي (W_PROBE_YAW) لـ(W_PROBE_YAW+θ)
    # يعني θ فقط، مش 180°.
    # اتجاه الحركة (probe_dir) بيتدور كمان بـθ ليكون عامودي فعلاً على الحافة.
    # =================================================================
    rospy.loginfo(">>> PASS 2: re-probing with yaw/dir rotated by theta")

    W_dir_rot  = _rot_2d(np.array(W_PROBE_DIR, dtype=float), theta_est)
    S_dir_rot  = _rot_2d(np.array(S_PROBE_DIR, dtype=float), theta_est)
    W_yaw_corr = W_PROBE_YAW + theta_est
    S_yaw_corr = S_PROBE_YAW + theta_est

    rospy.loginfo(f"  W: yaw={np.degrees(W_yaw_corr):+.2f}deg, "
                  f"dir=({W_dir_rot[0]:+.3f},{W_dir_rot[1]:+.3f})")
    rospy.loginfo(f"  S: yaw={np.degrees(S_yaw_corr):+.2f}deg, "
                  f"dir=({S_dir_rot[0]:+.3f},{S_dir_rot[1]:+.3f})")

    probes_p2 = [
        ("W1 (pass2)", W1_START_XY, W_dir_rot, W_yaw_corr),
        ("W2 (pass2)", W2_START_XY, W_dir_rot, W_yaw_corr),
        ("S1 (pass2)", S1_START_XY, S_dir_rot, S_yaw_corr),
        ("S2 (pass2)", S2_START_XY, S_dir_rot, S_yaw_corr),
    ]

    contacts = []
    probe_dirs = []
    for name, start_xy, direction, probe_yaw in probes_p2:
        contact = _do_probe(arm, force_monitor, gripper_client,
                            name, start_xy, direction, probe_yaw)
        if contact is None:
            rospy.logerr(f"{name}: aborting calibration.")
            return False
        contacts.append(contact)
        probe_dirs.append(direction / np.linalg.norm(direction))

    W1, W2, S1, S2 = contacts

    # --- حساب المحاور ---
    dN = W2 - W1
    eN_meas = dN / np.linalg.norm(dN)
    dH = S2 - S1
    eh_meas = dH / np.linalg.norm(dH)

    dot = float(np.dot(eh_meas, eN_meas))
    perp_err_deg = abs(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))) - 90.0)
    rospy.loginfo(f"Measured axes (pass2):")
    rospy.loginfo(f"  e_h_meas = ({eh_meas[0]:+.5f}, {eh_meas[1]:+.5f})")
    rospy.loginfo(f"  e_N_meas = ({eN_meas[0]:+.5f}, {eN_meas[1]:+.5f})")
    rospy.loginfo(f"  perpendicularity error = {perp_err_deg:.2f} deg")
    if perp_err_deg > 3.0:
        rospy.logwarn(f"  WARNING: perp. error large. Verify probe contacts.")

    eN_as_eh = np.array([ eN_meas[1], -eN_meas[0] ])
    eh_avg = (eh_meas + eN_as_eh) / 2.0
    eh_avg = eh_avg / np.linalg.norm(eh_avg)
    eN_fix = np.array([ -eh_avg[1], eh_avg[0] ])

    # --- تصحيح نقاط التلامس بازاحة نصف عرض الجريبر ---
    # الجريبر الآن عامودي على الحافة (بفضل yaw المصحّح)،
    # فاتجاه الحركة المدور هو نفسه عامود الحافة → التصحيح دقيق.
    rospy.loginfo(f"Applying gripper tip offset = "
                  f"{GRIPPER_TIP_OFFSET*1000:.1f} mm to contacts")
    contacts_corr = [c + d * GRIPPER_TIP_OFFSET
                     for c, d in zip(contacts, probe_dirs)]
    W1, W2, S1, S2 = contacts_corr
    for (name, _, _, _), c_raw, c_cor in zip(probes_p2, contacts, contacts_corr):
        rospy.loginfo(f"  {name}: raw=({c_raw[0]:+.4f},{c_raw[1]:+.4f}) "
                      f"-> corr=({c_cor[0]:+.4f},{c_cor[1]:+.4f})")

    # اعادة حساب المحاور من النقاط المصححة
    eN_meas = (W2 - W1) / np.linalg.norm(W2 - W1)
    eh_meas = (S2 - S1) / np.linalg.norm(S2 - S1)
    eN_as_eh = np.array([ eN_meas[1], -eN_meas[0] ])
    eh_avg = (eh_meas + eN_as_eh) / 2.0
    eh_avg = eh_avg / np.linalg.norm(eh_avg)
    eN_fix = np.array([ -eh_avg[1], eh_avg[0] ])

    A = np.column_stack([eN_fix, -eh_avg])
    b = S1 - W1
    try:
        uv = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        rospy.logerr("Line intersection failed (axes nearly parallel).")
        return False
    corner_meas = W1 + uv[0] * eN_fix

    # زاوية ميل اللوحة فقط (بدون قلبة 180 الناتجة عن اتجاه القياس):
    # في الحالة المثالية المقاسة eh_avg = (0, -1) لأن S2 تحت S1 بمحور Y.
    # نحسب الزاوية من (0, -1) الى eh_avg الفعلي → ميل اللوحة.
    # CCW: theta = atan2(eh.x, -eh.y)
    theta_meas = float(np.arctan2(eh_avg[0], -eh_avg[1]))

    board_corner = corner_meas
    e_h_axis     = eh_avg
    e_N_axis     = eN_fix
    board_theta  = theta_meas

    rospy.loginfo("="*62)
    rospy.loginfo("Calibration complete.")
    rospy.loginfo(f"  theta_est  (pass1)      = {np.degrees(theta_est):+.3f} deg")
    rospy.loginfo(f"  theta_final(pass2)      = {np.degrees(theta_meas):+.3f} deg")
    rospy.loginfo(f"  corner (outer, near a1) = "
                  f"({corner_meas[0]:+.4f}, {corner_meas[1]:+.4f}) m")
    rospy.loginfo(f"  e_h                     = "
                  f"({eh_avg[0]:+.5f}, {eh_avg[1]:+.5f})")
    rospy.loginfo(f"  e_N                     = "
                  f"({eN_fix[0]:+.5f}, {eN_fix[1]:+.5f})")
    rospy.loginfo("="*62)

    build_positions(board_corner, e_h_axis, e_N_axis)
    return True
 
# =====================================================================
# --- حفظ / تحميل المعايرة ---
# =====================================================================
def save_calibration(path=CALIB_FILE):
    data = {
        'corner_x': float(board_corner[0]),
        'corner_y': float(board_corner[1]),
        'theta_rad':float(board_theta),
        'e_h_x':    float(e_h_axis[0]),
        'e_h_y':    float(e_h_axis[1]),
        'e_N_x':    float(e_N_axis[0]),
        'e_N_y':    float(e_N_axis[1]),
        'square_size':     float(square_size),
        'margin':          float(MARGIN),
        'board_outer_size':float(BOARD_OUTER_SIZE),
    }
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False)
    rospy.loginfo(f"Calibration saved -> {path}")
 
def load_calibration(path=CALIB_FILE):
    global board_corner, board_theta, e_h_axis, e_N_axis
    if not os.path.exists(path):
        rospy.logwarn(f"No calibration file at {path}")
        return False
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    board_corner = np.array([data['corner_x'], data['corner_y']])
    board_theta  = float(data['theta_rad'])
    e_h_axis     = np.array([data['e_h_x'], data['e_h_y']])
    e_N_axis     = np.array([data['e_N_x'], data['e_N_y']])
    build_positions(board_corner, e_h_axis, e_N_axis)
    rospy.loginfo(f"Calibration loaded <- {path}")
    rospy.loginfo(f"  corner=({board_corner[0]:+.4f}, {board_corner[1]:+.4f}), "
                  f"theta={np.degrees(board_theta):+.3f} deg")
    return True
 
def show_calibration():
    print("--- Current calibration ---")
    print(f"  corner (outer near a1): "
          f"({board_corner[0]:+.4f}, {board_corner[1]:+.4f}) m")
    print(f"  theta (board yaw)     : "
          f"{np.degrees(board_theta):+.3f} deg  ({board_theta:+.5f} rad)")
    print(f"  e_h (a -> h)          : "
          f"({e_h_axis[0]:+.5f}, {e_h_axis[1]:+.5f})")
    print(f"  e_N (row1 -> row8)    : "
          f"({e_N_axis[0]:+.5f}, {e_N_axis[1]:+.5f})")
    print(f"  a1 center             : "
          f"({square_positions['a1'][0]:+.4f}, {square_positions['a1'][1]:+.4f})")
    print(f"  h8 center             : "
          f"({square_positions['h8'][0]:+.4f}, {square_positions['h8'][1]:+.4f})")
    print(f"  home                  : "
          f"({hx:+.4f}, {hy:+.4f}, {hz:+.4f})")
 
def test_calibration(arm):
    for sq in ('a1', 'h1', 'h8', 'a8'):
        p = mirrored_squares[sq]
        rospy.loginfo(f"  test -> {sq} @ ({p[0]:+.4f}, {p[1]:+.4f})")
        move_to_pose(arm, p[0], p[1], safe_h, v_slow, a_slow)
        rospy.sleep(0.3)
    rospy.loginfo("  test -> home")
    move_to_pose(arm, hx, hy, hz, v_slow, a_slow)
 
# =====================================================================
# --- حلقة التحكم اليدوي ---
# =====================================================================
def manual_control():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('manual_robot_control_globals', anonymous=True)
 
    arm = moveit_commander.MoveGroupCommander(
        "panda1_manipulator",
        robot_description="/panda1/robot_description",
        ns="/panda1")
    gripper_client = actionlib.SimpleActionClient(
        '/panda1/franka_gripper/move', MoveAction)
    gripper_client.wait_for_server()
 
    force_monitor = ForceMonitor()
    try:
        force_monitor.wait_for_data(timeout=5.0)
    except RuntimeError as e:
        rospy.logwarn(f"ForceMonitor: {e}")
 
    load_calibration()
 
    print(f"--- Manual Control (Open: {OPEN_WIDTH}, Close: {CLOSE_WIDTH}) ---")
    print(f"Home Coords: X={hx:.3f}, Y={hy:.3f}, Z={hz:.3f}")
    print(f"Board theta: {np.degrees(board_theta):+.3f} deg")
    print("Commands: a1..h8 | pq pr pb pn | x0..x23 | open | close | ready")
    print("          home | calibrate | save | load | show | test | exit")
 
    while not rospy.is_shutdown():
        cmd = input("Enter Target/Action: ").strip().lower()
        if cmd == 'exit': break
        if cmd == '':    continue
 
        if cmd == 'open':
            gripper_control(gripper_client, "open"); continue
        if cmd == 'close':
            gripper_control(gripper_client, "close"); continue
 
        if cmd == 'ready':
            arm.set_named_target('ready'); arm.go(wait=True); continue
 
        if cmd == 'home':
            move_to_pose(arm, hx, hy, hz, v_slow, a_slow)
            continue
 
        if cmd in ('calibrate', 'calib'):
            ok = calibrate_board(arm, force_monitor, gripper_client)
            if ok:
                ans = input("Save calibration to YAML? [y/N]: ").strip().lower()
                if ans == 'y':
                    save_calibration()
            continue
 
        if cmd == 'save':
            save_calibration(); continue
        if cmd == 'load':
            load_calibration(); continue
        if cmd == 'show':
            show_calibration(); continue
        if cmd == 'test':
            test_calibration(arm); continue
 
        target_pos = None
        if cmd in mirrored_squares:
            target_pos = mirrored_squares[cmd]
        elif cmd.startswith('p') and len(cmd) == 2 and cmd[1] in promotion_positions:
            target_pos = promotion_positions[cmd[1]]
        elif cmd.startswith('x'):
            try:
                idx = int(cmd[1:])
                if 0 <= idx < len(graveyard_positions):
                    target_pos = graveyard_positions[idx]
            except ValueError:
                pass
 
        if target_pos is not None:
            move_to_pose(arm, target_pos[0], target_pos[1], safe_h, v_slow, a_slow)
        else:
            print("Invalid Input!")
 
if __name__ == '__main__':
    try:
        manual_control()
    except rospy.ROSInterruptException:
        pass

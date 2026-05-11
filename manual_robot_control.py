#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
manual_robot_control.py
=======================
تحكم يدوي بذراع Franka Panda لمشروع روبوت الشطرنج.

الاضافات مقارنة بالنسخة الاصلية:
  - معايرة اللوحة بلمس 4 نقاط (نقطتان على الحافة الغربية + نقطتان على الحافة
    الجنوبية) باستخدام قراءة القوة الخارجية من /franka_states.
  - حساب الركن الخارجي للوحة (corner) وزاوية دورانها (theta).
  - اعادة بناء كل خرائط المواقع (مربعات الرقعة، الترقية، المقبرة) من
    الركن والمحاور المعايَرة.
  - تدوير الجريبر حول Z بزاوية اللوحة عبر quaternion_from_euler(pi, 0, theta).
  - حفظ/تحميل المعايرة من ملف YAML.

الأوامر في التحكم اليدوي:
  a1..h8 , pq/pr/pb/pn , x0..x23 , open , close , ready , home , exit
  calibrate   تشغيل تسلسل المعايرة (4 لمسات)
  save        حفظ المعايرة الحالية
  load        تحميل معايرة محفوظة
  show        عرض المعايرة الحالية
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
# --- متغيرات التحكم العالمية (الاصلية) ---
# =====================================================================
OPEN_WIDTH = 0.04
CLOSE_WIDTH = 0.025
GRIPPER_SPEED = 0.1

v_slow, a_slow = 0.7, 0.7
safe_h = 0.055

# --- ثوابت الهندسة للوحة (معلومة من التصنيع) ---
BOARD_OUTER_SIZE = 0.420                                   # 420mm اطار خارجي
PLAY_AREA_SIZE   = 0.360                                   # 360mm (8x8 * 45mm)
MARGIN           = (BOARD_OUTER_SIZE - PLAY_AREA_SIZE)/2.0 # 0.030m بين المربعات والاطار

# --- اعدادات الـprobing (المعايرة) ---
PROBE_Z            = safe_h   # ارتفاع اللمس (معلوم مسبقاً — لا نخاطر به)
PROBE_SPEED_SCALE  = 0.02     # 2% من السرعة القصوى — بطيء جدا لحماية اللوحة
PROBE_ACCEL_SCALE  = 0.02
PROBE_APPROACH_OFF = 0.040    # 40mm خارج الحافة المتوقعة (نقطة انطلاق آمنة)
PROBE_MAX_TRAVEL   = 0.070    # 70mm اقصى مسافة قبل الاستسلام
PROBE_RETREAT      = 0.020    # 20mm تراجع بعد التلامس
PROBE_FORCE_THRESH = 5.0      # نيوتن (|F_xy|) — عتبة كشف التلامس
PROBE_BIAS_SAMPLES = 50       # عدد عينات لحساب bias القوة
PROBE_BIAS_RATE_HZ = 100
PROBE_EDGE_SAMPLE_V = 0.050   # المسافة بين اللمسة والركن على نفس الحافة (حد ادنى)

CALIB_FILE = os.path.expanduser("~/board_calibration.yaml")

# ------------------------------------------------

origin_x, origin_y, origin_z = 0.3981098174137588, -6.41297277950631e-05, 0
square_size = 0.045

# =====================================================================
# --- حالة المعايرة (تُحدَّث بعد calibrate/load) ---
# =====================================================================
# الركن الخارجي للوحة القريب من a1 (تقاطع الحافة الغربية مع الجنوبية)
# الافتراضي مشتق من origin_x/origin_y ليعطي نفس سلوك النسخة الاصلية قبل المعايرة.
board_corner = np.array([
    origin_x + 7.5*square_size + MARGIN,
    origin_y - 7.5*square_size - MARGIN,
])
# محور a -> h في اطار الروبوت (متجه وحدة)
e_h_axis = np.array([0.0, 1.0])
# محور row1 -> row8 في اطار الروبوت (متجه وحدة، عمودي على e_h)
e_N_axis = np.array([-1.0, 0.0])
# زاوية اللوحة حول Z (rad). الصيغة: theta = atan2(e_h.y, e_h.x) - pi/2
# بحيث theta=0 يطابق الاتجاه الافتراضي e_h=(0,1)  =>  yaw الجريبر الاصلي = 0.
board_theta = 0.0

# خرائط المواقع (تُبنى من build_positions)
square_positions    = {}
mirrored_squares    = {}
promotion_positions = {}
graveyard_positions = []
hx = hy = hz = 0.0


# =====================================================================
# --- بناء خرائط المواقع من (corner, e_h, e_N) ---
# =====================================================================
def build_positions(corner=None, eh=None, eN=None):
    """
    يعيد بناء كل خرائط المواقع بناءً على الركن والمحاور.
    اذا لم تُمرر قيم يستخدم المتغيرات العالمية (board_corner, e_h_axis, e_N_axis).

    الاطار المحلي للوحة (u,v):
        u-axis = e_h (من a الى h)،  range: 0 .. BOARD_OUTER_SIZE
        v-axis = e_N (من row1 الى row8)،  range: 0 .. BOARD_OUTER_SIZE
        الركن عند (u=0, v=0) = الحافة الخارجية على جانبي a + row1.

    مواقع المربعات (u,v) = (MARGIN + (col_idx+0.5)*sq , MARGIN + (row-1+0.5)*sq)
    مواقع الترقية على خط u = MARGIN - 0.01 - 3.5*sq  (خارج جانب a)
        v للـ q=row8 ، r=row7 ، b=row6 ، n=row5
    المقبرة 3x8 على خط u خارج جانب a بمسافة (col_g+0.5)*sq + 0.01 - MARGIN
    """
    global square_positions, mirrored_squares, promotion_positions, graveyard_positions
    global hx, hy, hz

    if corner is None: corner = board_corner
    if eh    is None: eh    = e_h_axis
    if eN    is None: eN    = e_N_axis

    corner = np.asarray(corner, dtype=float)
    eh     = np.asarray(eh,     dtype=float)
    eN     = np.asarray(eN,     dtype=float)

    # 1) مربعات الرقعة
    sq_pos = {}
    for col_idx, col in enumerate('abcdefgh'):
        for row in range(1, 9):
            u = MARGIN + (col_idx + 0.5) * square_size
            v = MARGIN + (row - 1 + 0.5) * square_size
            p = corner + u * eh + v * eN
            sq_pos[f"{col}{row}"] = (float(p[0]), float(p[1]), origin_z)

    # 2) مواقع الترقية (off-board على جانب a، على نفس u، بـ v مطابق لصفوف 8..5)
    u_promo = MARGIN - 0.01 - 3.5 * square_size
    v_by_letter = {
        'q': MARGIN + 7.5 * square_size,   # محاذي row 8
        'r': MARGIN + 6.5 * square_size,   # محاذي row 7
        'b': MARGIN + 5.5 * square_size,   # محاذي row 6
        'n': MARGIN + 4.5 * square_size,   # محاذي row 5
    }
    promo = {}
    for letter, v in v_by_letter.items():
        p = corner + u_promo * eh + v * eN
        promo[letter] = (float(p[0]), float(p[1]), origin_z)

    # 3) المقبرة: 3 صفوف (col_g=0..2) * 8 اعمدة (row_g=0..7) كما في الاصل
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

    # النسخة المعكوسة
    original_keys = list(square_positions.keys())
    reversed_keys = list(reversed(original_keys))
    mirrored_squares = {original_keys[i]: square_positions[reversed_keys[i]]
                        for i in range(len(original_keys))}

    # --- موقع الـhome المخصص (مشتق من h8 + ازاحة ثابتة، كما في الاصل) ---
    base_hx, base_hy, _ = square_positions["h8"]
    hx = base_hx - 0.1
    hy = base_hy + 0.3
    hz = safe_h


# بناء القيم الافتراضية عند استيراد الملف
build_positions()


# =====================================================================
# --- مراقب القوة (subscriber على franka_states) ---
# =====================================================================
class ForceMonitor:
    """يقرأ O_F_ext_hat_K من رسالة FrankaState ويزوّد |F_xy| بعد خصم الـbias."""

    def __init__(self, topic="/panda1/franka_state_controller/franka_states"):
        self._lock = threading.Lock()
        self._force_xyz = np.zeros(3)
        self._bias_xyz  = np.zeros(3)
        self._got_msg   = False
        self._sub = rospy.Subscriber(topic, FrankaState, self._cb, queue_size=1)

    def _cb(self, msg):
        f = msg.O_F_ext_hat_K  # [Fx,Fy,Fz,Tx,Ty,Tz] في اطار القاعدة
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
        """يحسب bias القوة من القراءات الحالية (الذراع ساكن)."""
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
def _make_down_pose(x, y, z, yaw):
    """ينشئ Pose متوجه للأسفل (roll=pi) بزاوية yaw حول Z."""
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = float(x), float(y), float(z)
    q = quaternion_from_euler(np.pi, 0.0, float(yaw))
    pose.orientation = Quaternion(*q)
    return pose


def move_to_pose(move_group, x, y, z, vf, af, yaw=None):
    """
    حركة Cartesian الى (x,y,z). اذا لم يُمرّر yaw يُستخدم board_theta الحالي
    (بحيث يدور الجريبر مع اللوحة تلقائياً بعد المعايرة).
    """
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


# =====================================================================
# --- محرك اللمس (probe) ---
# =====================================================================
def probe_linear(move_group, force_monitor, direction_xy,
                 max_travel=PROBE_MAX_TRAVEL,
                 force_threshold=PROBE_FORCE_THRESH):
    """
    يتحرك من الموقع الحالي في direction_xy بسرعة بطيئة جداً حتى:
      - |F_xy| > force_threshold -> يوقف ويرجع (x,y) للاتصال.
      - او يقطع max_travel       -> يرجع None (لم يحدث تلامس).

    يفترض ان الذراع ساكن قبل الاستدعاء ليحسب bias القوة.
    """
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

    # zero force bias while stationary
    rospy.loginfo("  [probe] zeroing force bias ...")
    force_monitor.zero_bias()

    rospy.loginfo(f"  [probe] moving: dir=({d[0]:+.2f},{d[1]:+.2f}), "
                  f"max={max_travel*1000:.0f}mm, thresh={force_threshold:.1f}N")
    move_group.execute(plan, wait=False)

    rate = rospy.Rate(200)
    contact = False
    t0 = rospy.Time.now()
    timeout = 30.0  # انتظار كافٍ حتى عند السرعة البطيئة
    while not rospy.is_shutdown() and (rospy.Time.now() - t0).to_sec() < timeout:
        fmag = force_monitor.get_xy_magnitude()
        if fmag > force_threshold:
            move_group.stop()
            rospy.loginfo(f"  [probe] CONTACT! |F_xy|={fmag:.2f} N")
            contact = True
            break
        cur_now = move_group.get_current_pose().pose
        if np.hypot(target_xy[0]-cur_now.position.x,
                    target_xy[1]-cur_now.position.y) < 0.002:
            rospy.logwarn("  [probe] reached max_travel without contact")
            break
        rate.sleep()

    move_group.stop()
    move_group.clear_pose_targets()
    if not contact:
        return None

    rospy.sleep(0.25)  # استقرار
    final = move_group.get_current_pose().pose
    return np.array([final.position.x, final.position.y])


# =====================================================================
# --- معايرة اللوحة ---
# =====================================================================
def calibrate_board(arm, force_monitor):
    """
    تنفذ 4 لمسات مرتبة:
      W1, W2 على الحافة الغربية (جانب a)   -> يعطينا خط الغرب (اتجاه e_N وموقعه)
      S1, S2 على الحافة الجنوبية (جانب row1) -> يعطينا خط الجنوب (اتجاه e_h وموقعه)
    ثم تقاطع الخطين = الركن الخارجي للوحة قرب a1.

    يحدّث board_corner, board_theta, e_h_axis, e_N_axis ثم يعيد بناء المواقع.
    """
    global board_corner, board_theta, e_h_axis, e_N_axis

    # انطلق من القيم الحالية (افتراضية او من تحميل سابق)
    corner0 = board_corner.copy()
    eh0     = e_h_axis.copy()
    eN0     = e_N_axis.copy()

    # --- نقاط الانطلاق (في الاطار المحلي قبل المعايرة) ---
    # الحافة الغربية: u=0، نلمس في +e_h من نقطة عند u=-PROBE_APPROACH_OFF
    v_w1 = PROBE_EDGE_SAMPLE_V                              # قرب row 1
    v_w2 = BOARD_OUTER_SIZE - PROBE_EDGE_SAMPLE_V           # قرب row 8
    # الحافة الجنوبية: v=0، نلمس في +e_N من نقطة عند v=-PROBE_APPROACH_OFF
    u_s1 = PROBE_EDGE_SAMPLE_V                              # قرب a-side
    u_s2 = BOARD_OUTER_SIZE - PROBE_EDGE_SAMPLE_V           # قرب h-side

    W1_start = corner0 + (-PROBE_APPROACH_OFF) * eh0 + v_w1 * eN0
    W2_start = corner0 + (-PROBE_APPROACH_OFF) * eh0 + v_w2 * eN0
    S1_start = corner0 + u_s1 * eh0 + (-PROBE_APPROACH_OFF) * eN0
    S2_start = corner0 + u_s2 * eh0 + (-PROBE_APPROACH_OFF) * eN0

    probes = [
        ("W1 (west edge, near row 1)",  W1_start, eh0),
        ("W2 (west edge, near row 8)",  W2_start, eh0),
        ("S1 (south edge, near a-side)", S1_start, eN0),
        ("S2 (south edge, near h-side)", S2_start, eN0),
    ]

    rospy.loginfo("="*62)
    rospy.loginfo("Starting board calibration (4 probes)")
    rospy.loginfo(f"  approach offset : {PROBE_APPROACH_OFF*1000:.0f} mm outside edge")
    rospy.loginfo(f"  probe speed     : {PROBE_SPEED_SCALE*100:.0f}% of max")
    rospy.loginfo(f"  force threshold : {PROBE_FORCE_THRESH:.1f} N")
    rospy.loginfo(f"  probe height Z  : {PROBE_Z:.3f} m")
    rospy.loginfo("="*62)

    ans = input("Make sure the workspace is CLEAR. Proceed? [y/N]: ").strip().lower()
    if ans != 'y':
        rospy.loginfo("Calibration aborted by user.")
        return False

    contacts = []
    for name, start_xy, direction in probes:
        rospy.loginfo(f"\n[Probe] {name}")

        # 1) ارتفع للارتفاع الآمن
        cur = arm.get_current_pose().pose
        move_to_pose(arm, cur.position.x, cur.position.y, safe_h,
                     v_slow, a_slow, yaw=0.0)

        # 2) انتقل لنقطة الانطلاق (XY)
        rospy.loginfo(f"  approach -> ({start_xy[0]:+.4f}, {start_xy[1]:+.4f})")
        move_to_pose(arm, start_xy[0], start_xy[1], safe_h,
                     v_slow, a_slow, yaw=0.0)

        # 3) انزل لارتفاع الـprobe
        move_to_pose(arm, start_xy[0], start_xy[1], PROBE_Z,
                     v_slow, a_slow, yaw=0.0)

        # 4) نفّذ الـprobe
        contact = probe_linear(arm, force_monitor, direction)
        if contact is None:
            rospy.logerr(f"{name}: no contact. Aborting calibration.")
            cur = arm.get_current_pose().pose
            move_to_pose(arm, cur.position.x, cur.position.y, safe_h,
                         v_slow, a_slow, yaw=0.0)
            return False
        rospy.loginfo(f"  contact  @ ({contact[0]:+.4f}, {contact[1]:+.4f})")
        contacts.append(contact)

        # 5) تراجع قليلاً في اتجاه معاكس ثم ارتفع
        back = contact - np.asarray(direction)/np.linalg.norm(direction)*PROBE_RETREAT
        move_to_pose(arm, back[0], back[1], PROBE_Z, v_slow, a_slow, yaw=0.0)
        move_to_pose(arm, back[0], back[1], safe_h,  v_slow, a_slow, yaw=0.0)

    W1, W2, S1, S2 = contacts

    # --- احسب المحاور من النقاط ---
    # e_N من (W1 قرب row1) الى (W2 قرب row8)
    dN = W2 - W1
    eN_meas = dN / np.linalg.norm(dN)
    # e_h من (S1 قرب a-side) الى (S2 قرب h-side)
    dH = S2 - S1
    eh_meas = dH / np.linalg.norm(dH)

    # فحص التعامد
    dot = float(np.dot(eh_meas, eN_meas))
    perp_err_deg = abs(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))) - 90.0)
    rospy.loginfo(f"\nMeasured axes:")
    rospy.loginfo(f"  e_h_meas = ({eh_meas[0]:+.5f}, {eh_meas[1]:+.5f})")
    rospy.loginfo(f"  e_N_meas = ({eN_meas[0]:+.5f}, {eN_meas[1]:+.5f})")
    rospy.loginfo(f"  perpendicularity error = {perp_err_deg:.2f} deg")
    if perp_err_deg > 3.0:
        rospy.logwarn(f"  WARNING: perp. error large. Verify probe contacts.")

    # فرض التعامد بأخذ متوسط مقبول (e_N rotated -90° should match e_h)
    eN_as_eh = np.array([ eN_meas[1], -eN_meas[0] ])   # -90° rotation
    eh_avg = (eh_meas + eN_as_eh) / 2.0
    eh_avg = eh_avg / np.linalg.norm(eh_avg)
    eN_fix = np.array([ -eh_avg[1], eh_avg[0] ])       # +90° of e_h

    # --- تقاطع الخطين: corner = W1 + u*e_N = S1 + v*e_h ---
    # u*e_N - v*e_h = S1 - W1
    A = np.column_stack([eN_fix, -eh_avg])
    b = S1 - W1
    try:
        uv = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        rospy.logerr("Line intersection failed (axes nearly parallel).")
        return False
    corner_meas = W1 + uv[0] * eN_fix

    # الزاوية
    theta_meas = float(np.arctan2(eh_avg[1], eh_avg[0])) - np.pi/2.0

    # --- تحديث الحالة العالمية ---
    board_corner = corner_meas
    e_h_axis     = eh_avg
    e_N_axis     = eN_fix
    board_theta  = theta_meas

    rospy.loginfo("="*62)
    rospy.loginfo("Calibration complete.")
    rospy.loginfo(f"  corner (outer, near a1) = "
                  f"({corner_meas[0]:+.4f}, {corner_meas[1]:+.4f}) m")
    rospy.loginfo(f"  theta                   = {np.degrees(theta_meas):+.3f} deg")
    rospy.loginfo(f"  e_h                     = "
                  f"({eh_avg[0]:+.5f}, {eh_avg[1]:+.5f})")
    rospy.loginfo(f"  e_N                     = "
                  f"({eN_fix[0]:+.5f}, {eN_fix[1]:+.5f})")
    rospy.loginfo("="*62)

    # اعادة بناء المواقع + مركز الجريبر
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
    print("\n--- Current calibration ---")
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
        rospy.logwarn(f"ForceMonitor: {e}  (calibrate will be unavailable until topic is up)")

    # محاولة تحميل معايرة محفوظة عند البدء
    load_calibration()

    print(f"\n--- Manual Control (Open: {OPEN_WIDTH}, Close: {CLOSE_WIDTH}) ---")
    print(f"Home Coords: X={hx:.3f}, Y={hy:.3f}, Z={hz:.3f}")
    print(f"Board theta: {np.degrees(board_theta):+.3f} deg")
    print("Commands:")
    print("  a1..h8          : move to square")
    print("  pq pr pb pn     : promotion position")
    print("  x0..x23         : graveyard slot")
    print("  open | close    : gripper")
    print("  ready           : 'ready' named target")
    print("  home            : custom home above board")
    print("  calibrate       : run 4-point board calibration")
    print("  save | load     : persist / restore calibration YAML")
    print("  show            : print current calibration")
    print("  exit            : quit")

    while not rospy.is_shutdown():
        cmd = input("\nEnter Target/Action: ").strip().lower()
        if cmd == 'exit': break
        if cmd == '':    continue

        if cmd == 'open':
            gripper_control(gripper_client, "open"); continue
        if cmd == 'close':
            gripper_control(gripper_client, "close"); continue

        if cmd == 'ready':
            arm.set_named_target('ready'); arm.go(wait=True); continue

        if cmd == 'home':
            rospy.loginfo(f"Moving to custom Home: ({hx:.3f}, {hy:.3f}, {hz:.3f}), "
                          f"yaw={np.degrees(board_theta):+.2f}deg")
            move_to_pose(arm, hx, hy, hz, v_slow, a_slow)
            continue

        if cmd in ('calibrate', 'calib'):
            ok = calibrate_board(arm, force_monitor)
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

        # --- اهداف الحركة ---
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

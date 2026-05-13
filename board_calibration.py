#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
board_calibration.py
====================
معايرة لوحة الشطرنج وبناء مواقع المربعات لذراع Franka Panda.

المحتويات:
- ثوابت هندسة اللوحة + اعدادات الـprobing
- TF helpers (init_tf, get_current_pose_in_ref)
- دوال حركة مشتركة (move_to_pose, make_down_pose, gripper_full_close)
- ForceMonitor: مراقبة القوة الخارجية عبر FrankaState
- BoardCalibration: يمسك حالة المعايرة + يبنى خرائط المواقع
  + ينفذ الـprobing والمعايرة (two-pass) + save/load/show/test
"""

import os
import copy
import threading

import rospy
import numpy as np
import yaml
import tf2_ros
import tf2_geometry_msgs  # لازم للـPoseStamped.transform()

from geometry_msgs.msg import Pose, Quaternion
from tf.transformations import quaternion_from_euler
from franka_gripper.msg import MoveGoal
from franka_msgs.msg import FrankaState


# =====================================================================
# --- الاطار المرجعي ---
# =====================================================================
REFERENCE_FRAME = "world"


# =====================================================================
# --- ثوابت الحركة والقابض ---
# =====================================================================
V_SLOW = 0.7
A_SLOW = 0.7
SAFE_H = 0.12

OPEN_WIDTH    = 0.04
CLOSE_WIDTH   = 0.025
GRIPPER_SPEED = 0.1


# =====================================================================
# --- ثوابت هندسة اللوحة ---
# =====================================================================
BOARD_OUTER_SIZE = 0.360
PLAY_AREA_SIZE   = 0.360
MARGIN           = (BOARD_OUTER_SIZE - PLAY_AREA_SIZE) / 2.0  # 0.030m

ORIGIN_X    = 0.3981098174137588
ORIGIN_Y    = -6.41297277950631e-05
ORIGIN_Z    = 0
SQUARE_SIZE = 0.045


# =====================================================================
# --- اعدادات الـprobing ---
# =====================================================================
PROBE_Z              = 0.06
PROBE_SPEED_SCALE    = 0.02
PROBE_ACCEL_SCALE    = 0.02
PROBE_MAX_TRAVEL     = 0.4
PROBE_RETREAT        = 0.020
PROBE_POST_LIFT      = 0.010
PROBE_TRANSIT_Z      = SAFE_H + 0.080
PROBE_FORCE_THRESH   = 4   # عتبة W probes
PROBE_FORCE_THRESH_S = 4   # عتبة S probes
PROBE_BIAS_SAMPLES   = 50
PROBE_BIAS_RATE_HZ   = 100
PROBE_CONTACT_CONSEC = 3     # عدد العينات المتتالية لتأكيد اللمس
PROBE_MIN_TRAVEL     = 0.005 # أقل مسافة قبل قبول contact (لتجاهل spike البداية)

# --- ازاحة طرف القابض عن مركز TCP ---
GRIPPER_TIP_OFFSET = (0.018 / 2) - 0.00


# --- نقاط بدء الـprobing ---
W1_START_XY = (0.50,  0.20)
W2_START_XY = (0.52,  0.20)
W_PROBE_DIR = (0.0, -1.0)
W_PROBE_YAW = np.pi / 2.0

S1_START_XY = (0.30, -0.20)
S2_START_XY = (0.30, -0.22)
S_PROBE_DIR = (+1.0, 0.0)
S_PROBE_YAW = 0.0


CALIB_FILE = os.path.expanduser("~/board_calibration.yaml")


# =====================================================================
# --- TF buffer عالمي ---
# =====================================================================
_tf_buffer   = None
_tf_listener = None


def init_tf(timeout=5.0):
    """ينشئ TF buffer/listener ويتحقق من التحويل panda1_link0 -> world."""
    global _tf_buffer, _tf_listener
    if _tf_buffer is not None:
        return _tf_buffer

    _tf_buffer   = tf2_ros.Buffer()
    _tf_listener = tf2_ros.TransformListener(_tf_buffer)
    rospy.loginfo(f"Waiting for TF: panda1_link0 <-> {REFERENCE_FRAME} ...")
    try:
        _tf_buffer.can_transform(REFERENCE_FRAME, "panda1_link0",
                                 rospy.Time(0), rospy.Duration(timeout))
        rospy.loginfo(f"  TF available: panda1_link0 -> {REFERENCE_FRAME}")
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"  TF NOT available: {e}")
        rospy.logerr("  Check that static_transform_publisher or URDF "
                     "defines the link between panda1_link0 and "
                     f"{REFERENCE_FRAME}.")
    return _tf_buffer


def get_current_pose_in_ref(move_group, timeout=1.0):
    """
    ترجع current pose في REFERENCE_FRAME (world) بدل الـplanning frame.
    ترجع: geometry_msgs/Pose (بدون header).
    """
    ps_planning = move_group.get_current_pose()
    if _tf_buffer is None:
        return ps_planning.pose
    try:
        ps_ref = _tf_buffer.transform(ps_planning, REFERENCE_FRAME,
                                      rospy.Duration(timeout))
        return ps_ref.pose
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"get_current_pose_in_ref: TF transform failed: {e}")
        return ps_planning.pose


# =====================================================================
# --- دوال حركة مساعدة ---
# =====================================================================
def make_down_pose(x, y, z, yaw):
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    q = quaternion_from_euler(np.pi, 0.0, float(yaw))
    pose.orientation = Quaternion(*q)
    return pose


def move_to_pose(move_group, x, y, z, vf, af, yaw=0.0):
    """حركة Cartesian بسيطة (pointing down) مع retime."""
    pose = make_down_pose(x, y, z, yaw)
    waypoints = [copy.deepcopy(pose)]
    plan, fraction = move_group.compute_cartesian_path(waypoints, 0.05, False)
    if fraction > 0.9:
        plan = move_group.retime_trajectory(
            move_group.get_current_state(), plan, vf, af,
            "iterative_time_parameterization")
        move_group.execute(plan, wait=True)
    move_group.stop()
    move_group.clear_pose_targets()


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


def _rot_2d(v, angle):
    """دوران متجه 2D بزاوية angle (CCW) حول نقطة الأصل."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c*v[0] - s*v[1], s*v[0] + c*v[1]])


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
# --- فئة المعايرة وبناء المواقع ---
# =====================================================================
class BoardCalibration:
    """
    يمسك حالة المعايرة (corner, axes, theta) ويبنى عليها خرائط:
      - square_positions    : a1..h8
      - mirrored_squares    : reversed mapping (الـindex بيستخدم ده)
      - promotion_positions : q, r, b, n
      - graveyard_positions : 24 مربع
      - hx, hy, hz          : نقطة الـhome
    ويوفر probing + calibrate (two-pass) + save/load/show/test.
    """

    def __init__(self):
        # حالة افتراضية قبل المعايرة
        self.board_corner = np.array([
            ORIGIN_X + 7.5 * SQUARE_SIZE + MARGIN,
            ORIGIN_Y - 7.5 * SQUARE_SIZE - MARGIN,
        ])
        self.e_h_axis    = np.array([0.0, 1.0])
        self.e_N_axis    = np.array([-1.0, 0.0])
        self.board_theta = 0.0

        self.square_positions    = {}
        self.mirrored_squares    = {}
        self.promotion_positions = {}
        self.graveyard_positions = []
        self.hx = self.hy = self.hz = 0.0

        self.build_positions()

    # ------------------------------------------------------------------
    # --- بناء خرائط المواقع ---
    # ------------------------------------------------------------------
    def build_positions(self, corner=None, eh=None, eN=None):
        if corner is None: corner = self.board_corner
        if eh     is None: eh     = self.e_h_axis
        if eN     is None: eN     = self.e_N_axis

        corner = np.asarray(corner, dtype=float)
        eh     = np.asarray(eh,     dtype=float)
        eN     = np.asarray(eN,     dtype=float)

        sq_pos = {}
        for col_idx, col in enumerate('abcdefgh'):
            for row in range(1, 9):
                u = MARGIN + (col_idx + 0.5) * SQUARE_SIZE
                v = MARGIN + (row - 1 + 0.5) * SQUARE_SIZE
                p = corner + u * eh + v * eN
                sq_pos[f"{col}{row}"] = (float(p[0]), float(p[1]), ORIGIN_Z)

        # promotion + graveyard على الجهة الثانية من اللوحة (بعد h-file)،
        # دائماً ثابتات فيزيائياً بغض النظر عن لون اللاعب.
        # يمين الروبوت، بتبلش من الصف الأقرب للروبوت (row_g=0 ⇒ v=7.5*SQ).
        board_u_far = MARGIN + 8 * SQUARE_SIZE + 0.01  # بعد حافة اللوحة بـ1cm

        u_promo = board_u_far + 3.5 * SQUARE_SIZE
        v_by_letter = {
            'q': MARGIN + 7.5 * SQUARE_SIZE,
            'r': MARGIN + 6.5 * SQUARE_SIZE,
            'b': MARGIN + 5.5 * SQUARE_SIZE,
            'n': MARGIN + 4.5 * SQUARE_SIZE,
        }
        promo = {}
        for letter, v in v_by_letter.items():
            p = corner + u_promo * eh + v * eN
            promo[letter] = (float(p[0]), float(p[1]), ORIGIN_Z)

        # ترتيب الـgraveyard: row-major يبلش من الصف الأقرب للروبوت.
        # row_g=0 هو الصف الأقرب للروبوت، col_g=0 العمود الأقرب لحافة اللوحة.
        # فالمؤشرات:
        #   x0  x1  x2   ← أقرب صف للروبوت (3 أعمدة)
        #   x3  x4  x5   ← الصف اللي بعده
        #   ...
        #   x21 x22 x23  ← أبعد صف عن الروبوت
        graves = []
        for row_g in range(8):
            v = MARGIN + (7.5 - row_g) * SQUARE_SIZE
            for col_g in range(3):
                u = board_u_far + (col_g + 0.5) * SQUARE_SIZE
                p = corner + u * eh + v * eN
                graves.append((float(p[0]), float(p[1]), ORIGIN_Z))

        self.square_positions    = sq_pos
        self.promotion_positions = promo
        self.graveyard_positions = graves

        original_keys = list(self.square_positions.keys())
        reversed_keys = list(reversed(original_keys))
        self.mirrored_squares = {
            original_keys[i]: self.square_positions[reversed_keys[i]]
            for i in range(len(original_keys))
        }

        base_hx, base_hy, _ = self.square_positions["h8"]
        self.hx = base_hx - 0.1
        self.hy = base_hy + 0.3
        self.hz = SAFE_H

    # ------------------------------------------------------------------
    # --- probing ---
    # ------------------------------------------------------------------
    def _probe_linear(self, move_group, force_monitor, direction_xy,
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
        final = get_current_pose_in_ref(move_group)
        return np.array([final.position.x, final.position.y])

    def _do_probe(self, arm, force_monitor, gripper_client,
                  name, start_xy, direction, probe_yaw,
                  force_threshold=PROBE_FORCE_THRESH):
        """approach + touch + retreat. ترجع نقطة التلامس أو None."""
        rospy.loginfo(f"[Probe] {name}")

        if gripper_client is not None:
            rospy.loginfo("  [probe] full-closing gripper before probe")
            gripper_full_close(gripper_client)

        cur = get_current_pose_in_ref(arm)
        move_to_pose(arm, cur.position.x, cur.position.y, PROBE_TRANSIT_Z,
                     V_SLOW, A_SLOW, yaw=probe_yaw)

        rospy.loginfo(f"  move -> start ({start_xy[0]:+.4f}, {start_xy[1]:+.4f}) "
                      f"@ Z={PROBE_TRANSIT_Z:.3f} yaw={np.degrees(probe_yaw):+.2f}deg")
        move_to_pose(arm, start_xy[0], start_xy[1], PROBE_TRANSIT_Z,
                     V_SLOW, A_SLOW, yaw=probe_yaw)

        rospy.loginfo(f"  descend -> Z={PROBE_Z:.3f}")
        move_to_pose(arm, start_xy[0], start_xy[1], PROBE_Z,
                     V_SLOW, A_SLOW, yaw=probe_yaw)

        contact = self._probe_linear(arm, force_monitor, direction,
                                     max_travel=PROBE_MAX_TRAVEL,
                                     force_threshold=force_threshold)
        if contact is None:
            rospy.logerr(f"{name}: no contact detected within "
                         f"{PROBE_MAX_TRAVEL*1000:.0f}mm.")
            cur = get_current_pose_in_ref(arm)
            move_to_pose(arm, cur.position.x, cur.position.y, PROBE_TRANSIT_Z,
                         V_SLOW, A_SLOW, yaw=probe_yaw)
            return None

        rospy.loginfo(f"  contact @ ({contact[0]:+.4f}, {contact[1]:+.4f})")

        d_norm = np.asarray(direction, dtype=float)
        d_norm = d_norm / np.linalg.norm(d_norm)
        back_xy = contact - d_norm * PROBE_RETREAT
        rospy.loginfo(f"  retreat -> ({back_xy[0]:+.4f}, {back_xy[1]:+.4f})")
        move_to_pose(arm, back_xy[0], back_xy[1], PROBE_Z,
                     V_SLOW, A_SLOW, yaw=probe_yaw)

        lift_z = PROBE_Z + PROBE_POST_LIFT
        move_to_pose(arm, back_xy[0], back_xy[1], lift_z,
                     V_SLOW, A_SLOW, yaw=probe_yaw)

        move_to_pose(arm, back_xy[0], back_xy[1], PROBE_TRANSIT_Z,
                     V_SLOW, A_SLOW, yaw=probe_yaw)

        return contact

    # ------------------------------------------------------------------
    # --- المعايرة الرئيسية (Two-pass) ---
    # ------------------------------------------------------------------
    def calibrate(self, arm, force_monitor, gripper_client=None, confirm=True):
        rospy.loginfo("=" * 62)
        rospy.loginfo("Starting board calibration (two-pass)")
        rospy.loginfo(f"  probe height Z   : {PROBE_Z:.3f} m")
        rospy.loginfo(f"  transit height Z : {PROBE_TRANSIT_Z:.3f} m")
        rospy.loginfo(f"  post-contact lift: {PROBE_POST_LIFT*1000:.0f} mm")
        rospy.loginfo(f"  probe speed      : {PROBE_SPEED_SCALE*100:.0f}% of max")
        rospy.loginfo(f"  force threshold  : {PROBE_FORCE_THRESH:.1f} N")
        rospy.loginfo(f"  max travel       : {PROBE_MAX_TRAVEL*1000:.0f} mm")
        rospy.loginfo("=" * 62)

        if confirm:
            ans = input("Make sure the workspace is CLEAR. Proceed? [y/N]: "
                        ).strip().lower()
            if ans != 'y':
                rospy.loginfo("Calibration aborted by user.")
                return False

        # =============================================================
        # PASS 1: W1 + W2 بـyaw الاصلي لتقدير θ
        # =============================================================
        rospy.loginfo(">>> PASS 1: estimating board angle from W1 & W2")
        W1_p1 = self._do_probe(arm, force_monitor, gripper_client,
                               "W1 (pass1)", W1_START_XY,
                               W_PROBE_DIR, W_PROBE_YAW)
        if W1_p1 is None:
            return False

        W2_p1 = self._do_probe(arm, force_monitor, gripper_client,
                               "W2 (pass1)", W2_START_XY,
                               W_PROBE_DIR, W_PROBE_YAW)
        if W2_p1 is None:
            return False

        dW = W2_p1 - W1_p1
        theta_est = float(np.arctan2(dW[1], dW[0]))
        rospy.loginfo(f"  W2 - W1 = ({dW[0]:+.4f}, {dW[1]:+.4f})")
        rospy.loginfo(f"  Estimated board theta = {np.degrees(theta_est):+.3f} deg")

        # =============================================================
        # PASS 2: الرجوع بـyaw/dir مصححين بـθ
        # =============================================================
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
            ("W1 (pass2)", W1_START_XY, W_dir_rot, W_yaw_corr, PROBE_FORCE_THRESH),
            ("W2 (pass2)", W2_START_XY, W_dir_rot, W_yaw_corr, PROBE_FORCE_THRESH),
            ("S1 (pass2)", S1_START_XY, S_dir_rot, S_yaw_corr, PROBE_FORCE_THRESH_S),
            ("S2 (pass2)", S2_START_XY, S_dir_rot, S_yaw_corr, PROBE_FORCE_THRESH_S),
        ]

        contacts = []
        probe_dirs = []
        for name, start_xy, direction, probe_yaw, thresh in probes_p2:
            contact = self._do_probe(arm, force_monitor, gripper_client,
                                     name, start_xy, direction, probe_yaw,
                                     force_threshold=thresh)
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
            rospy.logwarn("  WARNING: perp. error large. Verify probe contacts.")

        eN_as_eh = np.array([eN_meas[1], -eN_meas[0]])
        eh_avg = (eh_meas + eN_as_eh) / 2.0
        eh_avg = eh_avg / np.linalg.norm(eh_avg)
        eN_fix = np.array([-eh_avg[1], eh_avg[0]])

        # --- تصحيح نقاط التلامس بازاحة نصف عرض الجريبر ---
        rospy.loginfo(f"Applying gripper tip offset = "
                      f"{GRIPPER_TIP_OFFSET*1000:.1f} mm to contacts")
        contacts_corr = [c + d * GRIPPER_TIP_OFFSET
                         for c, d in zip(contacts, probe_dirs)]
        W1, W2, S1, S2 = contacts_corr
        for (name, *_rest), c_raw, c_cor in zip(probes_p2, contacts, contacts_corr):
            rospy.loginfo(f"  {name}: raw=({c_raw[0]:+.4f},{c_raw[1]:+.4f}) "
                          f"-> corr=({c_cor[0]:+.4f},{c_cor[1]:+.4f})")

        # اعادة حساب المحاور من النقاط المصححة
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

        self.board_corner = corner_meas
        self.e_h_axis     = eh_avg
        self.e_N_axis     = eN_fix
        self.board_theta  = theta_meas

        rospy.loginfo("=" * 62)
        rospy.loginfo("Calibration complete.")
        rospy.loginfo(f"  theta_est  (pass1)      = {np.degrees(theta_est):+.3f} deg")
        rospy.loginfo(f"  theta_final(pass2)      = {np.degrees(theta_meas):+.3f} deg")
        rospy.loginfo(f"  corner (outer, near a1) = "
                      f"({corner_meas[0]:+.4f}, {corner_meas[1]:+.4f}) m")
        rospy.loginfo(f"  e_h                     = "
                      f"({eh_avg[0]:+.5f}, {eh_avg[1]:+.5f})")
        rospy.loginfo(f"  e_N                     = "
                      f"({eN_fix[0]:+.5f}, {eN_fix[1]:+.5f})")
        rospy.loginfo("=" * 62)

        self.build_positions()
        return True

    # ------------------------------------------------------------------
    # --- حفظ / تحميل ---
    # ------------------------------------------------------------------
    def save(self, path=CALIB_FILE):
        data = {
            'corner_x':         float(self.board_corner[0]),
            'corner_y':         float(self.board_corner[1]),
            'theta_rad':        float(self.board_theta),
            'e_h_x':            float(self.e_h_axis[0]),
            'e_h_y':            float(self.e_h_axis[1]),
            'e_N_x':            float(self.e_N_axis[0]),
            'e_N_y':            float(self.e_N_axis[1]),
            'square_size':      float(SQUARE_SIZE),
            'margin':           float(MARGIN),
            'board_outer_size': float(BOARD_OUTER_SIZE),
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False)
        rospy.loginfo(f"Calibration saved -> {path}")

    def load(self, path=CALIB_FILE):
        if not os.path.exists(path):
            rospy.logwarn(f"No calibration file at {path}")
            return False
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        self.board_corner = np.array([data['corner_x'], data['corner_y']])
        self.board_theta  = float(data['theta_rad'])
        self.e_h_axis     = np.array([data['e_h_x'], data['e_h_y']])
        self.e_N_axis     = np.array([data['e_N_x'], data['e_N_y']])
        self.build_positions()
        rospy.loginfo(f"Calibration loaded <- {path}")
        rospy.loginfo(f"  corner=({self.board_corner[0]:+.4f}, "
                      f"{self.board_corner[1]:+.4f}), "
                      f"theta={np.degrees(self.board_theta):+.3f} deg")
        return True

    # ------------------------------------------------------------------
    # --- عرض / اختبار ---
    # ------------------------------------------------------------------
    def show(self):
        print("--- Current calibration ---")
        print(f"  corner (outer near a1): "
              f"({self.board_corner[0]:+.4f}, {self.board_corner[1]:+.4f}) m")
        print(f"  theta (board yaw)     : "
              f"{np.degrees(self.board_theta):+.3f} deg  "
              f"({self.board_theta:+.5f} rad)")
        print(f"  e_h (a -> h)          : "
              f"({self.e_h_axis[0]:+.5f}, {self.e_h_axis[1]:+.5f})")
        print(f"  e_N (row1 -> row8)    : "
              f"({self.e_N_axis[0]:+.5f}, {self.e_N_axis[1]:+.5f})")
        print(f"  a1 center             : "
              f"({self.square_positions['a1'][0]:+.4f}, "
              f"{self.square_positions['a1'][1]:+.4f})")
        print(f"  h8 center             : "
              f"({self.square_positions['h8'][0]:+.4f}, "
              f"{self.square_positions['h8'][1]:+.4f})")
        print(f"  home                  : "
              f"({self.hx:+.4f}, {self.hy:+.4f}, {self.hz:+.4f})")

    def test(self, arm, safe_h=None):
        if safe_h is None:
            safe_h = SAFE_H
        for sq in ('a1', 'h1', 'h8', 'a8'):
            p = self.mirrored_squares[sq]
            rospy.loginfo(f"  test -> {sq} @ ({p[0]:+.4f}, {p[1]:+.4f})")
            move_to_pose(arm, p[0], p[1], safe_h,
                         V_SLOW, A_SLOW, yaw=self.board_theta)
            rospy.sleep(0.3)
        rospy.loginfo("  test -> home")
        move_to_pose(arm, self.hx, self.hy, self.hz,
                     V_SLOW, A_SLOW, yaw=self.board_theta)

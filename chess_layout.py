#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chess_layout.py
===============
Standalone module for computing world-frame positions of:

  * 64 chessboard squares (a1..h8), with color-based mirroring
  * Promotion slots (q, r, b, n)
  * Capture / graveyard slots (indexed from the row nearest the robot)
  * Home position (always on the robot's left, offset from the nearest-left
    square by a configurable distance)

Everything here is pure-Python + NumPy. The ROS-specific part (subscribing
to a color topic) lives in the ColorSubscriber class at the bottom and is
optional — import the module without rospy available and it still works.

Conventions
-----------
Board frame (from calibration):
    corner  : outer corner near a1 (in world frame)
    e_h     : unit vector along a -> h
    e_N     : unit vector along row 1 -> row 8

Robot stance (depends on color):
    color='white'  -> robot is positioned near row 1, faces +e_N
                      robot's LEFT  = -e_h direction (a-file side)
    color='black'  -> robot is positioned near row 8, faces -e_N
                      robot's LEFT  = +e_h direction (h-file side)

Auxiliary positions (capture, promotion, home) are always placed on the
robot's LEFT, so their physical location swaps between the a-file and
h-file sides when the color changes, but their pose relative to the robot
stays the same.

Square naming vs. color
-----------------------
    color='white' -> squares['a1'] = physical a1, squares['h8'] = physical h8
    color='black' -> squares['a1'] = physical h8, squares['h8'] = physical a1
                     (the 64 entries are reversed, matching the standard
                     "flip the board when you play the other side" trick)

Authoring notes
---------------
* Capture index 0 is always at the robot's NEAR-row on the column closest
  to the board, then fills that column toward the far row, then moves to
  the next column outward, etc.
* Home is at `home_offset` meters past the nearest-left square along the
  robot's left direction (i.e. further away from the board on the left).
"""

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import rospy
    from std_msgs.msg import String
    _ROS_AVAILABLE = True
except ImportError:  # pragma: no cover - usable without ROS installed
    _ROS_AVAILABLE = False


# =====================================================================
# --- Data classes ---
# =====================================================================

@dataclass
class BoardGeometry:
    """Physical constants of the chessboard."""
    square_size: float = 0.045
    board_outer_size: float = 0.360

    @property
    def play_area_size(self) -> float:
        return 8.0 * self.square_size

    @property
    def margin(self) -> float:
        return (self.board_outer_size - self.play_area_size) / 2.0


@dataclass
class LayoutConfig:
    """Tunable parameters for auxiliary (non-square) positions."""
    # Gap (m) between the outer edge of the board and the first aux column.
    aux_side_gap: float = 0.01
    # Capture / graveyard area
    capture_cols: int = 3
    capture_rows: int = 8
    # Home offset along robot's left direction from the nearest-left square.
    home_offset: float = 0.10
    # Home z (m). Usually equals safe_h.
    home_z: float = 0.12
    # Piece contact z (m) on the board surface.
    piece_z: float = 0.0
    # Promotion slot placement:
    # list of (letter, row_offset_from_near_row) with offset in [0..7].
    # Default places them at the far-row beside the board, matching where
    # pawns naturally promote.
    promotion_slots: Tuple[Tuple[str, int], ...] = (
        ('q', 7),  # far-row (row 8 for white, row 1 for black)
        ('r', 6),
        ('b', 5),
        ('n', 4),
    )


# =====================================================================
# --- ChessLayout ---
# =====================================================================

class ChessLayout:
    """Builds world-frame positions from calibration + color.

    This class is intentionally ROS-free so it can be unit-tested and
    reused in other scripts.
    """

    VALID_COLORS = ('white', 'black')

    def __init__(self,
                 color: str,
                 corner: np.ndarray,
                 e_h: np.ndarray,
                 e_N: np.ndarray,
                 board: Optional[BoardGeometry] = None,
                 config: Optional[LayoutConfig] = None):
        if color not in self.VALID_COLORS:
            raise ValueError(
                f"color must be one of {self.VALID_COLORS}, got {color!r}")
        self.color = color
        self.corner = np.asarray(corner, dtype=float)
        self.e_h = np.asarray(e_h, dtype=float)
        self.e_N = np.asarray(e_N, dtype=float)
        self.board = board or BoardGeometry()
        self.config = config or LayoutConfig()

        # Outputs
        self._abs_squares: Dict[str, Tuple[float, float, float]] = {}
        self.squares: Dict[str, Tuple[float, float, float]] = {}
        self.promotion: Dict[str, Tuple[float, float, float]] = {}
        self.graveyard: List[Tuple[float, float, float]] = []
        self.home: Tuple[float, float, float] = (0.0, 0.0, 0.0)

        self._build()

    # ---------- Geometry helpers (board-uv -> world xyz) ----------
    def _uv_to_world(self,
                     u: float,
                     v: float,
                     z: Optional[float] = None) -> Tuple[float, float, float]:
        if z is None:
            z = self.config.piece_z
        p = self.corner + u * self.e_h + v * self.e_N
        return (float(p[0]), float(p[1]), float(z))

    # ---------- Aux placement (robot-relative) ----------
    def _aux_u_of_col(self, col_idx: int) -> float:
        """u-coordinate of aux column index `col_idx` (0 = nearest the
        board edge, 1 = further out, etc.). Always on the robot's LEFT.
        """
        s = self.board.square_size
        m = self.board.margin
        gap = self.config.aux_side_gap
        if self.color == 'white':
            # Robot's left = -e_h. Columns extend into u < 0.
            # Center of the col'th column beyond the a-file outer edge.
            return -gap - (col_idx + 0.5) * s
        else:  # black
            # Robot's left = +e_h. Columns extend beyond u = board outer edge.
            u_edge = 2.0 * m + 8.0 * s
            return u_edge + gap + (col_idx + 0.5) * s

    def _aux_v_of_row_offset(self, row_offset_from_near: int) -> float:
        """v-coordinate for a slot that is `row_offset_from_near` squares
        away from the near-row (0 = near-row center, 7 = far-row center).
        """
        s = self.board.square_size
        m = self.board.margin
        if self.color == 'white':
            # Near-row = row 1. v grows toward row 8 (far).
            return m + (row_offset_from_near + 0.5) * s
        else:
            # Near-row = row 8. v shrinks toward row 1 (far).
            return m + (7.5 - row_offset_from_near) * s

    # ---------- Builders ----------
    def _build(self):
        self._build_abs_squares()
        self._build_squares_with_color()
        self._build_promotion()
        self._build_graveyard()
        self._build_home()

    def _build_abs_squares(self):
        s = self.board.square_size
        m = self.board.margin
        out: Dict[str, Tuple[float, float, float]] = {}
        for col_idx, col in enumerate('abcdefgh'):
            for row in range(1, 9):
                u = m + (col_idx + 0.5) * s
                v = m + (row - 1 + 0.5) * s
                out[f"{col}{row}"] = self._uv_to_world(u, v)
        self._abs_squares = out

    def _build_squares_with_color(self):
        """Maps chess notation to world position, mirroring for black.

        white: identity (a1 -> physical a1)
        black: 180 deg mapping (a1 -> physical h8, h8 -> physical a1, ...)
               Same 64-key reversal trick as the v3 `mirrored_squares`.
        """
        if self.color == 'white':
            self.squares = dict(self._abs_squares)
        else:
            keys = list(self._abs_squares.keys())
            rev = list(reversed(keys))
            self.squares = {
                keys[i]: self._abs_squares[rev[i]] for i in range(len(keys))
            }

    def _build_promotion(self):
        """Promotion slots on robot's LEFT column, at configured row offsets."""
        u = self._aux_u_of_col(0)  # adjacent to the board
        self.promotion = {}
        for letter, row_off in self.config.promotion_slots:
            v = self._aux_v_of_row_offset(row_off)
            self.promotion[letter] = self._uv_to_world(u, v)

    def _build_graveyard(self):
        """Capture/graveyard slots.

        Index 0 is the slot closest to the robot (near-row, nearest column).
        Fill order: column-by-column from near-column outward; within each
        column from near-row to far-row.
        """
        self.graveyard = []
        for col_g in range(self.config.capture_cols):
            u = self._aux_u_of_col(col_g)
            for r in range(self.config.capture_rows):
                v = self._aux_v_of_row_offset(r)
                self.graveyard.append(self._uv_to_world(u, v))

    def _build_home(self):
        """Home = nearest-left square center offset further left by home_offset.

        White: nearest-left square is a1. Offset in -e_h direction.
        Black: nearest-left square is h8. Offset in +e_h direction.
        """
        s = self.board.square_size
        m = self.board.margin
        if self.color == 'white':
            u = m + 0.5 * s - self.config.home_offset
            v = m + 0.5 * s
        else:
            u = m + 7.5 * s + self.config.home_offset
            v = m + 7.5 * s
        self.home = self._uv_to_world(u, v, z=self.config.home_z)

    # ---------- Introspection ----------
    def describe(self) -> str:
        lines = []
        lines.append(f"[ChessLayout] color={self.color}")
        lines.append(
            f"  corner = ({self.corner[0]:+.4f}, {self.corner[1]:+.4f})")
        lines.append(f"  e_h    = ({self.e_h[0]:+.4f}, {self.e_h[1]:+.4f})")
        lines.append(f"  e_N    = ({self.e_N[0]:+.4f}, {self.e_N[1]:+.4f})")
        lines.append(
            f"  square_size={self.board.square_size}, "
            f"margin={self.board.margin:.4f}")
        lines.append(f"  home_offset={self.config.home_offset}")
        lines.append(f"  squares['a1'] = {self.squares['a1']}")
        lines.append(f"  squares['h8'] = {self.squares['h8']}")
        lines.append(f"  home          = {self.home}")
        lines.append(f"  promotion[q]  = {self.promotion['q']}")
        lines.append(f"  graveyard[0]  = {self.graveyard[0]}   (near robot)")
        lines.append(
            f"  graveyard[-1] = {self.graveyard[-1]}   (far, outermost col)")
        return "\n".join(lines)


# =====================================================================
# --- ColorSubscriber (ROS) ---
# =====================================================================

class ColorSubscriber:
    """Subscribes to a std_msgs/String topic carrying 'white' or 'black'.

    Default topic: /chess/color
    Until a message arrives, `get()` returns the `initial` value.

    Use `set_on_change(cb)` to be notified when the color actually changes
    (or on first valid message). The callback runs in the ROS subscriber
    thread, keep it short or off-load work.
    """

    def __init__(self,
                 topic: str = '/chess/color',
                 initial: str = 'white',
                 latched: bool = True):
        if not _ROS_AVAILABLE:
            raise RuntimeError(
                "rospy is not available; ColorSubscriber needs ROS installed.")
        if initial not in ChessLayout.VALID_COLORS:
            raise ValueError(
                f"initial color must be one of {ChessLayout.VALID_COLORS}")
        self._lock = threading.Lock()
        self._color = initial
        self._got_msg = False
        self._callback = None
        self._topic = topic
        self._sub = rospy.Subscriber(topic, String, self._cb, queue_size=1)
        if latched:
            rospy.loginfo(
                f"[ColorSubscriber] listening on {topic!r}, "
                f"initial={initial!r}")

    def _cb(self, msg):
        val = msg.data.strip().lower() if msg.data else ''
        if val not in ChessLayout.VALID_COLORS:
            rospy.logwarn(
                f"[ColorSubscriber] ignoring invalid color: {msg.data!r}")
            return
        with self._lock:
            changed = (val != self._color) or (not self._got_msg)
            self._color = val
            self._got_msg = True
            cb = self._callback
        if changed and cb is not None:
            try:
                cb(val)
            except Exception as e:  # pragma: no cover
                rospy.logerr(f"[ColorSubscriber] callback error: {e}")

    def get(self) -> str:
        with self._lock:
            return self._color

    def has_received(self) -> bool:
        with self._lock:
            return self._got_msg

    def set_on_change(self, callback):
        with self._lock:
            self._callback = callback

    def wait_for_message(self, timeout: float = 2.0) -> bool:
        """Block up to `timeout` seconds for at least one message.

        Returns True if a message arrived, False on timeout.
        """
        start = rospy.Time.now()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            with self._lock:
                if self._got_msg:
                    return True
            if (rospy.Time.now() - start).to_sec() > timeout:
                return False
            rate.sleep()
        return False


# =====================================================================
# --- Demo / manual check ---
# =====================================================================

if __name__ == '__main__':
    # Quick manual check: print both layouts given a toy calibration.
    corner = np.array([0.398 + 7.5 * 0.045, -7.5 * 0.045])
    e_h = np.array([0.0, 1.0])
    e_N = np.array([-1.0, 0.0])

    for color in ChessLayout.VALID_COLORS:
        layout = ChessLayout(color, corner, e_h, e_N)
        print("=" * 62)
        print(layout.describe())
        print(f"  (corners check) a8 = {layout.squares['a8']}")
        print(f"  (corners check) h1 = {layout.squares['h1']}")

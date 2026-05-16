# مرجع: board_calibration.py

ملخّص بنية وحدة معايرة لوحة الشطرنج لروبوت Franka Panda.
الكود الفعلي في: `scripts/board_calibration.py`

## الهدف
بناء خرائط دقيقة للمواقع الديكارتية لمربعات الشطرنج (a1..h8) + مواقع
ترقية البيدق + 24 مربع للمقبرة + نقطة home، انطلاقاً من معايرة فيزيائية
عبر *probing* (لمس) لحواف اللوحة بالقابض.

## المكونات الرئيسية

### 1. الثوابت (وحدات SI)
- **هندسة اللوحة**: `BOARD_OUTER_SIZE = 0.360 m`, `SQUARE_SIZE = 0.045 m`,
  `MARGIN = 0.030 m`, `ORIGIN_X/Y/Z` نقطة مرجعية أولية.
- **الحركة**: `V_SLOW=A_SLOW=0.7`, `SAFE_H = 0.12 m`.
- **القابض**: `OPEN_WIDTH=0.04`, `CLOSE_WIDTH=0.025`, `GRIPPER_SPEED=0.1`,
  `GRIPPER_TIP_OFFSET = 0.018/2` (نصف عرض القابض).
- **probing**: `PROBE_Z=0.06`, `PROBE_TRANSIT_Z=SAFE_H+0.080`,
  `PROBE_FORCE_THRESH=4 N`, `PROBE_CONTACT_CONSEC=3` عينات متتالية،
  `PROBE_MIN_TRAVEL=5 mm` لتجاهل أول spike.
- **نقاط البدء**:
  - W1=(0.50, 0.20), W2=(0.55, 0.20) → اتجاه `-Y` بـyaw=π/2 (لمس حافة شمالية).
  - S1=(0.30,-0.20), S2=(0.30,-0.25) → اتجاه `+X` بـyaw=0 (لمس حافة شرقية).
- **ملف الحفظ**: `~/board_calibration.yaml`.

### 2. TF helpers
- `init_tf()`: ينشئ tf2 buffer/listener للتحويل من `panda1_link0` → `world`.
- `get_current_pose_in_ref()`: ترجع pose الحالي في إطار `world`.

### 3. دوال حركة
- `make_down_pose(x,y,z,yaw)`: pose مع orientation = down (roll=π).
- `move_to_pose(...)`: cartesian path + retime + execute.
- `gripper_full_close()`: يستخدم `MoveAction` (deprecated للـprobing لأنه ما بحافظ على القوة).
- `gripper_grasp_close()`: يستخدم `GraspAction` بقوة 20 N — **يُستخدم قبل كل probe** عشان الفكوك ما تنفتح تحت الحمل الجانبي.
- `_rot_2d`, `_yaw_from_pose`, `_wrap_angle`: utilities.

> **مهم**: `gripper_client` الممرَّر لـ`BoardCalibration` لازم يكون `SimpleActionClient`
> لـ`GraspAction` على `/panda1/franka_gripper/grasp` (وليس `/move`).

### 4. ForceMonitor
- يشترك في `/panda1/franka_state_controller/franka_states` ويقرأ `O_F_ext_hat_K`.
- `zero_bias()`: يأخذ متوسط 50 عينة عند 100 Hz كـbias.
- `get_xy_magnitude()`: |F_xy| - bias لاكتشاف اللمس.

### 5. BoardCalibration
يحتفظ بالحالة: `board_corner`, `e_h_axis` (a→h), `e_N_axis` (row1→row8),
`board_theta`, `square_positions`, `mirrored_squares`, `promotion_positions`,
`graveyard_positions`, `hx/hy/hz`.

#### `build_positions()`
- يبني 64 مربعاً من corner + محاور eh, eN.
- `mirrored_squares`: انعكاس للوصول من جهة الخصم.
- promotion (q,r,b,n): بعد حافة h-file بـ3.5 squares + 1cm ، محاذية للصفوف 1..4.
- graveyard: 24 موقعاً، 3 أعمدة × 8 صفوف، column-major.
- home: مزاح عن h8 بـ(-0.1, +0.3) عند SAFE_H.

#### `_probe_linear()`
الـprobing الفعلي: يحرك الذراع خطياً في اتجاه معطى، يراقب |F_xy|، ويوقف عند تجاوز
العتبة لـ`PROBE_CONTACT_CONSEC` عينات متتالية بشرط travel > `PROBE_MIN_TRAVEL`.
يقيس **delta_yaw** بين yaw البداية و yaw بعد الاستقرار، ويرجع (نقطة التلامس,
delta_yaw_settled). يحذر إذا الانحراف > 3°.

#### `_do_probe()`
سلسلة كاملة: full-close → ارتفاع transit → نقطة بدء → نزول إلى PROBE_Z →
`_probe_linear` → retreat بـ20mm → lift → return to transit Z.

#### `calibrate()` — Two-pass
- **Pass 1**: probe W1+W2 بـyaw الأصلي → تقدير θ من ميل الحافة.
- **Pass 2**: probe الأربعة (W1, W2, S1, S2) بـyaw/dir مدوّرة بـθ.
- يحسب eh_meas, eN_meas، يتحقق من perpendicularity (تحذير إن > 3°).
- يأخذ متوسط المحورين بعد إسقاط eN على نفس فضاء eh.
- يطبّق **gripper tip offset** على نقاط التلامس:
  - **مع** wrist-deflection correction (اختياري بسؤال): يدوّر offset بـdelta_yaw الفعلي.
  - **بدونها** (legacy): يستخدم direction الـnominal.
- يعيد حساب المحاور من النقاط المصححة.
- يحل intersection خطين (W1+t·eN ⟂ S1+s·eh) لإيجاد corner الفعلي.
- يحسب `theta_meas = arctan2(eh_avg[0], -eh_avg[1])`.

#### save/load YAML
يخزّن: corner_x/y, theta_rad, e_h_x/y, e_N_x/y, square_size, margin, board_outer_size.

#### show/test
`show`: طباعة الحالة. `test`: يحرك الذراع فوق a1, h1, h8, a8 ثم home بـSAFE_H.

## ملاحظات تصميمية مهمة
- الفرق بين `square_positions` و `mirrored_squares`: الاستخدام بالـindex
  (مثل قراءة كاميرا من زاوية الخصم) يستعمل `mirrored_squares`.
- الـyaw عند الحركات يطابق `board_theta` بعد المعايرة لتبقى pose طبيعية بالنسبة للوحة.
- العتبة 4 N والعينات المتتالية مهمتان لتفادي false-trigger من اهتزاز البداية.
- `GRIPPER_TIP_OFFSET = 0.018/2` يفترض أن TCP في مركز الفك، والاحتكاك يحصل
  على الحافة الخارجية بـ9mm إزاحة جانبية.

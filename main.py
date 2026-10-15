# -*- coding: utf-8 -*-
"""
手机 / 平板支架 —— 三阶段结构设计与力学分析工具 v10
==========================================================
v10 关键更新:
  - 每帧受力缓存 (Simulator.force_cache)
  - 受力可视化: 简略 / 详细 切换, 只显示选中零件的受力
  - 分力颜色: 重力黑 / 刚性蓝 / 刚度绿 / 阻尼橙 / 磁力紫
  - 力矩圆弧可视化
  - JSON 保存 / 加载方案
"""

import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import numpy as np
import math
import time
import json
import logging
import os

# ============================================================
#   仿真日志
#   文件名: <模型名>_<时间戳>.log
#   模型名: 未加载 JSON 时为脚本名; 加载 JSON 后自动切换为 JSON 主名
#   每次切换都会用新的时间戳新建一个日志文件
# ============================================================
logger = logging.getLogger('sim')
logger.setLevel(logging.DEBUG)     # 详细调试; 若嫌日志太多改成 INFO
logger.propagate = False           # 避免与 root logger 重复输出

_LOG_FILE_HANDLER = None           # 当前活动的 FileHandler
LOG_PATH = None                    # 当前日志文件全路径 (供界面展示)

def setup_log_for_model(model_name):
    """切换到 <model_name>_<时间戳>.log, 返回新日志文件的路径"""
    global _LOG_FILE_HANDLER, LOG_PATH
    if _LOG_FILE_HANDLER is not None:
        try:
            logger.removeHandler(_LOG_FILE_HANDLER)
            _LOG_FILE_HANDLER.close()
        except Exception:
            pass
        _LOG_FILE_HANDLER = None
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = f'{model_name}_{ts}.log'
    h = logging.FileHandler(path, mode='w', encoding='utf-8')
    h.setFormatter(logging.Formatter(
        '%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'))
    logger.addHandler(h)
    _LOG_FILE_HANDLER = h
    LOG_PATH = path
    return path

# 初始不再自动创建日志文件，调整为点击“开始仿真”后再创建
logger.addHandler(logging.NullHandler())

# ============================================================
#   受力可视化配置 —— 全部可在此处调整
# ============================================================
FORCE_VIZ = {
    'total_f':    {'color': '#c62828', 'label': '合力',     'width': 4},
    'total_m':    {'color': '#ad1457', 'label': '合力矩',   'width': 3},
    'gravity':    {'color': '#000000', 'label': '重力',     'width': 3},
    'revolute':   {'color': '#1565c0', 'label': '刚性接触', 'width': 3},
    'revolute_m': {'color': '#0d47a1', 'label': '铰链力矩', 'width': 3},
    'spring':     {'color': '#2e7d32', 'label': '弹簧刚度', 'width': 3},
    'spring_lim': {'color': '#880e4f', 'label': '限位(含阻尼)', 'width': 3},
    'damping':    {'color': '#ff8c00', 'label': '阻尼力',   'width': 3},
    'magnet':     {'color': '#7e57c2', 'label': '磁力',     'width': 3},
    'payload':    {'color': '#00695c', 'label': '载荷力',   'width': 3},
    'other':      {'color': '#555555', 'label': '其他力',   'width': 3},
}

FORCE_SCALE_PX_PER_N = 22.0     # 每牛顿对应的世界像素长度
FORCE_MAX_WORLD_PX   = 160.0    # 单根力箭头最大世界像素长度
FORCE_MIN_WORLD_PX   = 6.0      # 小于此长度不绘制

MOMENT_SCALE_PX_PER_NM = 18.0   # 每牛·米对应的弧半径 (世界像素)
MOMENT_MAX_WORLD_PX    = 75.0   # 力矩弧最大半径
MOMENT_MIN_WORLD_PX    = 12.0   # 力矩弧最小半径

FORCE_LABEL_FONT  = ('Consolas', 8, 'bold')
FORCE_LABEL_DX    = 6
FORCE_LABEL_DY    = -6

ANALYSIS_LABELS = {
    'off':      '受力: 关',
    'simple':   '受力: 简略',
    'detailed': '受力: 详细',
}

# ============================================================
#   常量
# ============================================================
PX_PER_M   = 1000.0
GRAVITY    = 9.81 * PX_PER_M
SUBSTEPS   = 8
ITERATIONS = 16
DT_MAX     = 0.033
ANG_DAMP   = 1.5
LIN_DAMP   = 0.8
SNAP_PX    = 26.0

DRAG_MAX_STEP   = 60.0
DRAG_ITER       = 12
DRAG_TOL        = 0.15
CORRECTION_ITER = 8
FEASIBILITY_ITER = 60
FEASIBILITY_TOL  = 0.5

# 弹簧限位模型
LIMIT_STIFFNESS_MULT = 39.0    # 限位额外刚度倍数 (总刚度 = k * (1+39) = 40k)
LIMIT_STOP_ITER      = 2       # 限位速度截停的 Gauss-Seidel 迭代次数
                               # (一次足够使 v_rel=0; 两次防止多弹簧同时触发时耦合)
FRICTION_ITER        = 2       # 转动副静摩擦的 Gauss-Seidel 迭代次数
                               # (单铰链一次即精确; 多铰链耦合时迭代收敛)

SIM_FIXED_DT        = 1.0 / 60.0
SIM_STEPS_PER_BATCH = 8
SIM_FRAME_INTERVAL  = 2
SIM_MAX_FRAMES      = 2500
STABLE_VEL_TOL      = 8.0
STABLE_ANG_TOL      = 0.15
STABLE_FRAMES_REQ   = 40
MAX_LIN_VEL         = 4000.0
MAX_ANG_VEL         = 40.0
POS_LIMIT           = 5e5
REG_EPS             = 1e-8

CP_GEOM = {
    'revolute': ('circle', 9.0),
    'spring':   ('square', 8.0),
    'magnet':   ('diamond', 8.0),
    'payload':  ('circle', 9.0),
}

STAGE_COLORS = {
    'design':   ('#4a90d9', '① 零部件设计'),
    'assemble': ('#e69500', '② 装配'),
    'simulate': ('#7cb342', '③ 仿真计算'),
}


def rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def perp(v):
    return np.array([-v[1], v[0]])


# ============================================================
#   几何工具
# ============================================================
def cp_vertices_local(cp):
    shape, r = CP_GEOM.get(cp.type, ('circle', 7.0))
    c = cp.local
    if shape == 'circle':
        return [c + r * np.array([math.cos(2*math.pi*i/8),
                                   math.sin(2*math.pi*i/8)])
                for i in range(8)]
    if shape == 'square':
        return [c + np.array([-r, -r]), c + np.array([r, -r]),
                c + np.array([r, r]),   c + np.array([-r, r])]
    if shape == 'diamond':
        return [c + np.array([0, -r]), c + np.array([r, 0]),
                c + np.array([0, r]),  c + np.array([-r, 0])]
    return []


def convex_hull(points):
    if len(points) <= 2:
        return list(points)
    pts = sorted(points, key=lambda p: (p[0], p[1]))

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-9:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-9:
            upper.pop()
        upper.append(p)
    h = lower[:-1] + upper[:-1]
    if len(h) < 3:
        return [pts[0], pts[-1]]
    return h


def compute_part_envelope(part, margin=10.0):
    pts = []
    for cp in part.points:
        pts.extend(cp_vertices_local(cp))
    if not pts:
        s = 20.0
        return [np.array([-s, -s]), np.array([s, -s]),
                np.array([s, s]),   np.array([-s, s])]
    arr = np.array([(p[0], p[1]) for p in pts])
    hull_pts = convex_hull([np.array(p) for p in arr])
    if len(hull_pts) < 3:
        x0, y0 = arr.min(axis=0)
        x1, y1 = arr.max(axis=0)
        pad = margin
        if x1 - x0 < 1e-6: x0 -= 10; x1 += 10
        if y1 - y0 < 1e-6: y0 -= 10; y1 += 10
        return [np.array([x0-pad, y0-pad]), np.array([x1+pad, y0-pad]),
                np.array([x1+pad, y1+pad]), np.array([x0-pad, y1+pad])]
    c = np.mean(hull_pts, axis=0)
    result = []
    for v in hull_pts:
        d = v - c
        L = float(np.linalg.norm(d))
        result.append(v + d / L * margin if L > 1e-6 else v.copy())
    return result


def point_seg_dist(px, py, a, b):
    ab = b - a
    L2 = float(ab.dot(ab))
    if L2 < 1e-9:
        return math.hypot(px - a[0], py - a[1])
    t = max(0.0, min(1.0, ((px - a[0])*ab[0] + (py - a[1])*ab[1]) / L2))
    proj = a + t * ab
    return math.hypot(px - proj[0], py - proj[1])


def convex_polygons_sat(poly_a, poly_b):
    """二维凸多边形 SAT 碰撞检测。

    返回 (overlap, normal, depth):
      overlap: bool
      normal:  np.array([nx, ny]) 从 A 指向 B 的单位法线
      depth:   float 最小穿透深度 (沿 normal 分离所需距离)
    """
    if len(poly_a) < 3 or len(poly_b) < 3:
        return False, None, 0.0
    best_depth = float('inf')
    best_axis = None

    for poly in (poly_a, poly_b):
        n = len(poly)
        for i in range(n):
            p1 = poly[i]; p2 = poly[(i + 1) % n]
            ex = float(p2[0] - p1[0]); ey = float(p2[1] - p1[1])
            ax = -ey; ay = ex
            nrm = math.hypot(ax, ay)
            if nrm < 1e-9:
                continue
            ax /= nrm; ay /= nrm
            proj_a = [ax * float(q[0]) + ay * float(q[1]) for q in poly_a]
            proj_b = [ax * float(q[0]) + ay * float(q[1]) for q in poly_b]
            min_a, max_a = min(proj_a), max(proj_a)
            min_b, max_b = min(proj_b), max(proj_b)
            if max_a < min_b or max_b < min_a:
                return False, None, 0.0
            overlap1 = max_a - min_b
            overlap2 = max_b - min_a
            if overlap1 < overlap2:
                overlap = overlap1
                dirx, diry = ax, ay
            else:
                overlap = overlap2
                dirx, diry = -ax, -ay
            if overlap < best_depth:
                best_depth = overlap
                best_axis = np.array([dirx, diry])

    if best_axis is None:
        return False, None, 0.0
    return True, best_axis, best_depth


# ============================================================
#   数据模型
# ============================================================
class ConnPoint:
    _pid = 0
    def __init__(self, part, local, type_='revolute'):
        ConnPoint._pid += 1
        self.id = ConnPoint._pid
        self.part = part
        self.local = np.array(local, dtype=float)
        self.type = type_
        self.damping = 2.0
        self.mass = 0.0
        self.link = None
        self.broken = False
        self.static_damping = 0.0   # 静态阻尼力矩 (N·m, 仅 revolute 使用)

    def world(self, pos=None, ang=None):
        if pos is None: pos = self.part.pos
        if ang is None: ang = self.part.angle
        return pos + rot(ang) @ self.local

    def world_pred(self):
        return self.part.pred_pos + rot(self.part.pred_angle) @ self.local


class Part:
    _ptid = 0
    def __init__(self, pos, hull_local=None, name=None):
        Part._ptid += 1
        self.id = Part._ptid
        self.name = name or f"零件{self.id}"
        if hull_local is None:
            s = 20.0
            hull_local = [(-s, -s), (s, -s), (s, s), (-s, s)]
        self.hull = [np.array(v, dtype=float) for v in hull_local]
        self.points = []
        self.locked = False
        self.body_mass = 0.20
        self.pos   = np.array(pos, dtype=float)
        self.angle = 0.0
        self.v     = np.zeros(2)
        self.omega = 0.0
        self.pred_pos   = self.pos.copy()
        self.pred_angle = 0.0
        self.f  = np.zeros(2)
        self.tq = 0.0

    def l2w(self, local):
        return self.pos + rot(self.angle) @ local

    def w2l(self, world):
        return rot(-self.angle) @ (world - self.pos)

    def com_local(self):
        """局部坐标系下的质心 (用凹包均值近似)"""
        if not self.hull:
            return np.zeros(2)
        return np.mean(self.hull, axis=0)

    def all_mass(self):
        m = self.body_mass
        for p in self.points:
            if p.type == 'payload':
                m += p.mass
        return max(m, 1e-4)

    def inertia(self):
        n = len(self.hull)
        if n < 3:
            return self.all_mass() * 500.0
        A = 0.0; I = 0.0
        for i in range(n):
            x0, y0 = self.hull[i]
            x1, y1 = self.hull[(i+1) % n]
            cr = x0*y1 - x1*y0
            A += cr
            I += cr * (x0*x0 + x0*x1 + x1*x1 + y0*y0 + y0*y1 + y1*y1)
        A *= 0.5
        if abs(A) < 1e-6:
            return self.all_mass() * 500.0
        rho = self.all_mass() / abs(A)
        return max(rho * I / 12.0, 1e-3)

    def recenter(self):
        if not self.hull: return
        c = np.mean(self.hull, axis=0)
        if np.linalg.norm(c) < 1e-9: return
        for i in range(len(self.hull)):
            self.hull[i] = self.hull[i] - c
        for p in self.points:
            p.local = p.local - c
        R = rot(self.angle)
        self.pos = self.pos + R @ c
        self.pred_pos = self.pos.copy()
        self.pred_angle = self.angle


class Link:
    _lid = 0
    def __init__(self, pA, pB, type_):
        Link._lid += 1
        self.id = Link._lid
        self.pA = pA; self.pB = pB
        self.type = type_
        pA.link = pB; pB.link = pA
        self.force = np.zeros(2)
        self.lmin  = 50.0
        self.l0    = 100.0
        self.lmax  = 200.0
        self.k     = 100.0
        self.c     = 5.0
        self.mk        = 500.0
        self.strength  = 3000.0
        self.broken    = False
        self.limit_active = False       # 弹簧限位是否已触发 (用于日志状态切换)
        self.static_damping = 0.0       # 静态摩擦极限力矩 (N·m, 仅 revolute)
        self.moment = 0.0               # 铰链"实际提供"的力矩 (kg·px²/s²)
                                        # 静摩擦时=维持平衡所需力矩
                                        # 滑动摩擦时=被钳制到 ±M_max
        self.moment_raw = 0.0           # 静力平衡解 (未钳制的"所需力矩")
        self.friction_saturated = False # 本轮是否已进入滑动摩擦


# ============================================================
#   仿真管理器
# ============================================================
class Simulator:
    def __init__(self, app):
        self.app = app
        self.snapshot_pre_sim = None
        self.frames = []
        self.force_cache = []          # 与 frames 一一对应: {part_id: {cp_id: {...}}}
        self.energy_cache = []         # 能量缓存
        self.initial_ep_g = 0.0        # 初始重力势能基准
        self.state = 'idle'
        self.revision = -1
        self.playhead = 0
        self.playing = False
        self.last_error = None
        self.stable_count = 0
        self.step_counter = 0
        self._gen = 0

    def capture_pre_sim(self):
        self.snapshot_pre_sim = {}
        for p in self.app.parts:
            self.snapshot_pre_sim[p.id] = (
                p.pos.copy(), p.angle, p.v.copy(), p.omega)

    def restore_pre_sim(self):
        if self.snapshot_pre_sim is None:
            return
        for p in self.app.parts:
            if p.id in self.snapshot_pre_sim:
                pos, ang, v, om = self.snapshot_pre_sim[p.id]
                p.pos = pos.copy()
                p.angle = ang
                p.v = v.copy()
                p.omega = om
                p.pred_pos = p.pos.copy()
                p.pred_angle = p.angle

    def invalidate(self):
        self._gen += 1
        self.frames.clear()
        self.force_cache.clear()
        self.energy_cache.clear()
        self.initial_ep_g = 0.0
        self.state = 'idle'
        self.playhead = 0
        self.playing = False
        self.stable_count = 0
        self.step_counter = 0
        self.last_error = None
        self.snapshot_pre_sim = None

    def _ensure_fresh(self):
        if self.revision != self.app.data_revision:
            self.invalidate()
            self.revision = self.app.data_revision

    def _append_frame(self):
        frame = {}
        for p in self.app.parts:
            frame[p.id] = (float(p.pos[0]), float(p.pos[1]), float(p.angle))
        self.frames.append(frame)
        # 每帧同步计算受力
        forces = self._compute_frame_forces()
        self.force_cache.append(forces)
        # 同步计算能量
        eng = self._compute_energies()
        if len(self.energy_cache) == 0:
            self.initial_ep_g = eng['ep_g']
        eng['ep_g'] -= self.initial_ep_g  # 相对初始位形
        eng['e_tot'] = eng['ek'] + eng['ep_g'] + eng['ep_e']
        self.energy_cache.append(eng)

    def _compute_energies(self):
        """计算当前帧系统的各种能量 (焦耳 J)"""
        ek = 0.0
        ep_g = 0.0
        ep_e = 0.0
        for p in self.app.parts:
            if p.locked: continue
            v_m = np.linalg.norm(p.v) / PX_PER_M
            ek += 0.5 * p.all_mass() * (v_m ** 2)
            I_kgm2 = p.inertia() / (PX_PER_M ** 2)
            ek += 0.5 * I_kgm2 * (p.omega ** 2)
            
            com_world = p.l2w(p.com_local())
            ep_g += p.all_mass() * 9.81 * (-com_world[1] / PX_PER_M)
            
        for lk in self.app.links:
            if lk.type == 'spring':
                L = float(np.linalg.norm(lk.pB.world() - lk.pA.world()))
                dx_m = (L - lk.l0) / PX_PER_M
                ep_e += 0.5 * lk.k * (dx_m ** 2)
                
        return {'ek': ek, 'ep_g': ep_g, 'ep_e': ep_e}

    def _compute_frame_forces(self):
        """计算当前帧每个零件每个连接附的受力分解

        返回数据结构 (每个零件):
          {
            'com_world':    (2,) 世界坐标质心
            'gravity':      (2,) 重力向量
            'cp_forces':    {cp_id: {
                'components': [(kind, F(2,), at(2,)), ...],
                'total_f':    (2,) 该挂接点上的合力,
                'at':         (2,) 挂接点世界位置,
            }}
            'total_force':  (2,) 零件总合力 (含重力)
            'total_moment': scalar 零件总合力矩 (关于质心, 含重力)
          }
        """
        self.app._recompute_analysis_for_current_frame()

        result = {}
        for p in self.app.parts:
            com_world = p.l2w(p.com_local())
            cp_forces = {}
            p_torques = []
            # 重力只算本体质量 (payload 力单独作为分量作用于挂接点)
            gravity = np.array([0.0, p.body_mass * GRAVITY])

            # 累计: 本体重力 (作用在质心, 关于质心力矩为 0)
            total_force = gravity.copy()
            total_moment = 0.0

            for cp in p.points:
                at_world = cp.world()
                components = []           # [(kind, F_vec, at), ...]
                moments    = []           # [(kind, M_scalar, at), ...]

                # ---- 1) 刚性接触反力 + 铰链力矩 (revolute) ----
                for lk in self.app.links:
                    if lk.type != 'revolute': continue
                    if lk.pA is cp:
                        components.append(('revolute',
                                           -lk.force.copy(), at_world))
                        if abs(lk.moment) > 1e-6:
                            moments.append(('revolute_m',
                                            -lk.moment, at_world))
                    elif lk.pB is cp:
                        components.append(('revolute',
                                            lk.force.copy(), at_world))
                        if abs(lk.moment) > 1e-6:
                            moments.append(('revolute_m',
                                            lk.moment, at_world))

                # ---- 2) 弹簧: 分解为刚度力 / 限位力 / 阻尼力 ----
                # 说明: 限位力同时包含限位刚度与限位单向阻尼
                for lk in self.app.links:
                    if lk.type != 'spring': continue
                    if lk.pA is not cp and lk.pB is not cp: continue
                    sign = 1.0 if lk.pA is cp else -1.0
                    parts = self.app._spring_force_components(lk)
                    for key, kind in (('base', 'spring'),
                                      ('limit', 'spring_lim'),
                                      ('damp', 'damping')):
                        F = parts[key]
                        if float(np.linalg.norm(F)) < 1e-9: continue
                        components.append((kind, sign * F, at_world))

                # ---- 3) 磁力 ----
                for lk in self.app.links:
                    if lk.type != 'magnet' or lk.broken: continue
                    if lk.pA is not cp and lk.pB is not cp: continue
                    other = lk.pB if lk.pA is cp else lk.pA
                    wa = at_world; wb = other.world()
                    d = wb - wa
                    L = float(np.linalg.norm(d))
                    if L < 1e-9: continue
                    u = d / L
                    F_mag = lk.mk * (L - lk.l0)
                    components.append(('magnet', F_mag * u, at_world))

                # ---- 4) payload 载荷力 (恒向下, 作用在挂接点) ----
                if cp.type == 'payload' and cp.mass > 0:
                    F_pl = np.array([0.0, cp.mass * GRAVITY])
                    components.append(('payload', F_pl, at_world))

                # 该挂接点上的合力 (含所有分量, 用于简略显示)
                cp_total_f = np.zeros(2)
                for _, F, _ in components:
                    cp_total_f = cp_total_f + F

                # 只把"静态分量"累加进零件整体平衡
                cp_static_f = np.zeros(2)
                cp_total_m = 0.0
                for kind, F, at in components:
                    if kind == 'damping':
                        continue
                    cp_static_f = cp_static_f + F
                    r = at - com_world
                    m_add = r[0]*F[1] - r[1]*F[0]
                    cp_total_m += m_add
                    
                    if abs(m_add) > 1e-6:
                        p_torques.append({
                            'label': f"附加力矩 (点#{cp.id}-{FORCE_VIZ.get(kind,{}).get('label', kind)})",
                            'M': m_add,
                            'kind': kind,
                            'is_direct': False
                        })

                # 铰链力矩: 直接加总 (关于质心的力矩也是 M 本身)
                for kind, M, at in moments:
                    cp_total_m += M
                    if abs(M) > 1e-6:
                        p_torques.append({
                            'label': f"直接力矩 (点#{cp.id}-{FORCE_VIZ.get(kind,{}).get('label', kind)})",
                            'M': M,
                            'kind': kind,
                            'is_direct': True
                        })

                total_force = total_force + cp_static_f
                total_moment += cp_total_m

                cp_forces[cp.id] = {
                    'components': components,
                    'moments':    moments,
                    'total_f': cp_total_f,
                    'total_m': cp_total_m,
                    'at': at_world,
                }

            result[p.id] = {
                'cp_forces': cp_forces,
                'gravity': gravity,
                'com_world': com_world,
                'total_force': total_force,
                'total_moment': total_moment,
                'torques': p_torques,
            }
        return result

    def apply_frame(self, idx):
        if not self.frames:
            return
        idx = max(0, min(len(self.frames) - 1, idx))
        self.playhead = idx
        frame = self.frames[idx]
        for p in self.app.parts:
            if p.id in frame:
                px, py, ang = frame[p.id]
                p.pos = np.array([px, py])
                p.angle = ang
                p.pred_pos = p.pos.copy()
                p.pred_angle = ang
                p.v[:] = 0.0
                p.omega = 0.0

    def start(self):
        self._ensure_fresh()
        if self.state in ('stable', 'error'):
            self.restore_pre_sim()
            self.frames.clear()
            self.force_cache.clear()
            self.energy_cache.clear()
            self.initial_ep_g = 0.0
            self.playhead = 0
            self.stable_count = 0
            self.step_counter = 0
        if self.snapshot_pre_sim is None:
            self.capture_pre_sim()
        for p in self.app.parts:
            p.v[:] = 0.0
            p.omega = 0.0
            p.pred_pos = p.pos.copy()
            p.pred_angle = p.angle
        self.frames.clear()
        self.force_cache.clear()
        self.energy_cache.clear()
        self.initial_ep_g = 0.0
        self._append_frame()
        self.state = 'computing'
        self.playing = False
        self._gen += 1
        gen = self._gen
        logger.info('=' * 60)
        logger.info(f'[start] 仿真开始 gen={gen} 零件={len(self.app.parts)} '
                    f'连接={len(self.app.links)}')
        for p in self.app.parts:
            logger.info(f'  part#{p.id} "{p.name}" locked={p.locked} '
                        f'm={p.all_mass():.4f}kg I={p.inertia():.3e} '
                        f'pos=({p.pos[0]:.1f},{p.pos[1]:.1f}) '
                        f'angle={math.degrees(p.angle):.2f}deg')
        for lk in self.app.links:
            logger.info(f'  link#{lk.id} type={lk.type} '
                        f'pA={lk.pA.id}@part{lk.pA.part.id} '
                        f'pB={lk.pB.id}@part{lk.pB.part.id} '
                        f'k={lk.k:.1f} c={lk.c:.1f} l0={lk.l0:.1f} '
                        f'lmax={lk.lmax:.1f}')
        self.app.root.after(1, lambda: self._tick(gen))
        self.app.set_status('仿真中...')

    def stop(self):
        if self.state == 'computing':
            self._gen += 1
            self.state = 'stable' if self.frames else 'idle'
            logger.info(f'[stop] 手动停止 帧数={len(self.frames)}')

    def _tick(self, gen):
        if gen != self._gen:
            return
        if self.state != 'computing':
            return
        try:
            for _ in range(SIM_STEPS_PER_BATCH):
                self.app.substep(SIM_FIXED_DT / SUBSTEPS)
                self.step_counter += 1
                if self.step_counter % SIM_FRAME_INTERVAL == 0:
                    self._append_frame()
                    if len(self.frames) >= SIM_MAX_FRAMES:
                        self.state = 'stable'
                        logger.warning(
                            f'[maxframes] 到达帧上限 {SIM_MAX_FRAMES}, '
                            f'step={self.step_counter}')
                        self.app.set_status(
                            f'已达到帧缓存上限 ({SIM_MAX_FRAMES}), 仿真停止')
                        return

                ok, msg = self._check_numerical_health()
                if not ok:
                    self.state = 'error'
                    self.last_error = msg
                    logger.error(f'[numerical] 病态终止 at step={self.step_counter} '
                                 f'frame={len(self.frames)}: {msg}')
                    for p in self.app.parts:
                        if p.locked: continue
                        logger.error(
                            f'  part#{p.id} "{p.name}" '
                            f'pos=({p.pos[0]:.2f},{p.pos[1]:.2f}) '
                            f'angle={p.angle:.3f} '
                            f'v=({p.v[0]:.2f},{p.v[1]:.2f}) '
                            f'w={p.omega:.3f}')
                    self.app.set_status(f'仿真终止: {msg}')
                    messagebox.showerror(
                        '数值病态',
                        f'仿真已终止, 原因:\n{msg}\n\n'
                        f'已缓存 {len(self.frames)} 帧, 可回放查看。')
                    self.app.refresh_sim_ui()
                    self.app.redraw()
                    return

                if self._is_stable():
                    self.stable_count += 1
                    if self.stable_count >= STABLE_FRAMES_REQ:
                        self.state = 'stable'
                        logger.info(
                            f'[stable] 到达稳态 step={self.step_counter} '
                            f'frames={len(self.frames)}')
                        # 输出最后一帧每零件平衡检查
                        if self.force_cache:
                            last = self.force_cache[-1]
                            for pid, d in last.items():
                                tf = d.get('total_force', np.zeros(2))
                                tm = d.get('total_moment', 0.0)
                                Fm = float(np.linalg.norm(tf)) / PX_PER_M
                                M = float(tm) / PX_PER_M
                                logger.info(
                                    f'  part#{pid} totalF=({tf[0]/PX_PER_M:+.4f},'
                                    f'{tf[1]/PX_PER_M:+.4f}) N  |F|={Fm:.4f} N  '
                                    f'M={M:+.5f} N.m')
                        self.app.set_status(
                            f'仿真到达稳态 (共 {len(self.frames)} 帧)')
                        self.app.refresh_sim_ui()
                        self.app.redraw()
                        return
                else:
                    self.stable_count = 0

                # 每 20 步记录一次运行状态 (含每根弹簧的完整状态)
                if self.step_counter % 20 == 0:
                    lines = [f'[RUN] step={self.step_counter} '
                             f'frame={len(self.frames)}']
                    for p in self.app.parts:
                        if p.locked: continue
                        lines.append(
                            f'  part#{p.id} "{p.name}" '
                            f'pos=({p.pos[0]:+.2f},{p.pos[1]:+.2f}) '
                            f'ang={math.degrees(p.angle):+.2f}deg '
                            f'v=({p.v[0]:+.3f},{p.v[1]:+.3f}) '
                            f'w={p.omega:+.4f}')
                    for lk in self.app.links:
                        if lk.type != 'spring': continue
                        L = float(np.linalg.norm(
                            lk.pB.world() - lk.pA.world()))
                        hit = (L <= lk.lmin or L >= lk.lmax)
                        va = lk.pA.part.v + lk.pA.part.omega * perp(
                            rot(lk.pA.part.angle) @ lk.pA.local)
                        vb = lk.pB.part.v + lk.pB.part.omega * perp(
                            rot(lk.pB.part.angle) @ lk.pB.local)
                        dvec = lk.pB.world() - lk.pA.world()
                        if np.linalg.norm(dvec) > 1e-9:
                            u = dvec / np.linalg.norm(dvec)
                            v_rel = float((vb - va) @ u)
                        else:
                            v_rel = 0.0
                        lines.append(
                            f'  spring#{lk.id} L={L:.4f} '
                            f'limits=[{lk.lmin:.2f}, {lk.lmax:.2f}] l0={lk.l0:.2f} '
                            f'v_rel={v_rel:+.4f} hit={hit} '
                            f'active={lk.limit_active}')
                    logger.debug('\n'.join(lines))
        except Exception as ex:
            self.state = 'error'
            self.last_error = str(ex)
            logger.exception(f'[exception] step={self.step_counter} '
                             f'frame={len(self.frames)}: {ex}')
            self.app.set_status(f'仿真异常: {ex}')
            self.app.refresh_sim_ui()
            return
        self.app.refresh_sim_ui()
        self.app.redraw()
        self.app.root.after(1, lambda: self._tick(gen))

    def _is_stable(self):
        for p in self.app.parts:
            if p.locked: continue
            if float(np.linalg.norm(p.v)) > STABLE_VEL_TOL: return False
            if abs(p.omega) > STABLE_ANG_TOL: return False
        return True

    def _check_numerical_health(self):
        for p in self.app.parts:
            if p.locked: continue
            if not (np.isfinite(p.pos).all() and np.isfinite(p.angle)
                    and np.isfinite(p.v).all() and np.isfinite(p.omega)):
                return False, f'零件 #{p.id} 出现 NaN / Inf'
            sp = float(np.linalg.norm(p.v))
            if sp > MAX_LIN_VEL:
                return False, f'零件 #{p.id} 线速度溢出 ({sp:.0f} px/s)'
            if abs(p.omega) > MAX_ANG_VEL:
                return False, f'零件 #{p.id} 角速度溢出 ({p.omega:.1f} rad/s)'
            if float(np.linalg.norm(p.pos)) > POS_LIMIT:
                return False, f'零件 #{p.id} 位置飞出'
        return True, ''

    def play(self):
        if not self.frames: return
        self.state = 'playback'
        self.playing = True

    def pause_playback(self):
        self.playing = False


# ============================================================
#   主应用
# ============================================================
class App:
    def __init__(self, root):
        self.root = root
        root.title("手机 / 平板支架 —— 三阶段结构设计与力学分析 (v10)")
        root.configure(bg='#2b2b2b')

        self.parts = []
        self.links = []
        self.collisions = []           # [(partA, partB), ...] 碰撞约束对
        self.stage = 'design'
        self.mode  = 'select'
        self.data_revision = 0

        self.sel = None
        self.assemble_sel = []
        self.analysis_detail = 'detailed'   # 默认选用详细受力进行显示
        self.show_spring_limits = False     # 仿真阶段是否常驻显示弹簧限位环

        # 右侧属性面板: 静态结构 vs 动态数值分离
        self._props_key = None         # 当前面板对应的唯一 key (变化时才重建)
        self._props_dynamic = []       # [(widget, updater_fn)] 用于逐帧只刷新数值

        self.zoom   = 1.0
        self.view_x = 0.0
        self.view_y = 0.0

        self.mouse_screen = (0, 0)
        self.mouse_world  = np.zeros(2)
        self.panning = False
        self.pan_start = None
        self.pan_view_start = None

        self.dragging = None
        self.drag_mouse0 = None
        self.drag_obj0   = None
        self.drag_grab_local = None

        self.create_part_start = None
        self._snap_target = None
        
        # 能量图表交互状态
        self.eng_auto_scale = True
        self.eng_ranges = [{'vmin': 0.0, 'vmax': 1.0} for _ in range(4)]
        self.eng_pan_idx = -1
        self.eng_pan_y = 0
        self.eng_pan_vmin = 0.0
        self.eng_pan_vmax = 1.0
        self.eng_panning = False

        self.model_name = os.path.splitext(os.path.basename(__file__))[0]
        self.sim = Simulator(self)

        self._build_ui()
        self.build_demo()
        self._sim_playback_timer = False
        self._suppress_slider = False
        self.root.after(150, self._initial_fit)
        self.tick()

    def _initial_fit(self):
        self.fit_view()

    # ---------------- 坐标变换 ----------------
    def s2w(self, sx, sy):
        return np.array([sx / self.zoom + self.view_x,
                         sy / self.zoom + self.view_y])

    def w2s(self, wp):
        return ((wp[0] - self.view_x) * self.zoom,
                (wp[1] - self.view_y) * self.zoom)

    def w2sx(self, x): return (x - self.view_x) * self.zoom
    def w2sy(self, y): return (y - self.view_y) * self.zoom

    def _touch(self):
        self.data_revision += 1
        self.sim.invalidate()

    # ---------------- UI ----------------
    def _build_ui(self):
        tabbar = tk.Frame(self.root, bg='#151515', height=46)
        tabbar.pack(side='top', fill='x')
        tabbar.pack_propagate(False)
        self.tab_btns = {}
        for key in ('design', 'assemble', 'simulate'):
            b = tk.Button(tabbar, text=STAGE_COLORS[key][1],
                          relief='flat', bg='#151515', fg='#777777',
                          font=('Microsoft YaHei', 11, 'bold'),
                          padx=26, pady=10, borderwidth=0,
                          activebackground='#222222',
                          command=lambda k=key: self.set_stage(k))
            b.pack(side='left', fill='y')
            self.tab_btns[key] = b

        self.toolbar = tk.Frame(self.root, bg='#2b2b2b', height=40)
        self.toolbar.pack(side='top', fill='x')
        self.toolbar.pack_propagate(False)

        body = tk.Frame(self.root, bg='#2b2b2b')
        body.pack(side='top', fill='both', expand=True)

        self.canvas = tk.Canvas(body, bg='#fafaf5', highlightthickness=0)
        self.canvas.pack(side='left', fill='both', expand=True)

        right = tk.Frame(body, width=420, bg='#3c3f41')  # 拓宽右侧面板
        right.pack(side='right', fill='y')
        right.pack_propagate(False)
        
        self.props_title = tk.Label(right, text='属性 / 分析结果',
                                    bg='#3c3f41', fg='#ffffff',
                                    font=('Microsoft YaHei', 11, 'bold'))
        self.props_title.pack(pady=8)
        
        # 属性详情区
        self.props_inner = tk.Frame(right, bg='#3c3f41')
        self.props_inner.pack(side='top', fill='both', expand=True, padx=6)

        # 能量统计区
        self.energy_frame = tk.Frame(right, bg='#2b2b2b', height=240)
        self.energy_frame.pack(side='bottom', fill='x')
        self.energy_frame.pack_propagate(False)
        
        eng_title_bar = tk.Frame(self.energy_frame, bg='#2b2b2b')
        eng_title_bar.pack(fill='x', pady=(4, 0), padx=8)
        tk.Label(eng_title_bar, text='能量统计 (J)', bg='#2b2b2b', fg='#dddddd',
                 font=('Microsoft YaHei', 9, 'bold')).pack(side='left')
        tk.Label(eng_title_bar, text='(滚轮缩放 / 拖拽平移纵轴 / 双击恢复)', bg='#2b2b2b', fg='#777777',
                 font=('Microsoft YaHei', 8)).pack(side='right')

        self.energy_canvas = tk.Canvas(self.energy_frame, bg='#2b2b2b', highlightthickness=0)
        self.energy_canvas.pack(fill='both', expand=True, padx=4, pady=4)
        
        self.energy_canvas.bind('<Button-1>', self.on_eng_press)
        self.energy_canvas.bind('<B1-Motion>', self.on_eng_drag)
        self.energy_canvas.bind('<ButtonRelease-1>', self.on_eng_release)
        self.energy_canvas.bind('<MouseWheel>', self.on_eng_wheel)
        self.energy_canvas.bind('<Button-4>', self.on_eng_wheel)
        self.energy_canvas.bind('<Button-5>', self.on_eng_wheel)
        self.energy_canvas.bind('<Double-Button-1>', self.on_eng_reset)
        self.energy_canvas.bind('<Button-3>', self.on_eng_reset)

        self.status = tk.Label(self.root, text='就绪', bg='#1e1e1e',
                               fg='#8bc34a', anchor='w',
                               font=('Microsoft YaHei', 9))
        self.status.pack(side='bottom', fill='x')
        self.hud = tk.Label(self.root, text='', bg='#1e1e1e', fg='#888888',
                            anchor='e', font=('Consolas', 9))
        self.hud.pack(side='bottom', fill='x')

        c = self.canvas
        c.bind('<Button-1>',        self.on_lmb)
        c.bind('<Button-2>',        self.on_mmb)
        c.bind('<Button-3>',        self.on_rmb)
        c.bind('<B1-Motion>',       self.on_motion)
        c.bind('<B2-Motion>',       self.on_pan_motion)
        c.bind('<ButtonRelease-1>', self.on_release)
        c.bind('<ButtonRelease-2>', self.on_pan_release)
        c.bind('<MouseWheel>',      self.on_wheel)
        c.bind('<Button-4>',        self.on_wheel)
        c.bind('<Button-5>',        self.on_wheel)
        c.bind('<Motion>',          self.on_hover)

        self.root.bind('<Delete>', lambda e: self.delete_selected())
        self.root.bind('<Escape>', lambda e: self.clear_selection())
        self.root.bind('<f>',      lambda e: self.fit_view())
        self.root.bind('<Control-s>', lambda e: self.save_to_json())
        self.root.bind('<Control-o>', lambda e: self.load_from_json())

        self._rebuild_toolbar()
        self._refresh_tabs()

    # ---------------- 阶段切换 ----------------
    def set_stage(self, stage):
        if stage == self.stage:
            return
        if self.stage == 'simulate':
            self.sim.pause_playback()
            self.sim.stop()
            self.sim.restore_pre_sim()

        if stage == 'simulate':
            self.sim._ensure_fresh()
            if self.sim.state == 'idle' and self.sim.snapshot_pre_sim is None:
                self.sim.capture_pre_sim()

        self.stage = stage
        self.dragging = None
        self.drag_grab_local = None
        self.create_part_start = None
        self.assemble_sel.clear()
        self.sel = None
        self._snap_target = None
        if stage == 'design':
            self.mode = 'select'
        self._refresh_tabs()
        self._rebuild_toolbar()
        self.update_props()
        self.redraw()
        self.set_status({
            'design':   '① 设计阶段: 编辑零部件与连接附; 松手自动重算包络',
            'assemble': '② 装配阶段: 拖动零件沿约束运动; Ctrl+点击多选; 点"装配"校验并吸附',
            'simulate': '③ 仿真阶段: 开始仿真后逐帧回放; 选中零件显示受力',
        }[stage])

    def _refresh_tabs(self):
        for k, b in self.tab_btns.items():
            if k == self.stage:
                b.configure(bg='#2b2b2b', fg=STAGE_COLORS[k][0])
            else:
                b.configure(bg='#151515', fg='#777777')

    def _rebuild_toolbar(self):
        for w in self.toolbar.winfo_children():
            w.destroy()
        self.sim_widgets = {}

        def mkbtn(parent, text, cmd, w=10, col='#3c3f41', fg='#dddddd'):
            b = tk.Button(parent, text=text, width=w, relief='flat',
                          bg=col, fg=fg, activebackground='#4a90d9',
                          command=cmd)
            b.pack(side='left', padx=2, pady=5)
            return b

        if self.stage == 'design':
            modes = [('选择/平移','select'), ('零部件','part'),
                     ('连接附','point'), ('删除','delete')]
            self.mode_btns = {}
            for text, m in modes:
                b = mkbtn(self.toolbar, text,
                          lambda mm=m: self.set_mode(mm))
                self.mode_btns[m] = b
            self.set_mode(self.mode)

        elif self.stage == 'assemble':
            mkbtn(self.toolbar, '装配连接附', self.do_assemble, w=12,
                  col='#e69500', fg='white')
            mkbtn(self.toolbar, '碰撞装配', self.do_collision_assemble, w=10,
                  col='#455a64', fg='white')
            mkbtn(self.toolbar, '清空多选', self.clear_assemble_sel, w=10)
            mkbtn(self.toolbar, '全选未连接', self.select_all_unconnected, w=12)
            mkbtn(self.toolbar, '适应视图', self.fit_view, w=10)

        elif self.stage == 'simulate':
            self._build_sim_toolbar()

        # 全局按钮
        mkbtn(self.toolbar, '保存', self.save_to_json, w=6, col='#455a64')
        mkbtn(self.toolbar, '加载', self.load_from_json, w=6, col='#455a64')
        mkbtn(self.toolbar, '示例', self.build_demo, w=6)
        mkbtn(self.toolbar, '清空', self.clear_all, w=6)

        tk.Label(self.toolbar,
                 text='滚轮缩放 | 中键/空白左键拖拽平移 | Ctrl+S/O 保存/加载',
                 bg='#2b2b2b', fg='#666666',
                 font=('Microsoft YaHei', 8)).pack(side='right', padx=10)

    def set_mode(self, m):
        """设计阶段工具模式切换 (选择/零部件/连接附/删除)"""
        self.mode = m
        if not hasattr(self, 'mode_btns'):
            return
        for k, b in self.mode_btns.items():
            b.configure(bg='#4a90d9' if k == m else '#3c3f41',
                        fg='white' if k == m else '#dddddd')

    def _build_sim_toolbar(self):
        self.sim_widgets = {}

        def mkbtn(text, cmd, w=10, col='#3c3f41', fg='#dddddd'):
            b = tk.Button(self.toolbar, text=text, width=w, relief='flat',
                          bg=col, fg=fg, activebackground='#4a90d9',
                          command=cmd)
            b.pack(side='left', padx=2, pady=5)
            return b

        btn_start = mkbtn('开始仿真', self.sim_start, w=10,
                          col='#2e7d32', fg='white')
        self.sim_widgets['start'] = btn_start

        btn_stop = mkbtn('停止', self.sim_stop, w=7,
                         col='#c62828', fg='white')
        self.sim_widgets['stop'] = btn_stop

        btn_reset = mkbtn('重置', self.sim_reset, w=7,
                          col='#455a64', fg='white')
        self.sim_widgets['reset'] = btn_reset

        tk.Frame(self.toolbar, width=10, bg='#2b2b2b').pack(side='left')
        mkbtn('|<', lambda: self.step_playback(-9999), w=3)
        mkbtn('<',  lambda: self.step_playback(-1),    w=3)
        btn_play = mkbtn('播放', self.sim_toggle_play, w=8)
        self.sim_widgets['play'] = btn_play
        mkbtn('>',  lambda: self.step_playback(1),     w=3)
        mkbtn('>|', lambda: self.step_playback(9999),  w=3)

        tk.Label(self.toolbar, text='帧:', bg='#2b2b2b', fg='#999999',
                 font=('Microsoft YaHei', 9)).pack(side='left', padx=(8, 2))
        self.sim_widgets['frame_var'] = tk.IntVar(value=0)
        self.frame_slider = tk.Scale(self.toolbar, from_=0, to=0,
                                     orient='horizontal',
                                     variable=self.sim_widgets['frame_var'],
                                     showvalue=False, length=180,
                                     bg='#2b2b2b', fg='#dddddd',
                                     troughcolor='#3c3f41',
                                     highlightthickness=0,
                                     command=self._on_slider)
        self.frame_slider.pack(side='left', padx=2)
        self.sim_widgets['slider'] = self.frame_slider

        self.sim_widgets['frame_label'] = tk.Label(
            self.toolbar, text='0 / 0', bg='#2b2b2b', fg='#cccccc',
            font=('Consolas', 9), width=12)
        self.sim_widgets['frame_label'].pack(side='left', padx=4)

        # 删除了受力切换开关，默认只用详细受力显示
        
        # 增加弹簧限位环的常驻显示/隐藏按钮
        btn_spring_lim = mkbtn('弹簧限位: 关', self.toggle_spring_limits, w=12,
                               col='#455a64', fg='white')
        self.sim_widgets['spring_limits'] = btn_spring_lim

        mkbtn('适应视图', self.fit_view, w=10)
        self.refresh_sim_ui()

    def refresh_sim_ui(self):
        if self.stage != 'simulate': return
        w = self.sim_widgets
        if not w: return
        state = self.sim.state
        n = len(self.sim.frames)

        if state == 'computing':
            w['start'].configure(text='计算中...', bg='#8a6d00',
                                 state='disabled')
            w['stop'].configure(state='normal')
        else:
            w['start'].configure(
                text=('重新仿真' if state in ('stable', 'error') else '开始仿真'),
                bg='#2e7d32', state='normal')
            w['stop'].configure(state='disabled')

        self._update_play_button()

        self._suppress_slider = True
        try:
            w['slider'].configure(to=max(0, n - 1))
            w['frame_var'].set(self.sim.playhead)
        except tk.TclError:
            pass
        finally:
            self._suppress_slider = False
        w['frame_label'].configure(
            text=f'{self.sim.playhead} / {max(0, n - 1)}')

    def _update_play_button(self):
        w = self.sim_widgets
        if not w or 'play' not in w: return
        if self.sim.playing:
            w['play'].configure(text='暂停', bg='#4a90d9', fg='white')
        else:
            w['play'].configure(text='播放', bg='#3c3f41', fg='#dddddd')

    def _update_playhead_ui(self):
        w = self.sim_widgets
        if not w: return
        self._suppress_slider = True
        try:
            w['frame_var'].set(self.sim.playhead)
        except tk.TclError:
            return
        finally:
            self._suppress_slider = False
        w['frame_label'].configure(
            text=f'{self.sim.playhead} / {max(0, len(self.sim.frames) - 1)}')

    def _on_slider(self, val):
        if self.stage != 'simulate': return
        if getattr(self, '_suppress_slider', False): return
        try:
            idx = int(float(val))
        except ValueError:
            return
        if not self.sim.frames: return
        if self.sim.playing:
            self.sim.pause_playback()
            self._update_play_button()
        self.sim.apply_frame(idx)
        
        # 实时刷新拖动条右侧的工具栏帧数提示文本
        if 'frame_label' in self.sim_widgets:
            total = max(0, len(self.sim.frames) - 1)
            self.sim_widgets['frame_label'].configure(text=f'{idx} / {total}')
            
        self._refresh_props_values()    # 只刷新数值, 不重建面板
        self.redraw()

    def sim_start(self):
        if self.stage != 'simulate': return
        
        # 只在点击开始仿真时记录一次日志文件
        global _LOG_FILE_HANDLER
        if _LOG_FILE_HANDLER is None:
            setup_log_for_model(self.model_name)
            logger.info(f'# 仿真日志开始: {LOG_PATH}')
            
        self.sim.start()
        self.refresh_sim_ui()
        self.redraw()

    def sim_stop(self):
        self.sim.stop()
        self.sim.pause_playback()
        self.refresh_sim_ui()
        self.set_status('已停止仿真 (保留缓存, 可回放)')

    def sim_reset(self):
        self.sim.pause_playback()
        self.sim.restore_pre_sim()
        self.sim.frames.clear()
        self.sim.force_cache.clear()
        self.sim.energy_cache.clear()
        self.sim.initial_ep_g = 0.0
        self.sim.state = 'idle'
        self.sim.playhead = 0
        self.sim.stable_count = 0
        self.sim.step_counter = 0
        self.refresh_sim_ui()
        self.redraw()
        self.set_status('已重置到仿真前状态')

    def sim_toggle_play(self):
        if not self.sim.frames: return
        if self.sim.playing:
            self.sim.pause_playback()
            self._update_play_button()
        else:
            self.sim.play()
            self._update_play_button()
            if not self._sim_playback_timer:
                self._sim_playback_timer = True
                self._playback_loop()

    def _playback_loop(self):
        if not (self.sim.playing and self.stage == 'simulate'):
            self._sim_playback_timer = False
            self._update_play_button()
            return
        if self.sim.playhead < len(self.sim.frames) - 1:
            self.sim.apply_frame(self.sim.playhead + 1)
            self._update_playhead_ui()
            # 播放时若选中零件且显示受力, 只刷新面板数值 (避免闪烁)
            if (self.analysis_detail != 'off'
                    and self.sel and self.sel[0] == 'part'):
                self._refresh_props_values()
            self.redraw()
            self.root.after(16, self._playback_loop)
        else:
            self.sim.playing = False
            self._sim_playback_timer = False
            self._update_play_button()

    def step_playback(self, delta):
        if not self.sim.frames: return
        self.sim.state = 'playback'
        self.sim.playing = False
        self.sim.apply_frame(self.sim.playhead + delta)
        self.refresh_sim_ui()
        self._refresh_props_values()
        self.redraw()

    def cycle_analysis_detail(self):
        """受力分析按钮: 关 -> 简略 -> 详细 -> 关"""
        if self.analysis_detail == 'off':
            self.analysis_detail = 'simple'
        elif self.analysis_detail == 'simple':
            self.analysis_detail = 'detailed'
        else:
            self.analysis_detail = 'off'
        if 'analysis' in self.sim_widgets:
            self.sim_widgets['analysis'].configure(
                text=ANALYSIS_LABELS[self.analysis_detail])
        self.update_props()
        self.redraw()

    def toggle_spring_limits(self):
        """仿真阶段：开启/关闭弹簧限位可视化圆环常驻显示"""
        self.show_spring_limits = not getattr(self, 'show_spring_limits', False)
        txt = '弹簧限位: 开' if self.show_spring_limits else '弹簧限位: 关'
        if 'spring_limits' in self.sim_widgets:
            self.sim_widgets['spring_limits'].configure(text=txt)
        self.redraw()

    # ---------------- 清空 / 示例 ----------------
    def clear_all(self):
        self.parts.clear(); self.links.clear(); self.collisions.clear()
        ConnPoint._pid = 0; Part._ptid = 0; Link._lid = 0
        self.sel = None
        self.assemble_sel.clear()
        self._snap_target = None
        self.dragging = None
        self.drag_grab_local = None
        self._touch()
        self.update_props()
        self.redraw()

    def build_demo(self):
        self.parts.clear(); self.links.clear(); self.collisions.clear()
        ConnPoint._pid = 0; Part._ptid = 0; Link._lid = 0
        self.sel = None
        self.assemble_sel.clear()
        self._snap_target = None
        self.dragging = None
        self.drag_grab_local = None

        g = Part((450, 645), None, '底座')
        g.locked = True; g.body_mass = 10.0
        self.parts.append(g)
        A  = ConnPoint(g, (320-450, 640-645)); A.type = 'revolute'
        B  = ConnPoint(g, (560-450, 640-645)); B.type = 'revolute'
        SA = ConnPoint(g, (200-450, 620-645)); SA.type = 'spring'
        g.points.extend([A, B, SA])
        self._rebuild_envelope(g)

        post = Part((320, 540), None, '立柱')
        post.body_mass = 0.15
        self.parts.append(post)
        A2 = ConnPoint(post, (0, 100)); A2.type = 'revolute'
        C  = ConnPoint(post, (0, -100)); C.type = 'revolute'
        SC = ConnPoint(post, (-60, -60)); SC.type = 'spring'
        post.points.extend([A2, C, SC])
        self._rebuild_envelope(post)

        arm = Part((460, 440), None, '托盘')
        arm.body_mass = 0.20
        self.parts.append(arm)
        C2 = ConnPoint(arm, (320-460, 0));  C2.type = 'revolute'
        D  = ConnPoint(arm, (560-460, 0));  D.type  = 'revolute'
        E  = ConnPoint(arm, (600-460, 0));  E.type  = 'payload'; E.mass = 0.30
        arm.points.extend([C2, D, E])
        self._rebuild_envelope(arm)

        back = Part((560, 540), None, '后支撑')
        back.body_mass = 0.15
        self.parts.append(back)
        D2 = ConnPoint(back, (0, -100)); D2.type = 'revolute'
        B2 = ConnPoint(back, (0, 100));  B2.type = 'revolute'
        back.points.extend([D2, B2])
        self._rebuild_envelope(back)

        self._create_link(A,  A2, 'revolute')
        self._create_link(C,  C2, 'revolute')
        self._create_link(D,  D2, 'revolute')
        self._create_link(B,  B2, 'revolute')
        self._create_link(SA, SC, 'spring',
                          natural_length=float(np.linalg.norm(SA.world() - SC.world())))

        self._touch()
        self.fit_view()
        self.set_status('已载入示例四杆支架 + 张拉弹簧 (托盘端 0.30 kg)')

    # ---------------- 工厂 ----------------
    def add_part(self, pos, hull_local=None, name=None):
        p = Part(pos, hull_local, name)
        self.parts.append(p)
        return p

    def add_point(self, part, world_pos, type_='revolute'):
        local = part.w2l(world_pos)
        cp = ConnPoint(part, local, type_)
        part.points.append(cp)
        return cp

    def _create_link(self, pA, pB, type_, natural_length=None):
        if pA is pB or pA.link is not None or pB.link is not None:
            return None
        if type_ not in ('revolute', 'spring', 'magnet'):
            return None
        if type_ == 'revolute':
            mid = (pA.world() + pB.world()) * 0.5
            pA.local = pA.part.w2l(mid)
            pB.local = pB.part.w2l(mid)
        lk = Link(pA, pB, type_)
        if type_ == 'revolute':
            lk.static_damping = (getattr(pA, 'static_damping', 0.0)
                                 + getattr(pB, 'static_damping', 0.0)) * 0.5
        elif type_ == 'spring':
            L = natural_length if natural_length else \
                float(np.linalg.norm(pA.world() - pB.world()))
            lk.l0    = max(L, 30.0)
            lk.lmin  = max(L * 0.4, 10.0)
            lk.lmax  = max(L * 1.6, 60.0)
            lk.k     = 100.0
            lk.c     = 5.0
        elif type_ == 'magnet':
            L = natural_length if natural_length else \
                float(np.linalg.norm(pA.world() - pB.world()))
            lk.l0       = max(L, 20.0)
            lk.strength = 3000.0
            lk.mk       = 500.0
        self.links.append(lk)
        return lk

    def _rebuild_envelope(self, part):
        if not part.points: return
        part.hull = compute_part_envelope(part, margin=10.0)
        part.recenter()

    # ---------------- 拾取 ----------------
    def pick(self, sx, sy):
        w = self.s2w(sx, sy)
        r_pt = 12.0 / self.zoom
        for part in reversed(self.parts):
            for pt in part.points:
                if np.linalg.norm(pt.world() - w) < r_pt:
                    return ('point', pt)
        r_lk = 10.0 / self.zoom
        for lk in reversed(self.links):
            if lk.type in ('spring', 'magnet'):
                a = lk.pA.world(); b = lk.pB.world()
                d = point_seg_dist(w[0], w[1], a, b)
                if d < r_lk:
                    return ('link', lk)
        for part in reversed(self.parts):
            if self._point_in_part(part, w):
                return ('part', part)
        return (None, None)

    @staticmethod
    def _point_in_part(part, world):
        local = part.w2l(world)
        n = len(part.hull)
        if n < 3: return False
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = part.hull[i]
            xj, yj = part.hull[j]
            if (yi > local[1]) != (yj > local[1]):
                xint = (xj - xi) * (local[1] - yi) / (yj - yi + 1e-12) + xi
                if local[0] < xint:
                    inside = not inside
            j = i
        return inside

    def find_snap_target(self, world_pos, type_, exclude=None,
                         exclude_same_part=False):
        if type_ != 'revolute': return None
        r = SNAP_PX / self.zoom
        excl = set()
        if exclude is not None:
            excl.add(id(exclude))
            if exclude.link is not None:
                excl.add(id(exclude.link))
        best, bd = None, r
        for p in self.parts:
            if exclude_same_part and exclude is not None and p is exclude.part:
                continue
            for cp in p.points:
                if id(cp) in excl: continue
                if cp.link is not None: continue
                if cp.type != type_: continue
                d = float(np.linalg.norm(cp.world() - world_pos))
                if d < bd:
                    bd = d; best = cp
        return best

    # ---------------- 约束求解 ----------------
    def _project_revolute(self, lk):
        pa, pb = lk.pA.part, lk.pB.part
        if pa.locked and pb.locked: return
        wA = lk.pA.world(); wB = lk.pB.world()
        e = wA - wB
        if float(np.linalg.norm(e)) < 1e-10: return
        invMA = 0.0 if pa.locked else 1.0 / pa.all_mass()
        invMB = 0.0 if pb.locked else 1.0 / pb.all_mass()
        invIA = 0.0 if pa.locked else 1.0 / pa.inertia()
        invIB = 0.0 if pb.locked else 1.0 / pb.inertia()
        rA = rot(pa.angle) @ lk.pA.local
        rB = rot(pb.angle) @ lk.pB.local
        pA_ = perp(rA); pB_ = perp(rB)
        K = (invMA + invMB) * np.eye(2)
        K += invIA * np.outer(pA_, pA_)
        K += invIB * np.outer(pB_, pB_)
        K = K + REG_EPS * np.eye(2)
        try:
            lam = np.linalg.solve(K, e)
        except np.linalg.LinAlgError:
            return
        if not np.all(np.isfinite(lam)): return
        if not pa.locked:
            pa.pos   = pa.pos   - invMA * lam
            pa.angle = pa.angle - invIA * (pA_ @ lam)
        if not pb.locked:
            pb.pos   = pb.pos   + invMB * lam
            pb.angle = pb.angle + invIB * (pB_ @ lam)

    def _project_spring_limit_on_pos(self, lk):
        """装配阶段拖拽专用：弹簧硬限位位置投影"""
        pa, pb = lk.pA, lk.pB
        pA, pB = pa.part, pb.part
        wa = pa.world(); wb = pb.world()
        d = wb - wa
        L = float(np.linalg.norm(d))
        if L < 1e-9: return 0.0
        u = d / L
        if L <= lk.lmin:
            C = L - lk.lmin
        elif L >= lk.lmax:
            C = L - lk.lmax
        else:
            return 0.0
        
        invMA = 0.0 if pA.locked else 1.0 / pA.all_mass()
        invMB = 0.0 if pB.locked else 1.0 / pB.all_mass()
        invIA = 0.0 if pA.locked else 1.0 / pA.inertia()
        invIB = 0.0 if pB.locked else 1.0 / pB.inertia()
        
        rA = rot(pA.angle) @ pa.local
        rB = rot(pB.angle) @ pb.local
        pA_ = perp(rA); pB_ = perp(rB)
        
        k_eff = (invMA + invMB) + invIA * float(pA_ @ u) ** 2 + invIB * float(pB_ @ u) ** 2
        if k_eff < 1e-12: return 0.0
        lam = C / k_eff
        
        if not pA.locked:
            pA.pos   = pA.pos   + invMA * lam * u
            pA.angle = pA.angle + invIA * float(pA_ @ u) * lam
        if not pB.locked:
            pB.pos   = pB.pos   - invMB * lam * u
            pB.angle = pB.angle - invIB * float(pB_ @ u) * lam
        return abs(C)

    def _project_collision_on_pos(self, pA, pB):
        """装配阶段拖拽专用：碰撞硬限位位置投影"""
        if pA.locked and pB.locked: return 0.0
        if len(pA.hull) < 3 or len(pB.hull) < 3: return 0.0
        polyA = [pA.pos + rot(pA.angle) @ v for v in pA.hull]
        polyB = [pB.pos + rot(pB.angle) @ v for v in pB.hull]
        overlap, normal, depth = convex_polygons_sat(polyA, polyB)
        if not overlap or depth < 1e-6: return 0.0
        invMA = 0.0 if pA.locked else 1.0 / pA.all_mass()
        invMB = 0.0 if pB.locked else 1.0 / pB.all_mass()
        tot = invMA + invMB
        if tot < 1e-12: return 0.0
        if not pA.locked:
            pA.pos = pA.pos - normal * depth * (invMA / tot)
        if not pB.locked:
            pB.pos = pB.pos + normal * depth * (invMB / tot)
        return depth

    def _correct_all_constraints(self):
        """拖拽中求解所有约束 (调整为铰链最后执行，确保硬约束不被破坏)"""
        for _ in range(CORRECTION_ITER * 2):  # 略微增加迭代次数保障收敛
            max_err = 0.0
            # 1. 先求解弹簧双边界限位
            for lk in self.links:
                if lk.type == 'spring':
                    err = self._project_spring_limit_on_pos(lk)
                    if err > max_err: max_err = err
            # 2. 求解碰撞分离
            for (pA, pB) in self.collisions:
                err = self._project_collision_on_pos(pA, pB)
                if err > max_err: max_err = err
            # 3. 最后求解转动副铰链 (具有最高优先级，确保强制贴合不拉断)
            for lk in self.links:
                if lk.type == 'revolute':
                    if lk.pA.part.locked and lk.pB.part.locked: continue
                    err = float(np.linalg.norm(lk.pA.world() - lk.pB.world()))
                    if err > max_err: max_err = err
                    self._project_revolute(lk)
            if max_err < 1e-4: break

    def _build_jacobian(self, free_parts):
        var_map = {}
        for i, pp in enumerate(free_parts):
            var_map[(id(pp), 0)] = i * 3
            var_map[(id(pp), 1)] = i * 3 + 1
            var_map[(id(pp), 2)] = i * 3 + 2
        n_vars = len(free_parts) * 3
        active = [lk for lk in self.links
                  if lk.type == 'revolute'
                  and not (lk.pA.part.locked and lk.pB.part.locked)]
        if not active: return None, var_map, 0
        J = np.zeros((len(active) * 2, n_vars))
        for k, lk in enumerate(active):
            pa, pb = lk.pA.part, lk.pB.part
            la = lk.pA.local; lb = lk.pB.local
            ca, sa = math.cos(pa.angle), math.sin(pa.angle)
            cb, sb = math.cos(pb.angle), math.sin(pb.angle)
            if not pa.locked:
                cx = var_map[(id(pa), 0)]; cy = var_map[(id(pa), 1)]
                ct = var_map[(id(pa), 2)]
                J[2*k,   cx] = 1.0
                J[2*k+1, cy] = 1.0
                J[2*k,   ct] = -la[0]*sa - la[1]*ca
                J[2*k+1, ct] =  la[0]*ca - la[1]*sa
            if not pb.locked:
                cx = var_map[(id(pb), 0)]; cy = var_map[(id(pb), 1)]
                ct = var_map[(id(pb), 2)]
                J[2*k,   cx] = -1.0
                J[2*k+1, cy] = -1.0
                J[2*k,   ct] =  lb[0]*sb + lb[1]*cb
                J[2*k+1, ct] = -lb[0]*cb + lb[1]*sb
        return J, var_map, len(active)

    def _nullspace_project(self, dragged_part, desired):
        free_parts = [pp for pp in self.parts if not pp.locked]
        if not free_parts or dragged_part.locked:
            return desired.copy(), 0.0, 0
        J, var_map, n_active = self._build_jacobian(free_parts)
        if J is None or n_active == 0:
            return desired.copy(), 0.0, 0
        n_vars = J.shape[1]
        d = np.zeros(n_vars)
        if (id(dragged_part), 0) in var_map:
            d[var_map[(id(dragged_part), 0)]] = desired[0]
            d[var_map[(id(dragged_part), 1)]] = desired[1]
        try:
            J_pinv = np.linalg.pinv(J, rcond=1e-8)
            d_proj = d - J_pinv @ (J @ d)
        except np.linalg.LinAlgError:
            d_proj = d
        out = np.zeros(3)
        if (id(dragged_part), 0) in var_map:
            out[0] = d_proj[var_map[(id(dragged_part), 0)]]
            out[1] = d_proj[var_map[(id(dragged_part), 1)]]
            out[2] = d_proj[var_map[(id(dragged_part), 2)]]
        return out[:2], out[2], n_active

    def _solve_assemble_drag(self):
        if not (self.dragging and self.dragging[0] == 'part'): return
        p = self.dragging[1]
        if p.locked or self.drag_grab_local is None: return
        for pp in self.parts:
            if not (np.isfinite(pp.pos).all() and np.isfinite(pp.angle)):
                return
        grab_local = self.drag_grab_local
        target     = self.mouse_world
        for _ in range(DRAG_ITER):
            grab_world = p.l2w(grab_local)
            desired = target - grab_world
            d_len = float(np.linalg.norm(desired))
            if d_len < DRAG_TOL: break
            if d_len > DRAG_MAX_STEP:
                desired *= DRAG_MAX_STEP / d_len
            dpos, dtheta, n_active = self._nullspace_project(p, desired)
            if n_active == 0:
                p.pos = p.pos + desired
            else:
                p.pos   = p.pos   + dpos
                p.angle = p.angle + dtheta
            self._correct_all_constraints()
            if not (np.isfinite(p.pos).all() and np.isfinite(p.angle)):
                return
        self._correct_all_constraints()

    def _try_converge_constraints(self):
        for _ in range(FEASIBILITY_ITER):
            max_err = 0.0
            for lk in self.links:
                if lk.type != 'revolute': continue
                if lk.pA.part.locked and lk.pB.part.locked: continue
                err = float(np.linalg.norm(lk.pA.world() - lk.pB.world()))
                if err > max_err: max_err = err
                self._project_revolute(lk)
            if max_err < FEASIBILITY_TOL:
                return True, max_err
            for pp in self.parts:
                if pp.locked: continue
                if not (np.isfinite(pp.pos).all() and np.isfinite(pp.angle)):
                    return False, float('inf')
        return False, max_err

    def _validate_assembly_static(self, cpA, cpB):
        if cpA is cpB: return False, '不能装配同一个连接附'
        if cpA.part is cpB.part: return False, '两个连接附属于同一个零件'
        if cpA.type != cpB.type:
            return False, f'类型不一致: {cpA.type} vs {cpB.type}'
        if cpA.link is not None or cpB.link is not None:
            return False, '至少一个连接附已被占用'
        t = cpA.type
        if t not in ('revolute', 'spring', 'magnet'):
            return False, f'类型 {t} 不支持装配'
        if t == 'revolute' and cpA.part.locked and cpB.part.locked:
            return False, '两个零件都已锁定'
        return True, ''

    # ---------------- 鼠标事件 ----------------
    def on_hover(self, e):
        self.mouse_screen = (e.x, e.y)
        self.mouse_world  = self.s2w(e.x, e.y)

    def on_lmb(self, e):
        self.mouse_screen = (e.x, e.y)
        self.mouse_world  = self.s2w(e.x, e.y)
        ctrl = bool(e.state & 0x0004)
        if self.stage == 'design':     self._design_lmb(e, ctrl)
        elif self.stage == 'assemble': self._assemble_lmb(e, ctrl)
        elif self.stage == 'simulate': self._simulate_lmb(e, ctrl)
        self.redraw()

    def _design_lmb(self, e, ctrl):
        kind, obj = self.pick(e.x, e.y)
        if self.mode == 'select':
            if kind == 'point':
                self.sel = ('point', obj)
                self.dragging = ('point', obj)
                self.drag_mouse0 = np.array([e.x, e.y])
                self.drag_obj0 = obj.world().copy()
            elif kind == 'part':
                self.sel = ('part', obj)
                self.dragging = ('part', obj)
                self.drag_mouse0 = np.array([e.x, e.y])
                self.drag_obj0 = obj.pos.copy()
            else:
                self.sel = None
                self.panning = True
                self.pan_start = (e.x, e.y)
                self.pan_view_start = (self.view_x, self.view_y)
            self.update_props()
        elif self.mode == 'part':
            if kind is None: self.create_part_start = (e.x, e.y)
        elif self.mode == 'point':
            if kind == 'part':
                cp = self.add_point(obj, self.mouse_world)
                self._rebuild_envelope(obj)
                self._touch()
                self.sel = ('point', cp)
                self.update_props()
        elif self.mode == 'delete':
            self.delete_object(kind, obj)

    def _assemble_lmb(self, e, ctrl):
        # 在拾取常规零件前，先检查是否点中了当前选中弹簧的调节圆环
        if self.sel and self.sel[0] == 'link' and self.sel[1].type == 'spring':
            lk = self.sel[1]
            center_w = lk.pA.world()
            dist = float(np.linalg.norm(self.mouse_world - center_w))
            tol = 8.0 / self.zoom  # 屏幕上 8 像素的命中容差
            
            if abs(dist - lk.lmin) < tol:
                self.dragging = ('spring_limit', lk, 'lmin')
                return
            elif abs(dist - lk.l0) < tol:
                self.dragging = ('spring_limit', lk, 'l0')
                return
            elif abs(dist - lk.lmax) < tol:
                self.dragging = ('spring_limit', lk, 'lmax')
                return
                
        kind, obj = self.pick(e.x, e.y)
        if kind == 'point':
            if ctrl:
                if obj in self.assemble_sel: self.assemble_sel.remove(obj)
                else: self.assemble_sel.append(obj)
                self.update_props()
            else:
                # 已连接的连接附 -> 进入 Link 属性面板
                # (转动副/弹簧/磁吸都可以在此编辑其 Link 参数)
                lk = self._find_link_of(obj) if obj.link is not None else None
                if lk is not None:
                    self.sel = ('link', lk)
                else:
                    self.sel = ('point', obj)
                self.assemble_sel.clear()
                self.update_props()
        elif kind == 'link':
            self.sel = ('link', obj)
            self.assemble_sel.clear()
            self.update_props()
        elif kind == 'part':
            if ctrl:
                if obj in self.assemble_sel:
                    self.assemble_sel.remove(obj)
                else:
                    self.assemble_sel.append(obj)
                self.sel = ('part', obj)
                self.update_props()
            else:
                self.sel = ('part', obj)
                if not obj.locked:
                    self.dragging = ('part', obj)
                    self.drag_grab_local = obj.w2l(self.mouse_world)
                self.assemble_sel.clear()
                self.update_props()
        else:
            if not ctrl:
                self.assemble_sel.clear()
                self.sel = None
            self.panning = True
            self.pan_start = (e.x, e.y)
            self.pan_view_start = (self.view_x, self.view_y)
            self.update_props()

    def _simulate_lmb(self, e, ctrl):
        kind, obj = self.pick(e.x, e.y)
        if kind in ('link', 'part', 'point'):
            self.sel = (kind, obj)
            self.update_props()
            return
        self.sel = None
        self.panning = True
        self.pan_start = (e.x, e.y)
        self.pan_view_start = (self.view_x, self.view_y)
        self.update_props()

    def on_motion(self, e):
        self.mouse_screen = (e.x, e.y)
        self.mouse_world  = self.s2w(e.x, e.y)
        if self.panning:
            dx = e.x - self.pan_start[0]
            dy = e.y - self.pan_start[1]
            self.view_x = self.pan_view_start[0] - dx / self.zoom
            self.view_y = self.pan_view_start[1] - dy / self.zoom
            self.redraw(); return
        if self.create_part_start is not None:
            self.redraw(); return
            
        # 处理弹簧圆环拖拽
        if self.stage == 'assemble' and self.dragging and self.dragging[0] == 'spring_limit':
            lk = self.dragging[1]
            prop = self.dragging[2]
            center_w = lk.pA.world()
            new_dist = float(np.linalg.norm(self.mouse_world - center_w))
            if new_dist < 1.0: new_dist = 1.0
            
            setattr(lk, prop, new_dist)
            
            # 合法性强制联动: lmin <= l0 <= lmax
            if prop == 'lmin':
                if lk.l0 < lk.lmin: lk.l0 = lk.lmin
                if lk.lmax < lk.lmin: lk.lmax = lk.lmin
            elif prop == 'l0':
                if lk.lmin > lk.l0: lk.lmin = lk.l0
                if lk.lmax < lk.l0: lk.lmax = lk.l0
            elif prop == 'lmax':
                if lk.l0 > lk.lmax: lk.l0 = lk.lmax
                if lk.lmin > lk.lmax: lk.lmin = lk.lmax
                
            self._touch()
            self.update_props() # 实时更新侧边栏数值
            self.redraw()
            return
            
        if self.stage == 'design' and self.dragging and self.drag_mouse0 is not None:
            cur_w   = self.mouse_world
            start_w = self.s2w(self.drag_mouse0[0], self.drag_mouse0[1])
            delta   = cur_w - start_w
            k = self.dragging[0]
            if k == 'part':
                p = self.dragging[1]
                p.pos = self.drag_obj0 + delta
                p.pred_pos = p.pos.copy()
            elif k == 'point':
                cp = self.dragging[1]
                raw = self.drag_obj0 + delta
                snap = None
                if cp.link is None and cp.type == 'revolute':
                    snap = self.find_snap_target(raw, cp.type,
                                                 exclude=cp,
                                                 exclude_same_part=True)
                self._snap_target = snap
                if snap is not None and cp.type == 'revolute':
                    cp.local = cp.part.w2l(snap.world())
                else:
                    cp.local = cp.part.w2l(raw)
            # 拖动后: 已建立的转动副若被拉开则自动断开
            self._auto_break_revolutes_if_separated()
            self.redraw(); return
        if (self.stage == 'assemble' and self.dragging
                and self.dragging[0] == 'part'):
            self._solve_assemble_drag()
            self.redraw()
            return

    def on_release(self, e):
        self.mouse_screen = (e.x, e.y)
        self.mouse_world  = self.s2w(e.x, e.y)
        if self.panning: self.panning = False
        if self.create_part_start is not None:
            x0, y0 = self.create_part_start
            x1, y1 = e.x, e.y
            if abs(x1 - x0) > 15 and abs(y1 - y0) > 15:
                wa = self.s2w(x0, y0); wb = self.s2w(x1, y1)
                cx, cy = (wa[0] + wb[0]) / 2, (wa[1] + wb[1]) / 2
                hw, hh = abs(wb[0] - wa[0]) / 2, abs(wb[1] - wa[1]) / 2
                hull = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
                p = self.add_part((cx, cy), hull)
                self._touch()
                self.sel = ('part', p)
                self.update_props()
            self.create_part_start = None
            self.redraw(); return
        if (self.stage == 'design' and self.dragging
                and self.dragging[0] == 'point'):
            cp = self.dragging[1]
            self._rebuild_envelope(cp.part)
            self._touch()
            self.update_props()
            self._snap_target = None
            self._auto_break_revolutes_if_separated()
        if (self.stage == 'design' and self.dragging
                and self.dragging[0] == 'part'):
            # 拖动整个零件后, 检查该零件上的转动副是否已被拉开
            self._auto_break_revolutes_if_separated()
        if (self.stage == 'assemble' and self.dragging
                and self.dragging[0] == 'part'):
            self._correct_all_constraints()
            self._touch()
        if self.dragging and self.dragging[0] == 'spring_limit':
            self._touch()
        self.dragging = None
        self.drag_mouse0 = None
        self.drag_obj0 = None
        self.drag_grab_local = None
        self.redraw()

    def on_mmb(self, e):
        self.panning = True
        self.pan_start = (e.x, e.y)
        self.pan_view_start = (self.view_x, self.view_y)

    def on_pan_motion(self, e):
        if not self.panning: return
        dx = e.x - self.pan_start[0]
        dy = e.y - self.pan_start[1]
        self.view_x = self.pan_view_start[0] - dx / self.zoom
        self.view_y = self.pan_view_start[1] - dy / self.zoom
        self.redraw()

    def on_pan_release(self, e):
        self.panning = False

    def on_rmb(self, e):
        if self.create_part_start is not None:
            self.create_part_start = None
            self.redraw(); return
        if self.stage == 'design':
            kind, obj = self.pick(e.x, e.y)
            self.delete_object(kind, obj)

    def on_wheel(self, e):
        if e.num == 4:   delta = 1
        elif e.num == 5: delta = -1
        else:            delta = 1 if (e.delta or 0) > 0 else -1
        factor = 1.1 if delta > 0 else 1/1.1
        wx, wy = self.s2w(e.x, e.y)
        self.zoom = max(0.1, min(10.0, self.zoom * factor))
        self.view_x = wx - e.x / self.zoom
        self.view_y = wy - e.y / self.zoom
        self.redraw()

    # ---------------- 删除 / 选择 ----------------
    def delete_selected(self):
        if self.sel is None: return
        kind, obj = self.sel
        if self.stage == 'design':
            self.delete_object(kind, obj)
        elif self.stage == 'assemble' and kind == 'link':
            self._unlink_by_link(obj)
            self._touch()
        self.sel = None

    def clear_selection(self):
        self.sel = None
        self.assemble_sel.clear()
        self.update_props()
        self.redraw()

    def delete_object(self, kind, obj):
        if obj is None: return
        if self.stage != 'design' and kind == 'point': return
        if kind == 'point':
            for lk in list(self.links):
                if lk.pA is obj or lk.pB is obj:
                    if lk.pA.link is lk.pB: lk.pA.link = None
                    if lk.pB.link is lk.pA: lk.pB.link = None
                    self.links.remove(lk)
            if obj in obj.part.points:
                obj.part.points.remove(obj)
            self._rebuild_envelope(obj.part)
        elif kind == 'part':
            for lk in list(self.links):
                if lk.pA.part is obj or lk.pB.part is obj:
                    if lk.pA.link is lk.pB: lk.pA.link = None
                    if lk.pB.link is lk.pA: lk.pB.link = None
                    self.links.remove(lk)
            if obj in self.parts:
                self.parts.remove(obj)
        elif kind == 'link':
            self._unlink_by_link(obj)
        if self.sel and self.sel[1] is obj:
            self.sel = None
        self._touch()
        self.update_props()
        self.redraw()

    def _unlink_by_link(self, lk):
        if lk.pA.link is lk.pB: lk.pA.link = None
        if lk.pB.link is lk.pA: lk.pB.link = None
        if lk in self.links:
            self.links.remove(lk)

    def _find_link_of(self, cp):
        """根据连接附找到它所属的 Link 对象。

        注意: ConnPoint.link 存的是"配对的另一端 ConnPoint",
        而不是 Link 对象本身。要取得 Link 必须遍历 self.links。
        """
        if cp is None: return None
        for lk in self.links:
            if lk.pA is cp or lk.pB is cp:
                return lk
        return None

    def _auto_break_revolutes_if_separated(self, tol=2.0):
        """设计阶段拖动后调用: 若已建立的转动副两端连接附世界坐标
        不再重合, 则自动断开这对连接。
        弹簧/磁吸等远程连接天然处于分离状态, 不检查。
        tol: 世界坐标 (px) 上的分离容差, 默认 2px (2mm)
        """
        broken = []
        for lk in list(self.links):
            if lk.type != 'revolute':
                continue
            try:
                wa = lk.pA.world()
                wb = lk.pB.world()
                d = float(np.linalg.norm(wa - wb))
            except Exception:
                continue
            if d > tol:
                broken.append((lk.id, d,
                               lk.pA.part.id, lk.pB.part.id))
                self._unlink_by_link(lk)
        if broken:
            parts_str = ', '.join(
                f'#{i}(Δ={d:.1f}px, part{a}↔part{b})'
                for i, d, a, b in broken)
            msg = f'自动断开 {len(broken)} 处转动副: {parts_str}'
            self.set_status(msg)
            logger.info(f'[autobreak] {msg}')
            self._touch()
        return broken

    # ---------------- 装配动作 ----------------
    def do_assemble(self):
        if self.stage != 'assemble': return
        cps = [x for x in self.assemble_sel if isinstance(x, ConnPoint)]
        if len(cps) != 2:
            messagebox.showinfo('提示', '请按住 Ctrl 选中恰好 2 个连接附')
            return
        cpA, cpB = cps[0], cps[1]
        ok, msg = self._validate_assembly_static(cpA, cpB)
        if not ok:
            messagebox.showinfo('无法装配', msg)
            self.set_status(f'装配失败: {msg}')
            return
        snap = {id(pp): (pp.pos.copy(), pp.angle,
                         pp.pred_pos.copy(), pp.pred_angle)
                for pp in self.parts}
        loc_snap = {id(cpA): cpA.local.copy(), id(cpB): cpB.local.copy()}
        n_links_before = len(self.links)
        type_ = cpA.type
        pre_dist = float(np.linalg.norm(cpA.world() - cpB.world()))
        partA, partB = cpA.part, cpB.part
        if type_ == 'revolute':
            if partB.locked:
                delta = cpB.world() - cpA.world()
                partA.pos = partA.pos + delta
            else:
                delta = cpA.world() - cpB.world()
                partB.pos = partB.pos + delta
        nat = pre_dist if type_ in ('spring', 'magnet') else None
        lk = self._create_link(cpA, cpB, type_, natural_length=nat)
        if lk is None:
            messagebox.showinfo('无法装配', '创建连接失败')
            return
        if type_ == 'revolute':
            converged, err = self._try_converge_constraints()
            if not converged:
                for pp in self.parts:
                    if id(pp) in snap:
                        pp.pos, pp.angle = snap[id(pp)][0], snap[id(pp)][1]
                        pp.pred_pos, pp.pred_angle = snap[id(pp)][2], snap[id(pp)][3]
                cpA.local = loc_snap[id(cpA)]
                cpB.local = loc_snap[id(cpB)]
                while len(self.links) > n_links_before:
                    lkd = self.links.pop()
                    lkd.pA.link = None; lkd.pB.link = None
                messagebox.showinfo(
                    '无法装配',
                    f'与现有装配约束冲突, 无法求解 (残差 {err:.2f}px)')
                self.set_status(f'装配失败: 约束冲突')
                return
        self._rebuild_envelope(partA)
        self._rebuild_envelope(partB)
        for pp in self.parts:
            pp.pred_pos   = pp.pos.copy()
            pp.pred_angle = pp.angle
        self._touch()
        self.assemble_sel.clear()
        if type_ == 'spring':
            self.sel = ('link', lk)
            self.set_status(f'装配成功: 弹簧 #{lk.id}')
        else:
            self.set_status(f'装配成功: {type_} 连接 #{lk.id}')
        self.update_props()
        self.redraw()

    def do_collision_assemble(self):
        """碰撞装配: Ctrl 选中两个零件后创建碰撞约束"""
        if self.stage != 'assemble': return
        parts = [x for x in self.assemble_sel if isinstance(x, Part)]
        if len(parts) != 2:
            messagebox.showinfo('提示', '请按住 Ctrl 选中恰好 2 个零件')
            return
        pA, pB = parts[0], parts[1]
        if pA is pB:
            messagebox.showinfo('提示', '两个零件相同'); return
        for (qa, qb) in self.collisions:
            if (qa is pA and qb is pB) or (qa is pB and qb is pA):
                messagebox.showinfo('提示', '碰撞约束已存在')
                return
        self.collisions.append((pA, pB))
        logger.info(f'[collision] 创建碰撞约束 零件#{pA.id}"{pA.name}"'
                    f' ↔ 零件#{pB.id}"{pB.name}"')
        self.assemble_sel.clear()
        self.set_status(f'碰撞装配: 零件#{pA.id} ↔ 零件#{pB.id}')
        self._touch()
        self.update_props()
        self.redraw()

    def clear_assemble_sel(self):
        self.assemble_sel.clear()
        self.update_props()
        self.redraw()

    def select_all_unconnected(self):
        self.assemble_sel.clear()
        for p in self.parts:
            for cp in p.points:
                if cp.link is None and cp.type in ('revolute', 'spring', 'magnet'):
                    self.assemble_sel.append(cp)
        self.update_props()
        self.redraw()

    # ---------------- 视图 ----------------
    def fit_view(self):
        pts = []
        for p in self.parts:
            for v in p.hull:
                pts.append(p.l2w(v))
        if not pts: return
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        W = self.canvas.winfo_width() or 980
        H = self.canvas.winfo_height() or 700
        if x1 > x0 and y1 > y0:
            zx = (W - 160) / (x1 - x0)
            zy = (H - 160) / (y1 - y0)
            self.zoom = max(0.1, min(3.0, min(zx, zy)))
        else:
            self.zoom = 1.0
        self.view_x = cx - W / 2 / self.zoom
        self.view_y = cy - H / 2 / self.zoom
        self.redraw()

    # ---------------- 主循环 ----------------
    def tick(self):
        self.redraw()
        state = self.sim.state
        n = len(self.sim.frames)
        self.hud.configure(text=f'Zoom {self.zoom*100:.0f}%   '
                                f'世界 ({self.mouse_world[0]:.0f}, '
                                f'{self.mouse_world[1]:.0f})   '
                                f'零件 {len(self.parts)}  连接 {len(self.links)}   '
                                f'仿真 {state}  帧 {n}')
        self.root.after(33, self.tick)

    # ---------------- 物理 ----------------
    def _apply_force_at(self, part, world_pt, force):
        if part.locked: return
        part.f  = part.f + force
        r = world_pt - part.pred_pos
        part.tq += r[0] * force[1] - r[1] * force[0]

    def substep(self, h):
        for p in self.parts:
            p.f[:] = 0.0; p.tq = 0.0
            if not p.locked:
                p.f[1] += p.all_mass() * GRAVITY
        for lk in self.links:
            if lk.type == 'spring': self._apply_spring(lk)
            elif lk.type == 'magnet': self._apply_magnet(lk)
            # 注意: 转动副静摩擦不在力累加阶段调用。
            # 它是角冲量, 直接修改 omega, 必须在速度回写之后调用
            # (见本方法末尾的 _apply_revolute_static_damping(lk, h))。
        for p in self.parts:
            if p.locked: continue
            m = p.all_mass(); I = p.inertia()
            p.v += p.f / m * h
            p.omega += p.tq / I * h
            p.v     *= max(0.0, 1.0 - LIN_DAMP * h)
            p.omega *= max(0.0, 1.0 - ANG_DAMP * h)
            sp = float(np.linalg.norm(p.v))
            if sp > MAX_LIN_VEL: p.v *= MAX_LIN_VEL / sp
        for p in self.parts:
            if p.locked:
                p.pred_pos = p.pos.copy(); p.pred_angle = p.angle
            else:
                p.pred_pos = p.pos + p.v * h
                p.pred_angle = p.angle + p.omega * h
        for _ in range(ITERATIONS):
            for lk in self.links:
                if lk.type == 'revolute':
                    self._solve_revolute(lk)
            # 弹簧限位位置约束 (硬约束, 恢复系数=0)
            for lk in self.links:
                if lk.type == 'spring':
                    self._solve_spring_limit_position(lk)
            # 碰撞位置约束 (硬约束, 沿最小穿透法线分离)
            for (pA, pB) in self.collisions:
                self._solve_collision_position(pA, pB)

        for p in self.parts:
            if p.locked: continue
            p.v = (p.pred_pos - p.pos) / h
            p.omega = (p.pred_angle - p.angle) / h
            p.pos = p.pred_pos.copy()
            p.angle = p.pred_angle

        # 转动副静摩擦 (角冲量式, 直接作用于 ω)
        # 迭代 FRICTION_ITER 次以处理多铰链耦合:
        #   - 单铰链时一次即精确归零 ω_rel
        #   - 当多个铰链共享同一零件时, 后一个铰链的冲量会改变前一个
        #     铰链的相对角速度, 通过 Gauss-Seidel 迭代逐步收敛,
        #     保证每个铰链的残余 ω_rel 都趋于零 → 静摩擦保持
        for _fric_it in range(FRICTION_ITER):
            for lk in self.links:
                if lk.type == 'revolute':
                    self._apply_revolute_static_damping(lk, h)

        # 弹簧限位速度级截停 (二次校正, 清除残余 v_rel)
        for lk in self.links:
            if lk.type == 'spring':
                self._apply_spring_limit_stop(lk)
                # 释放瞬间记录 (从 active -> inactive)
                if lk.limit_active:
                    pa, pb = lk.pA, lk.pB
                    L_now = float(np.linalg.norm(pb.world() - pa.world()))
                    still_on = (L_now <= lk.lmin or L_now >= lk.lmax)
                    if not still_on:
                        lk.limit_active = False
                        logger.info(
                            f'[LIMIT-OFF] link#{lk.id} '
                            f'L={L_now:.4f} limits=[{lk.lmin:.4f}, {lk.lmax:.4f}] (released)')

        # 碰撞速度约束 (非弹性, 法向相对速度归零, 吸收动能)
        for (pA, pB) in self.collisions:
            self._solve_collision_velocity(pA, pB)

    def _spring_force_components(self, lk):
        """分解弹簧的力 (都返回作用在 pA 上的力向量)

        返回 dict: {'base': 刚度力, 'limit': 限位力, 'damp': 阻尼力}

        限位模型 (单向硬限位, 刚度只提供恢复力):
          - 拉簧 (lmax >= l0): L > lmax 时施加限位刚度 (L - lmax) * 39k
          - 压簧 (lmax <  l0): L < lmax 时施加限位刚度 (L - lmax) * 39k
          限位处的动能吸收由速度级冲击约束处理
          (见 _apply_spring_limit_stop), 本函数不涉及阻尼。
        """
        pa, pb = lk.pA, lk.pB
        wa = pa.world(); wb = pb.world()
        d = wb - wa
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            z = np.zeros(2)
            return {'base': z, 'limit': z, 'damp': z}
        u = d / L

        va = pa.part.v + pa.part.omega * perp(rot(pa.part.angle) @ pa.local)
        vb = pb.part.v + pb.part.omega * perp(rot(pb.part.angle) @ pb.local)
        v_rel = float((vb - va) @ u)

        F_base = lk.k * (L - lk.l0)
        F_damp_base = lk.c * v_rel

        F_limit_stiff = 0.0
        if L < lk.lmin:
            F_limit_stiff = lk.k * LIMIT_STIFFNESS_MULT * (L - lk.lmin)
        elif L > lk.lmax:
            F_limit_stiff = lk.k * LIMIT_STIFFNESS_MULT * (L - lk.lmax)

        return {
            'base':  F_base        * u,
            'limit': F_limit_stiff * u,
            'damp':  F_damp_base   * u,
        }

    def _apply_spring(self, lk):
        parts = self._spring_force_components(lk)
        F_on_a = parts['base'] + parts['limit'] + parts['damp']
        F_on_b = -F_on_a
        wa = lk.pA.world(); wb = lk.pB.world()
        self._apply_force_at(lk.pA.part, wa, F_on_a)
        self._apply_force_at(lk.pB.part, wb, F_on_b)

    def _apply_spring_limit_stop(self, lk):
        """弹簧限位单向截停 (速度级冲击约束)

        原理: 当弹簧超过极限长度, 且两端在"继续超限"方向上仍有相对
        运动时, 直接把沿弹簧轴方向上的相对速度用冲量一次性清零。
        动能被瞬态吸收 (相当于完全非弹性碰撞, 恢复系数 e = 0),
        因此触底即止, 不反弹。

        约束:  C_dot = v_rel = 0
        冲量:   λ = -v_rel / (J · M⁻¹ · Jᵀ)
        应用:   Δv_A = -invM_A · λ · u,  Δω_A = -invI_A · λ · (rA × u)
                Δv_B = +invM_B · λ · u,  Δω_B = +invI_B · λ · (rB × u)

        一旦 v_rel 反向 (回弹), 本函数立即不再作用, 让零件自然脱离
        限位, 重新回到正常弹簧长度, 不受任何影响。
        """
        pa, pb = lk.pA, lk.pB
        pA, pB = pa.part, pb.part
        wa = pa.world(); wb = pb.world()
        d = wb - wa
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            return
        u = d / L

        if L <= lk.lmin:
            stop_dir = -1  # 触及下限，防止继续压缩
        elif L >= lk.lmax:
            stop_dir = +1  # 触及上限，防止继续拉伸
        else:
            return

        rA = rot(pA.angle) @ pa.local
        rB = rot(pB.angle) @ pb.local

        invMA = 0.0 if pA.locked else 1.0 / pA.all_mass()
        invMB = 0.0 if pB.locked else 1.0 / pB.all_mass()
        invIA = 0.0 if pA.locked else 1.0 / pA.inertia()
        invIB = 0.0 if pB.locked else 1.0 / pB.inertia()
        if invMA + invMB < 1e-12 and invIA + invIB < 1e-12:
            return

        # 有效质量 (只依赖位置, 速度变化不影响, 可一次算出)
        cross_rA_u = rA[0]*u[1] - rA[1]*u[0]
        cross_rB_u = rB[0]*u[1] - rB[1]*u[0]
        k_eff = (invMA + invMB)
        k_eff += invIA * cross_rA_u * cross_rA_u
        k_eff += invIB * cross_rB_u * cross_rB_u
        if k_eff < 1e-12:
            return

        # Gauss-Seidel 迭代 (通常一次即精确归零)
        for it in range(LIMIT_STOP_ITER):
            va = pA.v + pA.omega * perp(rA)
            vb = pB.v + pB.omega * perp(rB)
            v_rel = float((vb - va) @ u)
            # 判断是否"继续超限"
            if stop_dir > 0 and v_rel <= 0: break
            if stop_dir < 0 and v_rel >= 0: break

            lam = -v_rel / k_eff
            if not pA.locked:
                pA.v     = pA.v     - invMA * lam * u
                pA.omega = pA.omega - invIA * lam * cross_rA_u
            if not pB.locked:
                pB.v     = pB.v     + invMB * lam * u
                pB.omega = pB.omega + invIB * lam * cross_rB_u
            if it == 0:
                # 只在第一次迭代写日志 (反映实际被吸收的速度)
                logger.debug(
                    f'[STOP    ] link#{lk.id} it={it} '
                    f'v_rel_before={v_rel:+.4f} lam={lam:+.4e} '
                    f'stop_dir={stop_dir:+d} L={L:.3f} lmax={lk.lmax:.3f} '
                    f'vA=({pA.v[0]:+.2f},{pA.v[1]:+.2f}) wA={pA.omega:+.3f} | '
                    f'vB=({pB.v[0]:+.2f},{pB.v[1]:+.2f}) wB={pB.omega:+.3f}')

    def _apply_magnet(self, lk):
        if lk.broken: return
        pa, pb = lk.pA, lk.pB
        wa = pa.world(); wb = pb.world()
        d = wb - wa
        L = float(np.linalg.norm(d))
        if L < 1e-9: return
        u = d / L
        F = lk.mk * (L - lk.l0)
        if abs(F) > lk.strength:
            lk.broken = True
            return
        fa = F * u; fb = -fa
        self._apply_force_at(pa.part, wa, fa)
        self._apply_force_at(pb.part, wb, fb)

    def _solve_revolute(self, lk):
        pa, pb = lk.pA.part, lk.pB.part
        if pa.locked and pb.locked: return
        wA = lk.pA.world_pred(); wB = lk.pB.world_pred()
        e = wB - wA
        if np.linalg.norm(e) < 1e-10: return
        invMA = 0.0 if pa.locked else 1.0 / pa.all_mass()
        invMB = 0.0 if pb.locked else 1.0 / pb.all_mass()
        invIA = 0.0 if pa.locked else 1.0 / pa.inertia()
        invIB = 0.0 if pb.locked else 1.0 / pb.inertia()
        rA = rot(pa.pred_angle) @ lk.pA.local
        rB = rot(pb.pred_angle) @ lk.pB.local
        pA_ = perp(rA); pB_ = perp(rB)
        K = (invMA + invMB) * np.eye(2)
        K += invIA * np.outer(pA_, pA_)
        K += invIB * np.outer(pB_, pB_)
        K = K + REG_EPS * np.eye(2)
        try:
            lam = np.linalg.solve(K, e)
        except np.linalg.LinAlgError:
            return
        if not np.all(np.isfinite(lam)): return
        if not pa.locked:
            pa.pred_pos   = pa.pred_pos   + invMA * lam
            pa.pred_angle = pa.pred_angle + invIA * (pA_ @ lam)
        if not pb.locked:
            pb.pred_pos   = pb.pred_pos   - invMB * lam
            pb.pred_angle = pb.pred_angle - invIB * (pB_ @ lam)

    def _apply_revolute_static_damping(self, lk, h):
        """转动副的库仑静摩擦 / 滑动摩擦 (角冲量式)。

        物理意义: 铰链处的夹紧装置最多提供 ±M_max 的抗转动力矩。

        模型:
          - 消除相对角速度 ω_rel 所需角冲量 j_req = -ω_rel / k_eff
          - 若 |j_req| ≤ M_max·h: 静摩擦, 施加 j_req, ω_rel 一步归零
          - 若 |j_req| >  M_max·h: 滑动摩擦, 施加 ±M_max·h, ω_rel 减小

        其中 k_eff = invI_A + invI_B, 单位一致性:
            j 的单位是 kg·px²/s (= N·m·s 的内部表示)
            M_max·h 也是 kg·px²/s, 因为 static_damping 内部单位是 kg·px²/s²

        必须在速度回写之后调用 (直接修改 ω)。一次冲量即让静摩擦态
        完全停止相对转动, 不需要多次迭代。
        """
        if lk.type != 'revolute': return
        M_max_internal = lk.static_damping * PX_PER_M * PX_PER_M
        if M_max_internal <= 1e-9: return
        pa, pb = lk.pA.part, lk.pB.part
        if pa.locked and pb.locked: return

        omega_rel = pa.omega - pb.omega
        invIA = 0.0 if pa.locked else 1.0 / pa.inertia()
        invIB = 0.0 if pb.locked else 1.0 / pb.inertia()
        k_eff = invIA + invIB
        if k_eff < 1e-12: return

        # 消除 ω_rel 所需角冲量
        j_req = -omega_rel / k_eff
        j_max = M_max_internal * h

        if abs(j_req) <= j_max:
            j = j_req                       # 静摩擦: 一步完全停止
            lk.friction_saturated = False
        else:
            j = math.copysign(j_max, j_req) # 滑动摩擦: 提供极限力矩
            lk.friction_saturated = True

        if not pa.locked:
            pa.omega += j * invIA
            pa.angle += j * invIA * h  # 同步修正位置层面的漂移，彻底消除夹紧力矩抗转动时的蠕变现象
        if not pb.locked:
            pb.omega -= j * invIB
            pb.angle -= j * invIB * h

    def _solve_collision_position(self, pA, pB):
        """碰撞位置约束: 沿最小穿透法线分离两零件的凸包"""
        if pA.locked and pB.locked: return
        if len(pA.hull) < 3 or len(pB.hull) < 3: return
        polyA = [pA.pred_pos + rot(pA.pred_angle) @ v for v in pA.hull]
        polyB = [pB.pred_pos + rot(pB.pred_angle) @ v for v in pB.hull]
        overlap, normal, depth = convex_polygons_sat(polyA, polyB)
        if not overlap or depth < 1e-6: return
        invMA = 0.0 if pA.locked else 1.0 / pA.all_mass()
        invMB = 0.0 if pB.locked else 1.0 / pB.all_mass()
        tot = invMA + invMB
        if tot < 1e-12: return
        if not pA.locked:
            pA.pred_pos = pA.pred_pos - normal * depth * (invMA / tot)
        if not pB.locked:
            pB.pred_pos = pB.pred_pos + normal * depth * (invMB / tot)

    def _solve_collision_velocity(self, pA, pB):
        """碰撞速度约束 (完全非弹性): 法向相对速度归零, 吸收动能"""
        if pA.locked and pB.locked: return
        if len(pA.hull) < 3 or len(pB.hull) < 3: return
        polyA = [pA.pred_pos + rot(pA.pred_angle) @ v for v in pA.hull]
        polyB = [pB.pred_pos + rot(pB.pred_angle) @ v for v in pB.hull]
        overlap, normal, depth = convex_polygons_sat(polyA, polyB)
        if not overlap or depth < 1e-6: return
        vA = pA.v; vB = pB.v
        v_rel_n = float((vB - vA) @ normal)
        if v_rel_n >= 0: return                  # 正在分离, 不干预
        invMA = 0.0 if pA.locked else 1.0 / pA.all_mass()
        invMB = 0.0 if pB.locked else 1.0 / pB.all_mass()
        tot = invMA + invMB
        if tot < 1e-12: return
        lam = -v_rel_n / tot
        if not pA.locked:
            pA.v = pA.v - invMA * lam * normal
        if not pB.locked:
            pB.v = pB.v + invMB * lam * normal

    def _solve_spring_limit_position(self, lk):
        """弹簧限位位置约束 (硬约束, 恢复系数 = 0)

        这是让"触底即止"真正生效的关键:
          - 作用在 pred_pos / pred_angle 上, 而非 v / ω
          - 在 PBD 中, pos 在子步骤末尾被更新为 pred_pos,
            随后速度由 v = (pred_pos - pos_old) / h 推导出来。
            因此位置一旦被钳制到限位, 速度自然沿超限方向归零,
            动能被瞬间吸收, 零件不会在限位外侧来回弹跳。

        位置更新公式 (一维标量约束):
          约束 C = L - lmax
          λ = C / (J · M⁻¹ · Jᵀ)
          ΔwA = +invMA · u · λ,  ΔθA = +invIA · (pA_·u) · λ
          ΔwB = -invMB · u · λ,  ΔθB = -invIB · (pB_·u) · λ

        单向性: 只在超出限位一侧生效, 回弹方向不干预。
        """
        pa, pb = lk.pA, lk.pB
        pA, pB = pa.part, pb.part

        wa = pa.world_pred(); wb = pb.world_pred()
        d = wb - wa
        L = float(np.linalg.norm(d))
        if L < 1e-9: return
        u = d / L

        if L <= lk.lmin:
            C = L - lk.lmin
        elif L >= lk.lmax:
            C = L - lk.lmax
        else:
            return

        invMA = 0.0 if pA.locked else 1.0 / pA.all_mass()
        invMB = 0.0 if pB.locked else 1.0 / pB.all_mass()
        invIA = 0.0 if pA.locked else 1.0 / pA.inertia()
        invIB = 0.0 if pB.locked else 1.0 / pB.inertia()

        rA = rot(pA.pred_angle) @ pa.local
        rB = rot(pB.pred_angle) @ pb.local
        pA_ = perp(rA); pB_ = perp(rB)

        k_eff = (invMA + invMB) + invIA * float(pA_ @ u) ** 2 \
                                + invIB * float(pB_ @ u) ** 2
        if k_eff < 1e-12: return
        lam = C / k_eff

        if not pA.locked:
            pA.pred_pos   = pA.pred_pos   + invMA * lam * u
            pA.pred_angle = pA.pred_angle + invIA * float(pA_ @ u) * lam
        if not pB.locked:
            pB.pred_pos   = pB.pred_pos   - invMB * lam * u
            pB.pred_angle = pB.pred_angle - invIB * float(pB_ @ u) * lam

        # ---- 状态切换日志 (一次性, 只在触发/释放瞬间写) ----
        if not lk.limit_active:
            lk.limit_active = True
            kind = 'COMPRESSION' if C < 0 else 'TENSION'
            logger.warning(
                f'[LIMIT-ON ] link#{lk.id} kind={kind} '
                f'L={L:.4f} limits=[{lk.lmin:.4f}, {lk.lmax:.4f}] C={C:+.4f} '
                f'k_eff={k_eff:.3e} lam={lam:+.3e} '
                f'pA.part#{pA.id}(locked={pA.locked}) '
                f'pB.part#{pB.id}(locked={pB.locked})')
            logger.debug(
                f'           u=({u[0]:+.4f},{u[1]:+.4f}) '
                f'invM=({invMA:.3e},{invMB:.3e}) '
                f'invI=({invIA:.3e},{invIB:.3e})')

    # ---------------- 受力平衡求解 ----------------
    def _recompute_analysis_for_current_frame(self):
        """静力平衡求解。

        未知数分配 (关键):
          - 每条**有 static_damping** 的 revolute 铰链: 3 个未知数 (Fx, Fy, M)
          - 每条**无 static_damping** 的 revolute 铰链: 2 个未知数 (Fx, Fy)

        物理意义: 无夹紧装置的机械铰链在静力学里不应提供任何抵抗力矩,
        因此 M 必须为 0。若强行把 M 也作为未知数 (无约束), 那么方程组
        欠定, Tikhonov 正则化会给出"非零的极小范数解", 造成凭空出现
        的铰链力矩 (这是一个数值假象, 不是真实的夹紧力)。

        有 static_damping 时 M 为未知数, 表示铰链处的夹紧力矩;
        求解后若 |M| 超过 static_damping, 则视为滑动摩擦 (饱和)。
        """
        free = [p for p in self.parts if not p.locked]
        rev  = [lk for lk in self.links if lk.type == 'revolute']
        if not free or not rev:
            return

        # 每条铰链是否引入 M 未知数 (取决于 static_damping)
        M_limits = [lk.static_damping * PX_PER_M * PX_PER_M for lk in rev]
        has_M    = [lim > 1e-9 for lim in M_limits]

        # 变量分配: has_M 为 3 个 (Fx, Fy, M); 否则 2 个 (Fx, Fy)
        var_off = []
        nVar = 0
        for hm in has_M:
            var_off.append(nVar)
            nVar += 3 if hm else 2

        nEq = 3 * len(free)
        A = np.zeros((nEq, nVar)); b = np.zeros(nEq)
        pidx = {id(p): i for i, p in enumerate(free)}

        for j, lk in enumerate(rev):
            pA, pB = lk.pA.part, lk.pB.part
            wJ = lk.pA.world()
            off = var_off[j]
            cx = off; cy = off + 1
            cm = off + 2 if has_M[j] else None
            if id(pB) in pidx:
                i = pidx[id(pB)]; rB = wJ - pB.pos
                A[3*i,   cx] +=  1.0
                A[3*i+1, cy] +=  1.0
                A[3*i+2, cx] += -rB[1]
                A[3*i+2, cy] +=  rB[0]
                if cm is not None:
                    A[3*i+2, cm] +=  1.0
            if id(pA) in pidx:
                i = pidx[id(pA)]; rA = wJ - pA.pos
                A[3*i,   cx] += -1.0
                A[3*i+1, cy] += -1.0
                A[3*i+2, cx] +=  rA[1]
                A[3*i+2, cy] += -rA[0]
                if cm is not None:
                    A[3*i+2, cm] += -1.0

        for i, p in enumerate(free):
            Fx, Fy, Tq = 0.0, 0.0, 0.0
            # 本体重力 (只 body_mass)
            Fy += p.body_mass * GRAVITY
            # payload 力: 单独作用在各自挂接点, 产生真实力矩
            for cp in p.points:
                if cp.type == 'payload' and cp.mass > 0:
                    Fpy = cp.mass * GRAVITY
                    Fy += Fpy
                    r = cp.world() - p.pos
                    Tq += r[0] * Fpy
            for lk in self.links:
                if lk.type == 'spring':
                    for pt in (lk.pA, lk.pB):
                        if pt.part is not p: continue
                        sign = 1.0 if pt is lk.pA else -1.0
                        comp = self._spring_force_components(lk)
                        F_vec = sign * (comp['base'] + comp['limit'])
                        fx, fy = float(F_vec[0]), float(F_vec[1])
                        Fx += fx; Fy += fy
                        wa = pt.world()
                        r = wa - p.pos
                        Tq += r[0]*fy - r[1]*fx
                elif lk.type == 'magnet' and not lk.broken:
                    for pt in (lk.pA, lk.pB):
                        if pt.part is not p: continue
                        other = lk.pB if pt is lk.pA else lk.pA
                        wa = pt.world(); wb = other.world()
                        d = wb - wa; L = float(np.linalg.norm(d))
                        if L < 1e-9: continue
                        u = d / L
                        F = lk.mk * (L - lk.l0)
                        fx, fy = F * u[0], F * u[1]
                        Fx += fx; Fy += fy
                        r = wa - p.pos
                        Tq += r[0]*fy - r[1]*fx
            b[3*i]   = -Fx
            b[3*i+1] = -Fy
            b[3*i+2] = -Tq

        try:
            ATA = A.T @ A + REG_EPS * np.eye(A.shape[1])
            ATb = A.T @ b
            sol = np.linalg.solve(ATA, ATb)
        except np.linalg.LinAlgError:
            try:
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                return
        except Exception:
            return

        for j, lk in enumerate(rev):
            off = var_off[j]
            lk.force = np.array([sol[off], sol[off+1]])
            if has_M[j]:
                M_solved = float(sol[off+2])       # 静力平衡所需的"剩余力矩"
                M_limit  = M_limits[j]             # 铰链物理上限 ±M_max
                lk.moment_raw = M_solved
                if abs(M_solved) > M_limit:
                    # 所需力矩突破铰链上限 → 滑动摩擦
                    # 实际提供的力矩被钳制到 ±M_max (物理上不可能超过)
                    lk.moment = math.copysign(M_limit, M_solved)
                    lk.friction_saturated = True
                else:
                    # 静摩擦: 提供的力矩恰好等于维持平衡所需的剩余力矩
                    lk.moment = M_solved
                    lk.friction_saturated = False
            else:
                lk.moment = 0.0
                lk.moment_raw = 0.0
                lk.friction_saturated = False

    # ---------------- 渲染 ----------------
    def redraw(self):
        c = self.canvas
        W = c.winfo_width() or 980
        H = c.winfo_height() or 700
        c.delete('all')
        self._draw_grid(W, H)
        for p in self.parts:
            self._draw_part(p)
        # 碰撞约束可视化
        for (pA, pB) in self.collisions:
            xa, ya = self.w2s(pA.l2w(pA.com_local()))
            xb, yb = self.w2s(pB.l2w(pB.com_local()))
            self.canvas.create_line(xa, ya, xb, yb,
                                    fill='#d84315', width=2, dash=(3, 5))
            self.canvas.create_text((xa + xb) / 2, (ya + yb) / 2 - 10,
                                    text='碰撞', fill='#d84315',
                                    font=('Microsoft YaHei', 8, 'bold'))
        for lk in self.links:
            if lk.type == 'spring': self._draw_spring(lk)
            elif lk.type == 'magnet': self._draw_magnet(lk)
        for p in self.parts:
            for cp in p.points:
                self._draw_point(cp)
        if self.create_part_start is not None and self.mouse_screen:
            x0, y0 = self.create_part_start
            x1, y1 = self.mouse_screen
            self.canvas.create_rectangle(x0, y0, x1, y1,
                                         outline='#2196f3', dash=(5, 3), width=2)

        # 拖拽时的约束限制可视化 (弹簧极限圆弧辅助线)
        self._draw_spring_limits_overlay()

        # 受力可视化 (只在仿真阶段 / 有缓存 / 选中零件 / analysis_detail != off)
        if (self.stage == 'simulate'
                and self.analysis_detail != 'off'
                and self.sim.force_cache
                and self.sel and self.sel[0] == 'part'):
            self._draw_part_forces(self.sel[1])

        self._draw_ruler(W, H)
        self._draw_scale_bar(W, H)
        tip = {
            'design':   '① 设计阶段: 空白拖拽创建零件 | 点击零件加连接附 | 拖动连接附/顶点',
            'assemble': '② 装配阶段: 拖动零件沿约束运动 | Ctrl+多选连接附 → 点"装配"',
            'simulate': '③ 仿真阶段: 点击零件查看受力 (点击空白处隐藏受力)',
        }[self.stage]
        c.create_text(10, H - 12, anchor='sw', fill='#999999',
                      font=('Microsoft YaHei', 8), text=tip)
                      
        self._draw_energy_chart()

    def _draw_energy_chart(self):
        """在右下角专门的 Canvas 中绘制能量折线图 (分为4个独立子坐标系)"""
        c = self.energy_canvas
        c.delete('all')
        if self.stage != 'simulate':
            c.create_text(200, 120, text='仅在仿真阶段显示', fill='#666666', font=('Microsoft YaHei', 10))
            return
        if not self.sim.energy_cache:
            c.create_text(200, 120, text='暂无能量数据，请点击[开始仿真]', fill='#666666', font=('Microsoft YaHei', 10))
            return
            
        cache = self.sim.energy_cache
        n = len(cache)
        if n < 2: return
        
        W = c.winfo_width() or 410
        H = c.winfo_height() or 230
        pad_l, pad_r, pad_t, pad_b = 45, 10, 20, 20
        
        keys = ['e_tot', 'ek', 'ep_g', 'ep_e']
        colors = ['#f44336', '#2196f3', '#9e9e9e', '#4caf50']
        labels = ['总能量', '动能', '重力势', '弹性势']
        
        num_plots = 4
        plot_H = (H - pad_t - pad_b) / num_plots
        
        # 自适应计算各自的坐标范围
        if getattr(self, 'eng_auto_scale', True):
            for j, key in enumerate(keys):
                v_max_g = max(d[key] for d in cache)
                v_min_g = min(d[key] for d in cache)
                dv = v_max_g - v_min_g
                if dv < 1e-9: dv = 1.0
                self.eng_ranges[j]['vmax'] = v_max_g + dv * 0.1
                self.eng_ranges[j]['vmin'] = v_min_g - dv * 0.1
                
        # 逐个画出4个子坐标系
        for j in range(num_plots):
            y_top = pad_t + j * plot_H
            y_bot = y_top + plot_H
            
            v_max = self.eng_ranges[j]['vmax']
            v_min = self.eng_ranges[j]['vmin']
            if abs(v_max - v_min) < 1e-9:
                v_max += 1.0; v_min -= 1.0
            
            def p2c_j(i, v, ytop=y_top, ybot=y_bot, vmax=v_max, vmin=v_min):
                x = pad_l + (W - pad_l - pad_r) * (i / (n - 1))
                y = ytop + (ybot - ytop) * (1 - (v - vmin) / (vmax - vmin))
                return x, y
            
            pts = []
            for i, d in enumerate(cache):
                pts.extend(p2c_j(i, d[keys[j]]))
                
            # 画 0 基准线
            if v_min <= 0 <= v_max:
                _, y0 = p2c_j(0, 0)
                c.create_line(pad_l, y0, W - pad_r, y0, fill='#444444', dash=(2, 2))
                
            # 画曲线 (截断超出的部分避免溢出到其他子图区)
            pts_clipped = []
            for i in range(0, len(pts), 2):
                pts_clipped.extend([pts[i], max(y_top, min(y_bot, pts[i+1]))])
            c.create_line(*pts_clipped, fill=colors[j], width=1.5 if j>0 else 2)
            
            # 坐标轴和分界线
            c.create_line(pad_l, y_top, pad_l, y_bot, fill='#888888')
            if j == num_plots - 1:
                c.create_line(pad_l, y_bot, W - pad_r, y_bot, fill='#888888')
            else:
                c.create_line(pad_l, y_bot, W - pad_r, y_bot, fill='#555555', dash=(4,4))
                
            # 文本标签
            c.create_text(pad_l - 4, y_top + 6, text=f"{v_max:.2f}", fill=colors[j], anchor='e', font=('Consolas', 8))
            c.create_text(pad_l - 4, y_bot - 6, text=f"{v_min:.2f}", fill=colors[j], anchor='e', font=('Consolas', 8))
            c.create_text(pad_l + 4, y_top + 8, text=labels[j], fill=colors[j], anchor='nw', font=('Microsoft YaHei', 8, 'bold'))
        
        # 游标 (当前帧竖线)
        idx = self.sim.playhead
        if idx < n:
            cx = pad_l + (W - pad_l - pad_r) * (idx / (n - 1))
            c.create_line(cx, pad_t, cx, H - pad_b, fill='#ffeb3b', dash=(4, 2))
            
            d = cache[idx]
            # 画每个子图的交点圆圈
            for j, key in enumerate(keys):
                y_top = pad_t + j * plot_H
                y_bot = y_top + plot_H
                v_max = self.eng_ranges[j]['vmax']
                v_min = self.eng_ranges[j]['vmin']
                if abs(v_max - v_min) < 1e-9: v_max += 1.0; v_min -= 1.0
                cy = y_top + (y_bot - y_top) * (1 - (d[key] - v_min) / (v_max - v_min))
                
                if y_top <= cy <= y_bot:
                    c.create_oval(cx-3, cy-3, cx+3, cy+3, fill=colors[j], outline='')
            
            # 顶部数值显示框 (独立一行悬浮显示当前帧所有值)
            c.create_rectangle(pad_l, 0, W - pad_r + 40, pad_t - 2, fill='#2b2b2b', outline='')
            c.create_text(pad_l, 10, text=f"总: {d['e_tot']:.3f}", fill='#f44336', anchor='w', font=('Consolas', 9))
            c.create_text(pad_l + 75, 10, text=f"动能: {d['ek']:.3f}", fill='#2196f3', anchor='w', font=('Consolas', 9))
            c.create_text(pad_l + 150, 10, text=f"重力: {d['ep_g']:.3f}", fill='#9e9e9e', anchor='w', font=('Consolas', 9))
            c.create_text(pad_l + 230, 10, text=f"弹性: {d['ep_e']:.3f}", fill='#4caf50', anchor='w', font=('Consolas', 9))
            c.create_text(W - pad_r, 10, text=f"帧: {idx}", fill='#ffeb3b', anchor='e', font=('Consolas', 9))

    # ---------------- 能量图表交互事件 ----------------
    def _get_eng_plot_idx(self, py):
        H = self.energy_canvas.winfo_height() or 230
        pad_t, pad_b = 20, 20
        plot_H = (H - pad_t - pad_b) / 4.0
        if plot_H <= 0: return 0
        idx = int((py - pad_t) / plot_H)
        return max(0, min(3, idx))

    def on_eng_press(self, e):
        self.eng_panning = True
        self.eng_auto_scale = False
        self.eng_pan_idx = self._get_eng_plot_idx(e.y)
        self.eng_pan_y = e.y
        self.eng_pan_vmin = self.eng_ranges[self.eng_pan_idx]['vmin']
        self.eng_pan_vmax = self.eng_ranges[self.eng_pan_idx]['vmax']

    def on_eng_drag(self, e):
        if not getattr(self, 'eng_panning', False): return
        if getattr(self, 'eng_pan_idx', -1) < 0: return
        
        dy = e.y - self.eng_pan_y
        H = self.energy_canvas.winfo_height() or 230
        pad_t, pad_b = 20, 20
        plot_H = (H - pad_t - pad_b) / 4.0
        if plot_H <= 0: return
        
        # 画布向下拖动时，对应数值范围向上平移
        dv_per_py = (self.eng_pan_vmax - self.eng_pan_vmin) / plot_H
        shift_v = dy * dv_per_py
        
        self.eng_ranges[self.eng_pan_idx]['vmin'] = self.eng_pan_vmin + shift_v
        self.eng_ranges[self.eng_pan_idx]['vmax'] = self.eng_pan_vmax + shift_v
        self.redraw()

    def on_eng_release(self, e):
        self.eng_panning = False
        self.eng_pan_idx = -1

    def on_eng_wheel(self, e):
        self.eng_auto_scale = False
        H = self.energy_canvas.winfo_height() or 230
        pad_t, pad_b = 20, 20
        plot_H = (H - pad_t - pad_b) / 4.0
        if plot_H <= 0: return

        idx = self._get_eng_plot_idx(e.y)

        if e.num == 4:   delta = 1
        elif e.num == 5: delta = -1
        else:            delta = 1 if (e.delta or 0) > 0 else -1
        
        factor = 0.8 if delta > 0 else 1.25

        y_top = pad_t + idx * plot_H
        py = e.y
        if py < y_top: py = y_top
        if py > y_top + plot_H: py = y_top + plot_H
        
        vmin = self.eng_ranges[idx]['vmin']
        vmax = self.eng_ranges[idx]['vmax']
        
        # 根据鼠标当前所在 Y 坐标计算对应能量值，作为缩放定点中心
        ratio = 1.0 - (py - y_top) / plot_H
        v_mouse = vmin + ratio * (vmax - vmin)

        span = vmax - vmin
        new_span = span * factor
        
        self.eng_ranges[idx]['vmin'] = v_mouse - ratio * new_span
        self.eng_ranges[idx]['vmax'] = v_mouse + (1.0 - ratio) * new_span
        self.redraw()

    def on_eng_reset(self, e):
        """双击或右键恢复自动缩放默认范围"""
        self.eng_auto_scale = True
        self.redraw()

    def _draw_spring_limits_overlay(self):
        if self.stage not in ('assemble', 'simulate'): return
        
        def _draw_circles(lk, center_pt, interactive=False):
            c_s = self.w2s(center_pt.world())
            rmin = lk.lmin * self.zoom
            r0   = lk.l0 * self.zoom
            rmax = lk.lmax * self.zoom
            
            # 如果正在拖拽某个环，高亮该环，体现交互反馈
            active = None
            if self.dragging and self.dragging[0] == 'spring_limit' and self.dragging[1] is lk:
                active = self.dragging[2]
                
            w_min = 4 if active == 'lmin' else 2
            w_0   = 3 if active == 'l0' else 1
            w_max = 4 if active == 'lmax' else 2
            
            # 压缩极限制动辅助线 (最短)
            self.canvas.create_oval(c_s[0]-rmin, c_s[1]-rmin, c_s[0]+rmin, c_s[1]+rmin, 
                                    outline='#ef5350', dash=(4, 4), width=w_min)
            # 初始长度辅助线
            self.canvas.create_oval(c_s[0]-r0, c_s[1]-r0, c_s[0]+r0, c_s[1]+r0, 
                                    outline='#8bc34a', dash=(4, 4), width=w_0)
            # 拉伸极限制动辅助线 (最长)
            self.canvas.create_oval(c_s[0]-rmax, c_s[1]-rmax, c_s[0]+rmax, c_s[1]+rmax, 
                                    outline='#ef5350', dash=(4, 4), width=w_max)
                                    
            tag = ' (可拖拽)' if interactive else ''
            self.canvas.create_text(c_s[0], c_s[1] - rmin + 12, text='限位下界'+tag, fill='#ef5350', font=('Microsoft YaHei', 8, 'bold'))
            self.canvas.create_text(c_s[0], c_s[1] - r0 - 6, text='初始 L0'+tag, fill='#8bc34a', font=('Microsoft YaHei', 8, 'bold' if active == 'l0' else 'normal'))
            self.canvas.create_text(c_s[0], c_s[1] - rmax - 6, text='限位上界'+tag, fill='#ef5350', font=('Microsoft YaHei', 8, 'bold'))

        # 仿真阶段：检查常驻显示开关
        if self.stage == 'simulate':
            if getattr(self, 'show_spring_limits', False):
                for lk in self.links:
                    if lk.type == 'spring':
                        # 仿真时，默认以 pA 为中心画即可
                        _draw_circles(lk, lk.pA)
            return

        # 1. 装配阶段：拖动任意零件时，显示所有弹簧的极限圆环
        if self.dragging and self.dragging[0] == 'part':
            drag_part = self.dragging[1]
            for lk in self.links:
                if lk.type != 'spring': continue
                is_A_drag = (lk.pA.part is drag_part)
                is_B_drag = (lk.pB.part is drag_part)
                # 以未拖动端为圆心；如果两端都没拖动（其他零件被拖），默认 pA
                if is_A_drag and not is_B_drag:
                    _draw_circles(lk, lk.pB)
                elif is_B_drag and not is_A_drag:
                    _draw_circles(lk, lk.pA)
                else:
                    _draw_circles(lk, lk.pA)
            return
            
        # 2. 装配阶段：选中弹簧或正在拖拽圆环时的视觉交互
        lk_sel = None
        if self.sel and self.sel[0] == 'link' and self.sel[1].type == 'spring':
            lk_sel = self.sel[1]
        elif self.dragging and self.dragging[0] == 'spring_limit':
            lk_sel = self.dragging[1]
            
        if lk_sel:
            # 默认以 pA 作为圆心绘制交互调节环
            _draw_circles(lk_sel, lk_sel.pA, interactive=True)

    # ---------------- 受力可视化 ----------------
    def _draw_part_forces(self, part):
        idx = self.sim.playhead
        if idx >= len(self.sim.force_cache):
            idx = len(self.sim.force_cache) - 1
        if idx < 0: return
        cache = self.sim.force_cache[idx]
        if part.id not in cache: return
        data = cache[part.id]

        # ---- 各连接附: 力 + 力矩分量 ----
        for cp_id, cpd in data['cp_forces'].items():
            at = cpd['at']
            
            # 显示点ID，便于和右侧附加力矩对应
            sx, sy = self.w2s(at)
            self.canvas.create_text(sx + 14, sy - 14, text=f'点#{cp_id}', fill='#ff9800', font=('Microsoft YaHei', 9, 'bold'))

            if self.analysis_detail == 'simple':
                self._draw_force_arrow(at, cpd['total_f'], 'total_f')
                if abs(cpd.get('total_m', 0.0)) > 1e-3:
                    self._draw_moment_arc(at, cpd['total_m'], 'total_m')
            else:
                for kind, F, at_comp in cpd['components']:
                    self._draw_force_arrow(at_comp, F, kind)
                for kind, M, at_m in cpd.get('moments', []):
                    self._draw_moment_arc(at_m, M, kind)

        # ---- 质心处: 重力 + 附加力矩组 ----
        com = data['com_world']
        if self.analysis_detail == 'detailed':
            self._draw_force_arrow(com, data['gravity'], 'gravity')

        # 绘制所有由于受力偏离质心产生的附加力矩 (直接连接力矩如铰链的，保留在原地不画在这里)
        eccentric_torques = [t for t in data.get('torques', []) if not t.get('is_direct', False)]
        if eccentric_torques:
            self._draw_stacked_moments(com, eccentric_torques)

    def _draw_force_arrow(self, at_world, F, kind):
        cfg = FORCE_VIZ.get(kind)
        if cfg is None: return
        F = np.asarray(F)
        mag_px_kg_s2 = float(np.linalg.norm(F))   # 内部单位
        if mag_px_kg_s2 < 1e-6: return
        F_N = mag_px_kg_s2 / PX_PER_M             # 牛顿

        world_len = F_N * FORCE_SCALE_PX_PER_N
        if world_len < FORCE_MIN_WORLD_PX: return
        if world_len > FORCE_MAX_WORLD_PX:
            world_len = FORCE_MAX_WORLD_PX

        ux, uy = F[0] / mag_px_kg_s2, F[1] / mag_px_kg_s2
        sx, sy = self.w2s(at_world)
        ex, ey = sx + ux * world_len * self.zoom, sy + uy * world_len * self.zoom

        self.canvas.create_line(sx, sy, ex, ey,
                                fill=cfg['color'], width=cfg['width'],
                                arrow='last', arrowshape=(9, 12, 4))

        # 标注力值 (N): 沿箭头方向再偏移 14px, 并按方向选择对齐方式
        # 这样文字始终位于箭头尖端之外, 不会与箭头本体重叠
        extra = 14.0
        lx = ex + ux * extra
        ly = ey + uy * extra
        if ux > 0.3:
            anchor = 'w'
        elif ux < -0.3:
            anchor = 'e'
        else:
            anchor = 'center'
        # 竖直方向额外补偿, 避免压到箭头
        if abs(uy) > 0.7:
            ly += 4 if uy > 0 else -4
        self.canvas.create_text(lx, ly, text=f'{F_N:.2f}N',
                                anchor=anchor,
                                fill=cfg['color'],
                                font=FORCE_LABEL_FONT)

    def _draw_moment_arc(self, at_world, M, kind):
        """力矩圆弧 + 明显箭头 + 数值

        单位换算: M 内部单位 kg·px²/s² -> N·m 需除以 PX_PER_M²
        """
        cfg = FORCE_VIZ.get(kind)
        if cfg is None: return
        M_Nm = M / (PX_PER_M * PX_PER_M)   # kg·px²/s² -> N·m
        if abs(M_Nm) < 1e-6: return
        
        # 使用对数缩放基础半径，兼顾微小力矩与超大力矩的可视化
        r_world = MOMENT_MIN_WORLD_PX + math.log10(1.0 + abs(M_Nm) * 50.0) * 15.0
        if r_world > MOMENT_MAX_WORLD_PX:
            r_world = MOMENT_MAX_WORLD_PX
        r = r_world * self.zoom

        sx, sy = self.w2s(at_world)
        direction = 1 if M_Nm > 0 else -1

        start_deg = 40.0
        extent_deg = -140.0 * direction
        end_deg = start_deg + extent_deg

        # 圆弧本体
        self.canvas.create_arc(sx - r, sy - r, sx + r, sy + r,
                               start=start_deg, extent=extent_deg,
                               style='arc',
                               outline=cfg['color'],
                               width=cfg['width'])

        # 末端箭头: 在弧线末端沿切向绘制
        end_rad = math.radians(end_deg)
        ex = sx + r * math.cos(end_rad)
        ey = sy - r * math.sin(end_rad)   # tkinter arc 使用标准数学角

        # 切线方向: extent < 0 (逆时针) -> 切向 = 角 - 90°
        tangent_deg = end_deg - 90.0 * direction
        tangent_rad = math.radians(tangent_deg)
        tlen = 16.0
        tx = ex + tlen * math.cos(tangent_rad)
        ty = ey - tlen * math.sin(tangent_rad)

        self.canvas.create_line(ex, ey, tx, ty,
                                fill=cfg['color'],
                                width=max(2, cfg['width'] + 1),
                                arrow='last',
                                arrowshape=(14, 18, 7))

        # 数值
        self.canvas.create_text(sx + r * 0.75, sy - r * 0.75 - 8,
                                text=f'{abs(M_Nm):.3f}N·m',
                                fill=cfg['color'],
                                font=FORCE_LABEL_FONT)

    def _draw_stacked_moments(self, at_world, torques):
        """
        在指定位置绘制多个力矩的堆叠圆弧。
        torques: [{'M': float, 'kind': str}, ...]
        半径错开，切轴向角度首尾相连。
        """
        if not torques: return
        sx, sy = self.w2s(at_world)
        
        current_deg = 90.0  # 初始方向从正上方开始
        
        for i, tq in enumerate(torques):
            M_Nm = tq['M'] / (PX_PER_M * PX_PER_M)
            if abs(M_Nm) < 1e-6: continue
            
            kind = tq['kind']
            cfg = FORCE_VIZ.get(kind, FORCE_VIZ.get('other'))
            
            # 屏幕坐标系下 Y向下，M>0 对应顺时针。
            # tkinter 里的 create_arc, start=0 为右侧, extent > 0 为逆时针。
            # 所以 M>0 (顺时针) 对应 extent < 0。
            extent_deg = - M_Nm * 120.0
            if abs(extent_deg) < 20.0:
                extent_deg = math.copysign(20.0, extent_deg)
            if abs(extent_deg) > 340.0:
                extent_deg = math.copysign(340.0, extent_deg)
                
            start_deg = current_deg
            end_deg = start_deg + extent_deg
            
            # 半径采用对数缩放作为基础半径，加上偏移量分层显示
            base_r_world = MOMENT_MIN_WORLD_PX + math.log10(1.0 + abs(M_Nm) * 50.0) * 15.0
            r_world = base_r_world + i * 6.0
            if r_world > MOMENT_MAX_WORLD_PX * 1.5: r_world = MOMENT_MAX_WORLD_PX * 1.5
            r = r_world * self.zoom
            
            self.canvas.create_arc(sx - r, sy - r, sx + r, sy + r,
                                   start=start_deg, extent=extent_deg,
                                   style='arc',
                                   outline=cfg['color'],
                                   width=cfg['width'])
                                   
            # 末端箭头
            end_rad = math.radians(end_deg)
            ex = sx + r * math.cos(end_rad)
            ey = sy - r * math.sin(end_rad)
            
            # 切线方向 (依据 extent_deg 正负决定)
            direction = 1 if extent_deg > 0 else -1
            tangent_deg = end_deg + (90.0 * direction)
            tangent_rad = math.radians(tangent_deg)
            tlen = 14.0
            tx = ex + tlen * math.cos(tangent_rad)
            ty = ey - tlen * math.sin(tangent_rad)
            
            self.canvas.create_line(ex, ey, tx, ty,
                                    fill=cfg['color'],
                                    width=max(2, cfg['width']),
                                    arrow='last',
                                    arrowshape=(12, 14, 5))
                                    
            current_deg = end_deg

    def _draw_grid(self, W, H):
        c = self.canvas
        if   self.zoom >= 3.0: step = 20
        elif self.zoom >= 1.5: step = 50
        elif self.zoom >= 0.8: step = 100
        elif self.zoom >= 0.4: step = 200
        else:                  step = 500
        wx0, wy0 = self.s2w(0, 0)
        wx1, wy1 = self.s2w(W, H)
        x = math.floor(wx0 / step) * step
        while x <= wx1:
            sx = self.w2sx(x)
            c.create_line(sx, 0, sx, H, fill='#e8e8e0')
            x += step
        y = math.floor(wy0 / step) * step
        while y <= wy1:
            sy = self.w2sy(y)
            c.create_line(0, sy, W, sy, fill='#e8e8e0')
            y += step

    def _nice_step(self, rough):
        if rough <= 0: return 1.0
        exp = math.floor(math.log10(rough))
        base = 10 ** exp
        for f in (1, 2, 5, 10):
            if rough <= base * f:
                return base * f
        return base * 10

    def _draw_ruler(self, W, H):
        c = self.canvas
        step = self._nice_step(80 / self.zoom)
        wx0, wy0 = self.s2w(0, 0)
        wx1, wy1 = self.s2w(W, H)
        x = math.ceil(wx0 / step) * step
        while x <= wx1:
            sx = self.w2sx(x)
            c.create_line(sx, 0, sx, 6, fill='#888888')
            c.create_text(sx + 3, 10, anchor='nw', text=f'{x:.0f}',
                          fill='#666666', font=('Consolas', 8))
            x += step
        y = math.ceil(wy0 / step) * step
        while y <= wy1:
            sy = self.w2sy(y)
            c.create_line(0, sy, 6, sy, fill='#888888')
            c.create_text(8, sy + 1, anchor='nw', text=f'{y:.0f}',
                          fill='#666666', font=('Consolas', 8))
            y += step

    def _draw_scale_bar(self, W, H):
        c = self.canvas
        step = self._nice_step(120 / self.zoom)
        bar_px = step * self.zoom
        mm = step / PX_PER_M * 1000
        x1 = W - 30; x0 = x1 - bar_px; y0 = H - 40
        c.create_rectangle(x0 - 6, y0 - 20, x1 + 6, y0 + 10,
                           fill='#ffffff', outline='#cccccc')
        c.create_line(x0, y0, x1, y0, fill='#333333', width=3)
        c.create_line(x0, y0 - 5, x0, y0 + 5, fill='#333333', width=2)
        c.create_line(x1, y0 - 5, x1, y0 + 5, fill='#333333', width=2)
        c.create_text((x0 + x1) / 2, y0 - 9, text=f'{mm:.0f} mm',
                      fill='#333333', font=('Consolas', 9, 'bold'))
        c.create_text(x1 + 2, y0 + 4, anchor='ne',
                      text=f'zoom {self.zoom*100:.0f}%',
                      fill='#888888', font=('Consolas', 8))

    def _draw_part(self, part):
        c = self.canvas
        pts = []
        for v in part.hull:
            sx, sy = self.w2s(part.l2w(v))
            pts.extend([sx, sy])
        if len(pts) < 6: return
        if part.locked:
            fill_col = '#cfd8dc'; outline_col = '#455a64'
        else:
            fill_col = '#e3f2fd'; outline_col = '#1e88e5'

        # 装配阶段的多选高亮 (Part 在 assemble_sel 中)
        in_sel = (self.stage == 'assemble' and part in self.assemble_sel)
        if in_sel:
            outline_col = '#e69500'
            if not part.locked:
                fill_col = '#ffe0b2'

        if self.sel and self.sel[0] == 'part' and self.sel[1] is part:
            outline_col = '#ff6f00'

        c.create_polygon(*pts, fill=fill_col, outline=outline_col, width=2)

        cx = sum(pts[0::2]) / (len(pts) // 2)
        cy = sum(pts[1::2]) / (len(pts) // 2)

        # 多选序号角标
        if in_sel:
            idx = self.assemble_sel.index(part) + 1
            c.create_oval(cx - 13, cy - 32, cx + 13, cy - 10,
                          fill='#e69500', outline='#7f4f00')
            c.create_text(cx, cy - 21, text=str(idx),
                          fill='white',
                          font=('Arial', 11, 'bold'))

        if part.locked:
            c.create_text(cx, cy, text='锁定', fill='#263238',
                          font=('Microsoft YaHei', 8, 'bold'))

    def _draw_point(self, cp):
        c = self.canvas
        sx, sy = self.w2s(cp.world())
        col_map = {'revolute': '#37474f', 'spring': '#2e7d32',
                   'magnet':   '#7e57c2', 'payload': '#00695c'}
        col = col_map.get(cp.type, '#333')
        r = 7
        if cp.type == 'revolute':
            c.create_oval(sx-r, sy-r, sx+r, sy+r, fill='white',
                          outline=col, width=2)
            c.create_oval(sx-3, sy-3, sx+3, sy+3, fill=col, outline=col)
        elif cp.type == 'spring':
            c.create_rectangle(sx-r, sy-r, sx+r, sy+r, fill='white',
                               outline=col, width=2)
        elif cp.type == 'magnet':
            c.create_polygon(sx, sy-r, sx+r, sy, sx, sy+r, sx-r, sy,
                             fill='white', outline=col, width=2)
        elif cp.type == 'payload':
            c.create_oval(sx-r-1, sy-r-1, sx+r+1, sy+r+1,
                          fill='#b2dfdb', outline=col, width=2)
            c.create_text(sx, sy, text='▼', fill=col,
                          font=('Arial', 9, 'bold'))
        if cp in self.assemble_sel:
            c.create_oval(sx-r-8, sy-r-8, sx+r+8, sy+r+8,
                          outline='#e69500', width=3)
            c.create_text(sx, sy - r - 14,
                          text=str(self.assemble_sel.index(cp) + 1),
                          fill='#e69500', font=('Arial', 9, 'bold'))
        if self.sel and self.sel[0] == 'point' and self.sel[1] is cp:
            c.create_oval(sx-r-4, sy-r-4, sx+r+4, sy+r+4,
                          outline='#ff6f00', width=2)
        if (self.dragging and self.dragging[0] == 'point'
                and self.dragging[1] is cp):
            c.create_oval(sx-r-6, sy-r-6, sx+r+6, sy+r+6,
                          outline='#00b0ff', width=2, dash=(4, 2))
        if self._snap_target is cp:
            c.create_oval(sx-r-10, sy-r-10, sx+r+10, sy+r+10,
                          outline='#00e676', width=3)

    def _draw_spring(self, lk):
        c = self.canvas
        xa, ya = self.w2s(lk.pA.world())
        xb, yb = self.w2s(lk.pB.world())
        dx, dy = xb - xa, yb - ya
        L = math.hypot(dx, dy)
        if L < 1e-6: return
        ux, uy = dx / L, dy / L
        px, py = -uy, ux
        n = 10
        seg = L / (n * 2 + 2)
        pts = [xa, ya]
        for i in range(n * 2):
            off = 7.0 if i % 2 == 0 else -7.0
            t = seg * (i + 1)
            pts.extend([xa + ux * t + px * off, ya + uy * t + py * off])
        pts.extend([xb, yb])
        
        Lw = float(np.linalg.norm(lk.pB.world() - lk.pA.world()))
        
        # 智能动态颜色: 双边渐变到红色 (#c62828)
        if Lw <= lk.lmin or Lw >= lk.lmax:
            col = '#c62828'
        elif Lw < lk.l0:
            ratio = min(1.0, (lk.l0 - Lw) / max(0.1, lk.l0 - lk.lmin))
            r = int(0x54 + ratio * (0xc6 - 0x54))
            g = int(0x6e + ratio * (0x28 - 0x6e))
            b = int(0x7a + ratio * (0x28 - 0x7a))
            col = f'#{r:02x}{g:02x}{b:02x}'
        else:
            ratio = min(1.0, (Lw - lk.l0) / max(0.1, lk.lmax - lk.l0))
            r = int(0x54 + ratio * (0xc6 - 0x54))
            g = int(0x6e + ratio * (0x28 - 0x6e))
            b = int(0x7a + ratio * (0x28 - 0x7a))
            col = f'#{r:02x}{g:02x}{b:02x}'
                
        sel = bool(self.sel and self.sel[0] == 'link' and self.sel[1] is lk)
        w = 3 if sel else 2
        c.create_line(*pts, fill=col, width=w)
        
        if self.zoom > 0.5:
            mx, my = (xa + xb) / 2, (ya + yb) / 2
            dl_m = (Lw - lk.l0) / PX_PER_M
            c.create_text(mx, my + 17,
                          text=f'k={lk.k:.0f} c={lk.c:.1f} ΔL={dl_m:+.3f}m',
                          fill=col, font=('Consolas', 8))

    def _draw_magnet(self, lk):
        c = self.canvas
        xa, ya = self.w2s(lk.pA.world())
        xb, yb = self.w2s(lk.pB.world())
        col = '#9e9e9e' if lk.broken else '#7e57c2'
        sel = bool(self.sel and self.sel[0] == 'link' and self.sel[1] is lk)
        w = 4 if sel else 3
        c.create_line(xa, ya, xb, yb, fill=col, width=w, dash=(6, 4))
        if self.zoom > 0.55:
            c.create_text((xa+xb)/2, (ya+yb)/2 - 10,
                          text='磁吸' + ('×' if lk.broken else ''),
                          fill=col, font=('Microsoft YaHei', 8))

    # ---------------- 属性面板 ----------------
    def set_status(self, t):
        self.status.configure(text=t)

    def _field(self, parent, label, value, cmd, width=12):
        f = tk.Frame(parent, bg='#3c3f41')
        f.pack(fill='x', pady=2)
        tk.Label(f, text=label, width=11, anchor='w',
                 bg='#3c3f41', fg='#cccccc',
                 font=('Microsoft YaHei', 9)).pack(side='left')
        var = tk.StringVar(value=str(value))
        e = tk.Entry(f, textvariable=var, width=width, bg='#2b2b2b',
                     fg='#eeeeee', insertbackground='white', relief='flat')
        e.pack(side='left', fill='x', expand=True)
        e.bind('<Return>',   lambda ev: cmd(var.get()))
        e.bind('<FocusOut>', lambda ev: cmd(var.get()))
        
        # 增加对变量写入变动的实时监听（只要发生改变就尝试执行保存指令）
        def _on_write(*args):
            cmd(var.get())
        var.trace_add('write', _on_write)
        
        return var

    def _get_props_key(self):
        """右侧面板当前对应的唯一标识; 变化时需重建结构"""
        if self.sel is None:
            return ('none', self.stage, self.analysis_detail,
                    len(self.sim.frames) > 0, self.sim.state)
        kind, obj = self.sel
        return (kind, id(obj), self.stage, self.analysis_detail,
                len(self.sim.frames) > 0, self.sim.state,
                len(self.assemble_sel) if self.stage == 'assemble' else 0)

    def update_props(self, force=False):
        """更新右侧面板。
        结构未变化时只刷新动态数值, 避免整面板重建导致的闪烁。
        """
        key = self._get_props_key()
        need_rebuild = (force
                        or key != self._props_key
                        or not self.props_inner.winfo_children())
        if need_rebuild:
            self._props_key = key
            self._rebuild_props()
        self._refresh_props_values()

    def _rebuild_props(self):
        """销毁并重建右侧面板结构 (仅在选中对象/阶段/受力模式变化时调用)"""
        for w in self.props_inner.winfo_children():
            w.destroy()
        self._props_dynamic = []
        self._build_props_structure()

    def _refresh_props_values(self):
        """只刷新动态数值, 不重建任何 widget"""
        for widget, updater in self._props_dynamic:
            try:
                updater(widget)
            except Exception:
                pass

    def _reg_dyn(self, widget, updater):
        """注册动态刷新 widget (updater(widget) 负责用当前数据重写 widget 属性)"""
        self._props_dynamic.append((widget, updater))
        return widget

    def _get_current_force_data(self):
        """返回当前帧当前选中零件的受力缓存 (若无则 None)"""
        if self.stage != 'simulate': return None
        if self.analysis_detail == 'off': return None
        if not self.sim.force_cache: return None
        if not self.sel or self.sel[0] != 'part': return None
        idx = min(self.sim.playhead, len(self.sim.force_cache) - 1)
        cache = self.sim.force_cache[idx]
        return cache.get(self.sel[1].id, None)

    def _build_props_structure(self):
        """构建右侧面板的静态结构 (仅在需要时调用一次)"""
        p = self.props_inner
        color, name = STAGE_COLORS[self.stage]
        tk.Label(p, text=f'◆ {name}', bg='#3c3f41', fg=color,
                 font=('Microsoft YaHei', 10, 'bold')
                 ).pack(anchor='w', pady=(4, 8))
        if self.stage == 'simulate':
            self._props_sim_status(p)
        if self.stage == 'assemble':
            self._props_assemble_dof(p)
            tk.Label(p, text=f'已选中: {len(self.assemble_sel)} 个',
                     bg='#3c3f41', fg='#e69500',
                     font=('Microsoft YaHei', 9, 'bold')
                     ).pack(anchor='w', pady=2)
            for i, item in enumerate(self.assemble_sel):
                if isinstance(item, Part):
                    lbl = f'  {i+1}. 零件 #{item.id} "{item.name}"'
                else:
                    lbl = (f'  {i+1}. 连接附 #{item.id} [{item.type}]'
                           f' @ {item.part.name}')
                tk.Label(p, text=lbl, bg='#3c3f41', fg='#cccccc',
                         anchor='w', font=('Consolas', 9)
                         ).pack(anchor='w')
            tk.Frame(p, height=1, bg='#555555').pack(fill='x', pady=8)
        obj = self.sel
        if obj is None:
            tk.Label(p, text='操作提示:\n'
                             '• 左键点击选中对象\n'
                             '• 空白处左键拖拽 = 平移\n'
                             '• 中键拖拽 = 平移\n'
                             '• 滚轮 = 缩放\n'
                             '• 仿真阶段点零件显示受力',
                     bg='#3c3f41', fg='#999999', justify='left',
                     font=('Microsoft YaHei', 9)).pack(pady=10, anchor='w')
            return
        kind, o = obj
        if kind == 'part':   self._props_part(p, o)
        elif kind == 'point': self._props_point(p, o)
        elif kind == 'link':  self._props_link(p, o)

    def _props_assemble_dof(self, p):
        free_parts = [pp for pp in self.parts if not pp.locked]
        # 只有涉及到至少一个活动零件的转动副才计入有效约束
        active_revolutes = [lk for lk in self.links if lk.type == 'revolute' and not (lk.pA.part.locked and lk.pB.part.locked)]
        
        N_free = len(free_parts)
        J = len(active_revolutes)
        # 二维平面库兹巴赫准则
        dof = 3 * N_free - 2 * J
        
        # 统计设有静态阻力矩（夹紧力）的关节个数 M
        M_active = sum(1 for lk in active_revolutes if getattr(lk, 'static_damping', 0.0) > 1e-9)
        
        f = tk.Frame(p, bg='#3c3f41')
        f.pack(fill='x', pady=4)
        tk.Label(f, text='机构自由度 (DOF): ', bg='#3c3f41', fg='#cccccc', font=('Microsoft YaHei', 9)).pack(side='left')
        col = '#8bc34a' if dof > 0 else ('#ef5350' if dof < 0 else '#ff9800')
        tk.Label(f, text=str(dof), bg='#3c3f41', fg=col, font=('Consolas', 10, 'bold')).pack(side='left')
        
        if M_active > 0:
            tk.Label(p, text=f'带阻力矩转动附数量: {M_active}', bg='#3c3f41', fg='#4a90d9', font=('Microsoft YaHei', 9)).pack(anchor='w', pady=(4,0))
            if M_active > max(0, dof):
                msg = f'⚠ 超静定警告: 当前自由度剩余 {max(0, dof)}，但设置了 {M_active} 个夹紧力矩关节。这会导致未知数超出静力平衡方程组数量，阻力矩分配可能出现多解或不唯一情况，仿真受力分析将使用近似分配。'
                tk.Label(p, text=msg, bg='#3c3f41', fg='#ef5350', wraplength=270, justify='left', font=('Microsoft YaHei', 8)).pack(anchor='w', pady=(2, 6))
        tk.Frame(p, height=1, bg='#555555').pack(fill='x', pady=6)

    def _props_sim_status(self, p):
        sim = self.sim
        # 状态 (动态)
        row = tk.Frame(p, bg='#3c3f41')
        row.pack(anchor='w', fill='x', pady=2)
        tk.Label(row, text='状态: ', bg='#3c3f41', fg='#cccccc',
                 font=('Microsoft YaHei', 9)).pack(side='left')
        state_lbl = tk.Label(row, text='--', bg='#3c3f41', fg='#7cb342',
                             font=('Microsoft YaHei', 9, 'bold'))
        state_lbl.pack(side='left')
        self._reg_dyn(state_lbl, lambda w: w.configure(
            text=self.sim.state,
            fg=('#ef5350' if self.sim.state == 'error' else '#7cb342')))

        # 帧数 / 当前帧 (动态)
        frame_lbl = tk.Label(p, text='--', bg='#3c3f41', fg='#999999',
                             font=('Consolas', 9))
        frame_lbl.pack(anchor='w', pady=2)
        self._reg_dyn(frame_lbl, lambda w: w.configure(
            text=f'帧数: {len(self.sim.frames)}   当前: {self.sim.playhead}'))

        # 受力模式 (静态, 只在 analysis_detail 变化时重建)
        tk.Label(p, text=f'受力: {ANALYSIS_LABELS[self.analysis_detail]}',
                 bg='#3c3f41', fg='#e69500',
                 font=('Microsoft YaHei', 9, 'bold')
                 ).pack(anchor='w', pady=2)
        if sim.last_error:
            tk.Label(p, text=f'⚠ {sim.last_error}', bg='#3c3f41',
                     fg='#ef5350', wraplength=270, justify='left',
                     font=('Microsoft YaHei', 9)).pack(anchor='w', pady=4)
        tk.Frame(p, height=1, bg='#555555').pack(fill='x', pady=6)

    def _props_part(self, p, o):
        tk.Label(p, text=f'零部件 #{o.id}   {o.name}',
                 bg='#3c3f41', fg='#4a90d9',
                 font=('Microsoft YaHei', 10, 'bold')
                 ).pack(anchor='w', pady=(4, 6))
        if self.stage == 'design':
            self._field(p, '名称', o.name, lambda v: self._set_name(o, v))
            self._field(p, '本体质量kg', f'{o.body_mass:.3f}',
                        lambda v: self._set_attr(o, 'body_mass', v))
        tk.Label(p, text=f'总质量: {o.all_mass():.3f} kg',
                 bg='#3c3f41', fg='#999999').pack(anchor='w', pady=2)
        tk.Label(p, text=f'惯量 I: {o.inertia():.3e}',
                 bg='#3c3f41', fg='#999999').pack(anchor='w', pady=2)
                 
        if self.stage != 'simulate':
            tk.Label(p, text=f'连接附: {len(o.points)}  包络顶点: {len(o.hull)}',
                     bg='#3c3f41', fg='#999999').pack(anchor='w', pady=2)
            state_txt = '■ 已锁定 (边界条件)' if o.locked else '○ 自由'
            state_col = '#ef5350' if o.locked else '#8bc34a'
            tk.Label(p, text=state_txt, bg='#3c3f41', fg=state_col,
                     font=('Microsoft YaHei', 9, 'bold')
                     ).pack(anchor='w', pady=6)
            tk.Button(p, text='切换 自由 / 锁定', relief='flat',
                      bg='#455a64', fg='white',
                      command=lambda: self._toggle_lock(o)
                      ).pack(fill='x', pady=2)
            if self.stage == 'design':
                tk.Button(p, text='重新计算包络 (基于连接附)',
                          relief='flat', bg='#455a64', fg='white',
                          command=lambda: (self._rebuild_envelope(o),
                                           self._touch(),
                                           self.update_props(), self.redraw())
                          ).pack(fill='x', pady=2)
            tk.Label(p, text='连接附:', bg='#3c3f41', fg='#cccccc',
                     font=('Microsoft YaHei', 9)
                     ).pack(anchor='w', pady=(10, 2))
            for cp in o.points:
                tag = ' [已连接]' if cp.link else ''
                def _sel(c=cp):
                    # 装配阶段点已连接的连接附 -> 进入其 Link 属性面板
                    if self.stage == 'assemble' and c.link is not None:
                        lk = self._find_link_of(c)
                        if lk is not None:
                            self.sel = ('link', lk)
                        else:
                            self.sel = ('point', c)
                    else:
                        self.sel = ('point', c)
                    self.update_props()
                    self.redraw()
                tk.Button(p, text=f'  #{cp.id}  [{cp.type}]{tag}',
                          relief='flat', bg='#3c3f41', fg='#dddddd',
                          anchor='w', command=_sel).pack(fill='x', pady=1)

        # 仿真阶段的受力数值面板 (结构固定, 数值动态)
        if (self.stage == 'simulate'
                and self.analysis_detail != 'off'
                and self.sim.force_cache
                and o.id in self.sim.force_cache[0]):
            tk.Frame(p, height=20, bg='#3c3f41').pack(fill='x') # 多空出行展示下方受力明细
            self._props_part_forces(p, o)

    @staticmethod
    def _possible_component_kinds_for(cp):
        """根据挂接点类型返回可能的分量种类 (结构固定, 用于详细模式)"""
        if cp.type == 'revolute': return ['revolute']
        if cp.type == 'spring':   return ['spring', 'spring_lim', 'damping']
        if cp.type == 'magnet':   return ['magnet']
        if cp.type == 'payload':  return ['payload']
        return []

    def _props_part_forces(self, p, part_obj):
        """构建受力明细的静态结构, 并把所有数值 label 注册为动态刷新"""
        def get_data():
            return self._get_current_force_data()

        tk.Frame(p, height=1, bg='#555555').pack(fill='x', pady=8)

        # 标题 (帧号动态)
        hrow = tk.Frame(p, bg='#3c3f41')
        hrow.pack(anchor='w', fill='x', pady=(4, 4))
        tk.Label(hrow, text='受力明细', bg='#3c3f41', fg='#7cb342',
                 font=('Microsoft YaHei', 10, 'bold')).pack(side='left')
        fr_lbl = tk.Label(hrow, text='', bg='#3c3f41', fg='#999999',
                          font=('Consolas', 9))
        fr_lbl.pack(side='left', padx=4)
        self._reg_dyn(fr_lbl, lambda w: w.configure(
            text=f'(帧 {self.sim.playhead})'))

        # ---- 零件整体 ----
        head_lbl = tk.Label(p, text='★ 零件整体 (关于质心)',
                            bg='#3c3f41', fg='#8bc34a', anchor='w',
                            font=('Microsoft YaHei', 9, 'bold'))
        head_lbl.pack(anchor='w', pady=(2, 2))

        def upd_head(w):
            d = get_data()
            if d is None: return
            tf = d.get('total_force', np.zeros(2))
            tm = d.get('total_moment', 0.0)
            Fm_N = float(np.linalg.norm(tf)) / PX_PER_M
            M_Nm = float(tm) / (PX_PER_M * PX_PER_M)
            balanced = (Fm_N < 0.01 and abs(M_Nm) < 1e-3)
            w.configure(fg=('#8bc34a' if balanced else '#ef5350'))
        self._reg_dyn(head_lbl, upd_head)

        # 四个数值行 (每个都动态)
        def make_row(prefix, color, fmt):
            row = tk.Frame(p, bg='#3c3f41')
            row.pack(anchor='w', fill='x')
            tk.Label(row, text=prefix, bg='#3c3f41', fg=color,
                     anchor='w', font=('Consolas', 9)).pack(side='left')
            val = tk.Label(row, text='--', bg='#3c3f41', fg=color,
                           anchor='w', font=('Consolas', 9))
            val.pack(side='left')

            def upd(w):
                d = get_data()
                if d is None:
                    w.configure(text='--'); return
                try:
                    w.configure(text=fmt(d))
                except Exception:
                    pass
            self._reg_dyn(val, upd)

        make_row('  总合力: ', FORCE_VIZ['total_f']['color'],
                 lambda d: f'{float(np.linalg.norm(d.get("total_force", np.zeros(2))))/PX_PER_M:.4f} N')
        make_row('  合力 X: ', FORCE_VIZ['total_f']['color'],
                 lambda d: f'{float(d.get("total_force", np.zeros(2))[0])/PX_PER_M:+.4f} N')
        make_row('  合力 Y: ', FORCE_VIZ['total_f']['color'],
                 lambda d: f'{float(d.get("total_force", np.zeros(2))[1])/PX_PER_M:+.4f} N')
        make_row('  总合力矩: ', FORCE_VIZ['total_m']['color'],
                 lambda d: f'{float(d.get("total_moment", 0.0))/(PX_PER_M*PX_PER_M):+.5f} N·m')

        # 详细列举各个附加力矩与直接力矩
        torque_details_lbl = tk.Label(p, text='', bg='#3c3f41', fg='#cccccc',
                                      anchor='w', justify='left', font=('Consolas', 9))
        torque_details_lbl.pack(anchor='w', padx=4, pady=(2, 6))
        
        def upd_torque_details(w):
            d = get_data()
            if d is None:
                w.configure(text=''); return
            lines = []
            for tq in d.get('torques', []):
                val_Nm = float(tq['M']) / (PX_PER_M * PX_PER_M)
                if abs(val_Nm) >= 1e-6:
                    lines.append(f"  ├ {tq['label']}:\n  │    {val_Nm:+.5f} N·m")
            w.configure(text='\n'.join(lines) if lines else "  ├ 无附加力矩")
        self._reg_dyn(torque_details_lbl, upd_torque_details)

        # 平衡状态提示 (动态)
        bal_lbl = tk.Label(p, text='', bg='#3c3f41', fg='#8bc34a',
                           wraplength=270, anchor='w',
                           font=('Microsoft YaHei', 8))
        bal_lbl.pack(anchor='w', pady=(2, 4))

        def upd_bal(w):
            d = get_data()
            if d is None: return
            tf = d.get('total_force', np.zeros(2))
            tm = d.get('total_moment', 0.0)
            Fm_N = float(np.linalg.norm(tf)) / PX_PER_M
            M_Nm = float(tm) / (PX_PER_M * PX_PER_M)
            balanced = (Fm_N < 1e-3 and abs(M_Nm) < 1e-4)
            w.configure(
                text=('  OK 静态平衡 (合力/合力矩约为 0)'
                      if balanced else '  警告: 未平衡 (非稳态静态)'),
                fg=('#8bc34a' if balanced else '#ef5350'))
        self._reg_dyn(bal_lbl, upd_bal)

        tk.Frame(p, height=1, bg='#444444').pack(fill='x', pady=4)

        # ---- 重力 (动态) ----
        grow = tk.Frame(p, bg='#3c3f41')
        grow.pack(anchor='w', fill='x')
        tk.Label(grow, text='  重力: ', bg='#3c3f41',
                 fg=FORCE_VIZ['gravity']['color'],
                 font=('Consolas', 9)).pack(side='left')
        gval = tk.Label(grow, text='--', bg='#3c3f41',
                        fg=FORCE_VIZ['gravity']['color'],
                        font=('Consolas', 9))
        gval.pack(side='left')

        def upd_g(w):
            d = get_data()
            if d is None:
                w.configure(text='--'); return
            Fg = float(np.linalg.norm(d['gravity'])) / PX_PER_M
            w.configure(text=f'{Fg:.3f} N')
        self._reg_dyn(gval, upd_g)

        # ---- 各连接附: 结构按挂接点类型固定 ----
        type_name_map = {
            'revolute': '转动附',
            'spring': '弹簧',
            'magnet': '磁吸',
            'payload': '载荷'
        }
        for cp in part_obj.points:
            type_str = type_name_map.get(cp.type, cp.type)
            tk.Label(p, text=f'连接附 #{cp.id}（{type_str}）',
                     bg='#3c3f41', fg='#e69500',
                     anchor='w', font=('Microsoft YaHei', 9, 'bold')
                     ).pack(anchor='w', pady=(6, 1))

            if self.analysis_detail == 'simple':
                row = tk.Frame(p, bg='#3c3f41')
                row.pack(anchor='w', fill='x')
                tk.Label(row, text='  合力: ', bg='#3c3f41',
                         fg=FORCE_VIZ['total_f']['color'],
                         font=('Consolas', 9)).pack(side='left')
                val = tk.Label(row, text='--', bg='#3c3f41',
                               fg=FORCE_VIZ['total_f']['color'],
                               font=('Consolas', 9))
                val.pack(side='left')

                def upd_simple(w, cid=cp.id):
                    d = get_data()
                    if d is None:
                        w.configure(text='--'); return
                    cpd = d['cp_forces'].get(cid)
                    if cpd is None:
                        w.configure(text='--'); return
                    F = float(np.linalg.norm(cpd['total_f'])) / PX_PER_M
                    w.configure(text=f'{F:.4f} N')
                self._reg_dyn(val, upd_simple)
            else:
                for kind in self._possible_component_kinds_for(cp):
                    cfg = FORCE_VIZ.get(kind, FORCE_VIZ['other'])
                    row = tk.Frame(p, bg='#3c3f41')
                    row.pack(anchor='w', fill='x')
                    tk.Label(row, text=f'  {cfg["label"]}: ',
                             bg='#3c3f41', fg=cfg['color'],
                             font=('Consolas', 9)).pack(side='left')
                    val = tk.Label(row, text='--', bg='#3c3f41',
                                   fg=cfg['color'],
                                   font=('Consolas', 9))
                    val.pack(side='left')

                    def upd_det(w, cid=cp.id, k=kind):
                        d = get_data()
                        if d is None:
                            w.configure(text='--'); return
                        cpd = d['cp_forces'].get(cid)
                        if cpd is None:
                            w.configure(text='--'); return
                        s = np.zeros(2)
                        for kk, F, at in cpd['components']:
                            if kk == k:
                                s = s + F
                        F_N = float(np.linalg.norm(s)) / PX_PER_M
                        w.configure(text=f'{F_N:.4f} N')
                    self._reg_dyn(val, upd_det)

    def _props_point(self, p, cp):
        tk.Label(p, text=f'连接附 #{cp.id}  ({cp.part.name})',
                 bg='#3c3f41', fg='#4a90d9',
                 font=('Microsoft YaHei', 10, 'bold')
                 ).pack(anchor='w', pady=(4, 6))
        if self.stage == 'design':
            tk.Label(p, text='类型', bg='#3c3f41', fg='#cccccc',
                     font=('Microsoft YaHei', 9)).pack(anchor='w')
            var = tk.StringVar(value=cp.type)
            combo = ttk.Combobox(p, textvariable=var, state='readonly',
                                 values=['revolute', 'spring',
                                         'magnet', 'payload'])
            combo.pack(fill='x', pady=2)
            combo.bind('<<ComboboxSelected>>',
                       lambda e: self._change_type(cp, var.get()))
            if cp.type == 'revolute':
                self._field(p, '阻尼', f'{cp.damping:.2f}',
                            lambda v: self._set_attr(cp, 'damping', v))
            elif cp.type == 'spring':
                tk.Label(p, text='弹簧挂接点',
                         bg='#3c3f41', fg='#999999', justify='left',
                         font=('Microsoft YaHei', 9)).pack(anchor='w', pady=4)
            elif cp.type == 'magnet':
                tk.Label(p, text='磁吸挂接点',
                         bg='#3c3f41', fg='#999999', justify='left',
                         font=('Microsoft YaHei', 9)).pack(anchor='w', pady=4)
            elif cp.type == 'payload':
                self._field(p, '质量 kg', f'{cp.mass:.3f}',
                            lambda v: self._set_attr(cp, 'mass', v))
        else:
            # 在装配或仿真阶段，也对载荷类型开放质量修改
            if cp.type == 'payload':
                self._field(p, '载荷质量 kg', f'{cp.mass:.3f}',
                            lambda v: self._set_attr(cp, 'mass', v))
                
        tk.Label(p, text=f'局部坐标: ({cp.local[0]:.1f}, {cp.local[1]:.1f})',
                 bg='#3c3f41', fg='#999999').pack(anchor='w', pady=4)
        if cp.link:
            lk = self._find_link_of(cp)
            if lk is not None:
                tk.Label(p, text=f'已连接 连接 #{lk.id} ({lk.type})',
                         bg='#3c3f41', fg='#8bc34a',
                         font=('Microsoft YaHei', 9, 'bold')
                         ).pack(anchor='w', pady=2)
            else:
                tk.Label(p, text='已配对 (未找到连接对象)',
                         bg='#3c3f41', fg='#ef5350',
                         font=('Microsoft YaHei', 9, 'bold')
                         ).pack(anchor='w', pady=2)
            if self.stage == 'design':
                tk.Button(p, text='断开连接', relief='flat', bg='#7f2d2d',
                          fg='white',
                          command=lambda c=cp: self._unlink_cp(c)
                          ).pack(fill='x', pady=3)
        else:
            tk.Label(p, text='未连接', bg='#3c3f41', fg='#999999'
                     ).pack(anchor='w', pady=2)
        if self.stage == 'design':
            tk.Button(p, text='删除连接附', relief='flat', bg='#7f2d2d',
                      fg='white',
                      command=lambda: self.delete_object('point', cp)
                      ).pack(fill='x', pady=3)

    def _props_link(self, p, lk):
        # 类型安全: 如果误传入 ConnPoint (没有 pA/pB), 回退到连接附面板
        if not hasattr(lk, 'pA') or not hasattr(lk, 'pB'):
            # lk 是 ConnPoint, 尝试找到它所属的 Link
            real_lk = self._find_link_of(lk)
            if real_lk is not None:
                lk = real_lk
            else:
                self._props_point(p, lk)
                return
        if lk.type == 'revolute':
            self._props_revolute_link(p, lk)
            return
        type_name = {'spring': '弹簧', 'magnet': '磁吸连接'}.get(lk.type, lk.type)
        color = {'spring': '#2e7d32', 'magnet': '#7e57c2'}.get(lk.type, '#4a90d9')
        tk.Label(p, text=f'◆ {type_name} #{lk.id}',
                 bg='#3c3f41', fg=color,
                 font=('Microsoft YaHei', 10, 'bold')
                 ).pack(anchor='w', pady=(4, 6))
        tk.Label(p, text=f'两端: #{lk.pA.id} ↔ #{lk.pB.id}',
                 bg='#3c3f41', fg='#999999',
                 font=('Consolas', 9)).pack(anchor='w', pady=2)
        cur_L = float(np.linalg.norm(lk.pB.world() - lk.pA.world()))
        tk.Label(p, text=f'当前长度: {cur_L:.1f} px',
                 bg='#3c3f41', fg='#cccccc',
                 font=('Consolas', 9)).pack(anchor='w', pady=2)
        tk.Frame(p, height=1, bg='#555555').pack(fill='x', pady=6)
        
        def _set_spring_len(lk_obj, prop, val_str):
            try: v = float(val_str)
            except ValueError: return
            if v < 1.0: v = 1.0
            setattr(lk_obj, prop, v)
            
            # 合法性强制联调联动: lmin <= l0 <= lmax
            if prop == 'lmin':
                if lk_obj.l0 < lk_obj.lmin: lk_obj.l0 = lk_obj.lmin
                if lk_obj.lmax < lk_obj.lmin: lk_obj.lmax = lk_obj.lmin
            elif prop == 'l0':
                if lk_obj.lmin > lk_obj.l0: lk_obj.lmin = lk_obj.l0
                if lk_obj.lmax < lk_obj.l0: lk_obj.lmax = lk_obj.l0
            elif prop == 'lmax':
                if lk_obj.l0 > lk_obj.lmax: lk_obj.l0 = lk_obj.lmax
                if lk_obj.lmin > lk_obj.lmax: lk_obj.lmin = lk_obj.lmax
            
            self._touch()
            self.update_props()
            self.redraw()

        if lk.type == 'spring':
            self._field(p, '最短长度', f'{lk.lmin:.1f}',
                        lambda v: _set_spring_len(lk, 'lmin', v))
            self._field(p, '初始长度', f'{lk.l0:.1f}',
                        lambda v: _set_spring_len(lk, 'l0', v))
            self._field(p, '最大长度', f'{lk.lmax:.1f}',
                        lambda v: _set_spring_len(lk, 'lmax', v))
            self._field(p, '刚度 k', f'{lk.k:.1f}',
                        lambda v: self._set_attr(lk, 'k', v))
            self._field(p, '阻尼 c', f'{lk.c:.2f}',
                        lambda v: self._set_attr(lk, 'c', v))
            F_now = lk.k * (cur_L - lk.l0) / PX_PER_M
            tk.Label(p, text=f'当前弹簧力: {F_now:+.3f} N',
                     bg='#3c3f41', fg='#a5d6a7',
                     font=('Microsoft YaHei', 9, 'bold')
                     ).pack(anchor='w', pady=6)
            tk.Button(p, text='以当前长度作为自然长度',
                      relief='flat', bg='#455a64', fg='white',
                      command=lambda: (self._set_attr(lk, 'l0', f'{cur_L:.1f}'),
                                       self.update_props())
                      ).pack(fill='x', pady=2)
        elif lk.type == 'magnet':
            self._field(p, '自然长度', f'{lk.l0:.1f}',
                        lambda v: self._set_attr(lk, 'l0', v))
            self._field(p, '刚度 k', f'{lk.mk:.1f}',
                        lambda v: self._set_attr(lk, 'mk', v))
            self._field(p, '强度', f'{lk.strength:.0f}',
                        lambda v: self._set_attr(lk, 'strength', v))
            st = '已脱开' if lk.broken else '吸合中'
            col = '#ef5350' if lk.broken else '#8bc34a'
            tk.Label(p, text=f'状态: {st}', bg='#3c3f41', fg=col,
                     font=('Microsoft YaHei', 10, 'bold')
                     ).pack(anchor='w', pady=6)
            if lk.broken:
                tk.Button(p, text='重新吸合', relief='flat', bg='#455a64',
                          fg='white',
                          command=lambda: (setattr(lk, 'broken', False),
                                           self._touch(),
                                           self.update_props(), self.redraw())
                          ).pack(fill='x', pady=3)
        tk.Button(p, text='删除该连接', relief='flat', bg='#7f2d2d',
                  fg='white',
                  command=lambda: (self._unlink_by_link(lk),
                                   self._touch(),
                                   self.clear_selection())
                  ).pack(fill='x', pady=6)

    def _props_revolute_link(self, p, lk):
        """转动副属性面板 (装配界面)"""
        tk.Label(p, text=f'◆ 转动副 #{lk.id}',
                 bg='#3c3f41', fg='#4a90d9',
                 font=('Microsoft YaHei', 10, 'bold')
                 ).pack(anchor='w', pady=(4, 6))
        tk.Label(p, text=f'{lk.pA.part.name} ↔ {lk.pB.part.name}',
                 bg='#3c3f41', fg='#999999',
                 font=('Microsoft YaHei', 9)).pack(anchor='w', pady=2)
        wa = lk.pA.world(); wb = lk.pB.world()
        dev = float(np.linalg.norm(wa - wb))
        tk.Label(p, text=f'两端偏差: {dev:.3f} px',
                 bg='#3c3f41', fg='#999999',
                 font=('Consolas', 9)).pack(anchor='w', pady=2)

        # 当前帧的铰链力矩 (静力平衡解 + 物理钳制)
        M_max = lk.static_damping
        M_provide_Nm = lk.moment / (PX_PER_M * PX_PER_M)
        M_needed_Nm  = getattr(lk, 'moment_raw', lk.moment) / (PX_PER_M * PX_PER_M)

        if M_max > 1e-9:
            if lk.friction_saturated:
                sat_txt = (f'⚠ 已进入滑动摩擦\n'
                           f'   需要力矩: {M_needed_Nm:+.5f} N·m\n'
                           f'   提供力矩: {M_provide_Nm:+.5f} N·m\n'
                           f'   (已达上限 ±{M_max:.4f} N·m)')
                sat_col = '#ef5350'
            else:
                sat_txt = (f'✓ 静摩擦 (提供所需力矩)\n'
                           f'   需要力矩: {M_needed_Nm:+.5f} N·m\n'
                           f'   提供力矩: {M_provide_Nm:+.5f} N·m\n'
                           f'   (≤ ±{M_max:.4f} N·m)')
                sat_col = '#8bc34a'
        else:
            sat_txt = f'当前铰链力矩: {M_provide_Nm:+.6f} N·m (无夹紧装置)'
            sat_col = '#999999'
        tk.Label(p, text=sat_txt, bg='#3c3f41', fg=sat_col,
                 wraplength=270, justify='left',
                 font=('Microsoft YaHei', 9)).pack(anchor='w', pady=2)

        tk.Frame(p, height=1, bg='#555555').pack(fill='x', pady=6)

        tk.Label(p, text='静态摩擦极限力矩 (N·m)',
                 bg='#3c3f41', fg='#cccccc',
                 font=('Microsoft YaHei', 9)).pack(anchor='w')
        var = tk.StringVar(value=f'{lk.static_damping:.4f}')
        entry = tk.Entry(p, textvariable=var, bg='#2b2b2b', fg='#eeeeee',
                         insertbackground='white', relief='flat')
        entry.pack(fill='x', pady=2)

        def _apply_live(*args):
            try:
                v = float(var.get())
            except ValueError:
                return
            if v < 0: return
            if abs(v - lk.static_damping) < 1e-12: return
            lk.static_damping = v
            logger.info(f'[revolute] link#{lk.id} static_damping = {v:.4f} '
                        f'(实时写入)')
            self.set_status(
                f'转动副 #{lk.id} 静态阻尼力矩 = {v:.4f} N·m (已写入)')
            self._touch()
            
        entry.bind('<FocusOut>', _apply_live)
        entry.bind('<Return>',   _apply_live)
        
        # 对单独构建的输入框添加实时变动监听
        var.trace_add('write', _apply_live)
        tk.Label(p, text='模拟铰链处的夹紧装置: 铰链最多提供 ±M_max 的\n'
                         '力矩抵抗转动。需要更大时进入滑动摩擦。\n'
                         '(0 = 无夹紧, 铰链自由转动)',
                 bg='#3c3f41', fg='#888888', justify='left',
                 font=('Microsoft YaHei', 8)).pack(anchor='w', pady=(2, 6))

        def on_confirm():
            try:
                v = float(var.get())
            except ValueError:
                messagebox.showerror('输入错误', '请输入有效的数值')
                return
            if v < 0:
                messagebox.showerror('输入错误', '不能为负值')
                return
            lk.static_damping = v
            self._touch()
            self.set_status(f'转动副 #{lk.id} 静态阻尼力矩 = {v:.4f} N·m')
            logger.info(f'[revolute] link#{lk.id} static_damping = {v:.4f}')

        tk.Button(p, text='确定', relief='flat', bg='#2e7d32', fg='white',
                  command=on_confirm).pack(fill='x', pady=3)
        tk.Button(p, text='断开该转动副', relief='flat', bg='#7f2d2d',
                  fg='white',
                  command=lambda: (self._unlink_by_link(lk),
                                   self._touch(),
                                   self.clear_selection())
                  ).pack(fill='x', pady=3)

    def _set_name(self, o, v):
        o.name = v
        self._touch()

    def _set_attr(self, o, name, val):
        try: v = float(val)
        except ValueError: return
        setattr(o, name, v)
        self._touch()
        self.redraw()

    def _toggle_lock(self, p):
        p.locked = not p.locked
        p.v[:] = 0; p.omega = 0
        self._touch()
        self.update_props(); self.redraw()

    def _change_type(self, cp, new_type):
        if cp.link is not None:
            self._unlink_cp(cp)
        cp.type = new_type
        self._rebuild_envelope(cp.part)
        self._touch()
        self.update_props(); self.redraw()

    def _unlink_cp(self, cp):
        lk = None
        for l in list(self.links):
            if l.pA is cp or l.pB is cp:
                lk = l; break
        if lk: self._unlink_by_link(lk)
        self._touch()
        self.update_props(); self.redraw()

    # ---------------- JSON 保存 / 加载 ----------------
    def save_to_json(self):
        path = filedialog.asksaveasfilename(
            defaultextension='.json',
            filetypes=[('JSON', '*.json'), ('All', '*.*')],
            title='保存方案')
        if not path: return
        data = {
            'version': 1,
            'parts': [],
            'links': [],
            'collisions': [],
        }
        for p in self.parts:
            pd = {
                'id': p.id,
                'name': p.name,
                'hull': [[float(v[0]), float(v[1])] for v in p.hull],
                'locked': bool(p.locked),
                'body_mass': float(p.body_mass),
                'pos': [float(p.pos[0]), float(p.pos[1])],
                'angle': float(p.angle),
                'points': [],
            }
            for cp in p.points:
                pd['points'].append({
                    'id': cp.id,
                    'local': [float(cp.local[0]), float(cp.local[1])],
                    'type': cp.type,
                    'damping': float(cp.damping),
                    'mass': float(cp.mass),
                    'static_damping': float(getattr(cp, 'static_damping', 0.0)),
                })
            data['parts'].append(pd)
        for lk in self.links:
            data['links'].append({
                'id': lk.id,
                'pA_part_id': lk.pA.part.id,
                'pA_id': lk.pA.id,
                'pB_part_id': lk.pB.part.id,
                'pB_id': lk.pB.id,
                'type': lk.type,
                'lmin': float(lk.lmin),
                'l0': float(lk.l0),
                'lmax': float(lk.lmax),
                'k': float(lk.k),
                'c': float(lk.c),
                'mk': float(lk.mk),
                'strength': float(lk.strength),
                'broken': bool(lk.broken),
                'static_damping': float(getattr(lk, 'static_damping', 0.0)),
            })
        for (pA, pB) in self.collisions:
            data['collisions'].append([pA.id, pB.id])
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.set_status(f'已保存到 {path}')
        except Exception as ex:
            messagebox.showerror('保存失败', str(ex))

    def load_from_json(self):
        path = filedialog.askopenfilename(
            filetypes=[('JSON', '*.json'), ('All', '*.*')],
            title='加载方案')
        if not path: return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as ex:
            messagebox.showerror('加载失败', str(ex))
            return

        self.parts.clear(); self.links.clear(); self.collisions.clear()
        self.sel = None
        self.assemble_sel.clear()
        self.dragging = None
        self.drag_grab_local = None

        id_to_part = {}
        id_to_cp = {}

        for pd in data.get('parts', []):
            p = Part(pd['pos'], pd['hull'], pd['name'])
            p.id = pd['id']
            p.locked = bool(pd.get('locked', False))
            p.body_mass = float(pd.get('body_mass', 0.2))
            p.angle = float(pd.get('angle', 0.0))
            p.pred_pos = p.pos.copy()
            p.pred_angle = p.angle
            for cpd in pd.get('points', []):
                cp = ConnPoint(p, cpd['local'], cpd.get('type', 'revolute'))
                cp.id = cpd['id']
                cp.damping = float(cpd.get('damping', 2.0))
                cp.mass = float(cpd.get('mass', 0.0))
                cp.static_damping = float(cpd.get('static_damping', 0.0))
                p.points.append(cp)
                id_to_cp[(pd['id'], cpd['id'])] = cp
            self.parts.append(p)
            id_to_part[pd['id']] = p

        Part._ptid = max([p.id for p in self.parts] + [0])
        ConnPoint._pid = max([cp.id for p in self.parts for cp in p.points] + [0])

        for ld in data.get('links', []):
            keyA = (ld['pA_part_id'], ld['pA_id'])
            keyB = (ld['pB_part_id'], ld['pB_id'])
            pa = id_to_cp.get(keyA)
            pb = id_to_cp.get(keyB)
            if pa is None or pb is None: continue
            lk = Link(pa, pb, ld['type'])
            lk.id = ld['id']
            lk.l0    = float(ld.get('l0', 100.0))
            lk.lmax  = float(ld.get('lmax', 200.0))
            lk.lmin  = float(ld.get('lmin', lk.l0 * 0.5))
            lk.k     = float(ld.get('k', 100.0))
            lk.c     = float(ld.get('c', 5.0))
            lk.mk    = float(ld.get('mk', 500.0))
            lk.strength = float(ld.get('strength', 3000.0))
            lk.broken   = bool(ld.get('broken', False))
            lk.static_damping = float(ld.get('static_damping', 0.0))
            self.links.append(lk)
        Link._lid = max([l.id for l in self.links] + [0])

        # 恢复碰撞对
        for item in data.get('collisions', []):
            try:
                a_id, b_id = int(item[0]), int(item[1])
            except Exception:
                continue
            pA = id_to_part.get(a_id)
            pB = id_to_part.get(b_id)
            if pA is not None and pB is not None and pA is not pB:
                self.collisions.append((pA, pB))

        self._touch()
        self.fit_view()
        self.update_props()
        self.redraw()
        
        # 记录模型名，不再立刻创建日志，留到开始仿真时使用
        self.model_name = os.path.splitext(os.path.basename(path))[0]
        
        global _LOG_FILE_HANDLER
        if _LOG_FILE_HANDLER is not None:
            try:
                logger.removeHandler(_LOG_FILE_HANDLER)
                _LOG_FILE_HANDLER.close()
            except Exception:
                pass
            _LOG_FILE_HANDLER = None
            
        self.set_status(f'已加载 {path}')


# ====================== 入口 ======================
if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.geometry('1440x900') # 适当增大工具窗口尺寸
    root.mainloop()
# -*- coding: utf-8 -*-
r"""难度二：初始居中（PART 3「照片 + 纯白混合」+ 5×5）

开局 30 块堆在作答区里，先全部搬到棋盘左右两侧，再按难度一的流程拼。

热键：F8 开始 / 停止
用法：python demo_solve_last2.py | --image 截图.png | --plan 截图.png
依赖：D:\repo\models\best.pt
"""

import ctypes
import itertools
import os
import sys
import threading
import time

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ============================== 常量 ==============================
# 路径一律写绝对路径（本文件以 D:\repo 为固定安装位置）
MODEL = r"D:\repo\models\best.pt"
OUT_DIR = r"D:\repo\out"

# YOLO
IMGSZ, CONF = 1280, 0.20

# 颜色 / 尺寸（2560×1600 全屏、150% 缩放下实测）
BOARD_BGR = np.array([42, 23, 15], np.int16)     # 作答区底色 #0f172a
TRAY_BGR = np.array([17, 9, 6], np.int16)        # 托盘底色
WHITE_PIX = 250                                   # "纯白"的最低通道值

PIECE_W, PIECE_H = 146.0, 109.0    # 块面基准尺寸（只作回退用，实际从板区现算）
CELL_W, CELL_H = 114.0, 86.0       # 参考格基准尺寸（同上）
BOARD_FIXED = (607, 648, 730, 547)  # 板区回退常量（标定失败时用）

# 白格判据：纯白占比 ≥0.98 且灰度 std <2
PURE_MIN, STD_MAX = 0.98, 2.0

# 鼠标节奏（越小越快，太小浏览器会吞事件）
T_HOLD = 0.025        # 按下后停一下，浏览器才认这次拖拽
T_SETTLE = 0.05       # 松手后等界面刷新，再截图
T_BEFORE_UP = 0.025   # 松手前停一下

# ---- 热键（全局）----
VK_START = 0x77          # F8：开始一局；运行中再按一次 = 立刻停下

# ---- 搬运阶段（难度二特有）----
STAGE_INSET = 25        # 只看作答区往里缩 25 px 的部分（外面停的块 tab 会伸进来）
STAGE_EXIT_RATIO = 0.02  # 碎片占比低于这个就算搬空
STAGE_MAX_MOVES = 34     # 保险：30 块 + 4 次容错
# 槽距 = 块尺寸 + 2×容差 + 余量，保证容差内不会压到隔壁
SLOT_PITCH_X = 184.0     # 块宽 146 + 2×12 + 12 余量
SLOT_PITCH_Y = 145.0     # 块高 109 + 2×12 + 12 余量
STAGE_TOP = 400.0        # 落点最高能到哪（再往上块落歪会砸到工具条）
CORRECT_TOL = 12.0       # 落下后偏离槽心超过这么多像素，就再拖一次摆正

SIDES = ("top", "right", "bottom", "left")

# 判白块/白格形状时用的几何（按 146×109 的块面调出来的）
GAP = 4      # 条带离块面多远
THK = 12     # 条带多厚（凸起约 15 px，取 12 够用）
MID = 0.40   # 只取边长的中间 40%，避开四个角

# 三种边的"参考值"：(外侧白占比, 内侧白占比)
EXPECT = {
    "tab": (1.0, 1.0),
    "notch": (0.0, 0.0),
    "flat": (0.0, 1.0),
}

FLAT_W = 2.5     # 平边的权重：盘面边界上的"必须平边"是绝对可靠的
FLAT_PEN = 0.8   # 缺一条必需平边的额外惩罚


def log(msg):
    print(msg, flush=True)


def stopping(abort):
    """按过 F8 没有。"""
    return abort is not None and abort.is_set()


# ============================== 一、基础工具 ==============================
def dpi_aware():
    """让截图像素和鼠标坐标一致（必须最先调用）。"""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def grab():
    """截一张全屏图（BGR）。"""
    import mss
    with mss.mss() as s:
        shot = np.array(s.grab(s.monitors[1]))
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


def patch_diff(a, b, p, r=35):
    """两帧在同一小块上的平均像素差。"""
    x, y = int(p[0]), int(p[1])
    x0, x1 = max(0, x - r), min(a.shape[1], x + r)
    y0, y1 = max(0, y - r), min(a.shape[0], y + r)
    pa = a[y0:y1, x0:x1].astype(np.int16)
    pb = b[y0:y1, x0:x1].astype(np.int16)
    if pa.size == 0 or pa.shape != pb.shape:
        return 999.0
    return float(np.abs(pa - pb).mean())


# ============================== 二、界面元素 ==============================
def find_start_button(img):
    """找「开始这一部分」按钮 → (中心点, 框)；没有返回 None。"""
    b, g, r = cv2.split(img.astype(np.int16))
    mask = ((b > 150) & (r > 60) & (r < 150) & (g < 120) & ((b - g) > 80)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
    best, box = 0, None
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 20000 or h > 120 or w < 200:      # 按钮是"宽而扁"的实心块
            continue
        if a > best:
            best, box = a, (int(x), int(y), int(w), int(h))
    if box is None:
        return None
    return (box[0] + box[2] // 2, box[1] + box[3] // 2), box


def find_board(img):
    """找作答区矩形 → (x, y, w, h)；找不到返回 None。"""
    rect = _board_by_color(img, 12)
    if rect is not None:
        return rect
    if _board_corners_ok(img, BOARD_FIXED):
        return BOARD_FIXED
    return _board_by_color(img, 30)


def _board_by_color(img, tol):
    """按板底色的连通域找 4:3 矩形 → 外接框。"""
    d = np.abs(img.astype(np.int16) - BOARD_BGR).sum(2)
    m = (d < tol).astype(np.uint8)
    h, w = m.shape[:2]
    m[:150, :] = 0                       # 顶部 UI
    m[:, int(w * 0.74):] = 0             # 右侧参考栏
    closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(closed, 8)
    best = None
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if a < 80000:
            continue
        ar = bw / float(bh)
        if 1.15 < ar < 1.55 and 500 < bw < 1200 and 350 < bh < 800:
            if best is None or a > best[4]:
                best = (i, x, y, bw, bh)
    if best is None:
        return None
    # 外接矩形要回到"原始底色像素"上算：闭运算会把边界撑大十几像素
    idx = best[0]
    ys, xs = np.where((m > 0) & (lab == idx))
    if len(xs) < 1000:
        return None
    return (int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def _board_corners_ok(img, rect, inset=6, k=6):
    """四个角都不是托盘底色，就认为这个矩形是作答区。"""
    x, y, w, h = [int(v) for v in rect]
    H, W = img.shape[:2]
    for (cx, cy) in ((x + inset, y + inset), (x + w - inset - k, y + inset),
                     (x + inset, y + h - inset - k), (x + w - inset - k, y + h - inset - k)):
        if cx < 0 or cy < 0 or cx + k > W or cy + k > H:
            return False
        patch = img[cy:cy + k, cx:cx + k].astype(np.int16)
        if float(np.abs(patch - TRAY_BGR).sum(2).mean()) < 40:
            return False
    return True


def find_reference_grid(img, board):
    """找右侧参考图的 5×5 网格 → (x0, y0, cw, ch)；没有返回 None。"""
    bx, by, bw, bh = board
    H, W = img.shape[:2]
    lw, lh = bw / 5.0, bh / 5.0
    cw0, ch0 = lw * 0.781, lh * 0.781

    white = ((img[:, :, 0] >= WHITE_PIX) & (img[:, :, 1] >= WHITE_PIX)
             & (img[:, :, 2] >= WHITE_PIX)).astype(np.uint8)
    white[:150, :] = 0
    white[:, :int(bx + bw + 20)] = 0
    n, lab, st, cen = cv2.connectedComponentsWithStats(white, 8)
    cells = []
    for i in range(1, n):
        x, y, cw_, ch_, a = st[i]
        if not (0.85 * cw0 <= cw_ <= 1.15 * cw0 and 0.85 * ch0 <= ch_ <= 1.15 * ch0):
            continue
        if a < 0.80 * cw_ * ch_:
            continue
        cells.append((float(x), float(y), float(cw_), float(ch_)))
    if len(cells) < 2:
        return None

    cw = float(np.median([c[2] for c in cells]))
    ch = float(np.median([c[3] for c in cells]))
    rx = float(np.median([(c[0] + c[2] / 2.0) % cw for c in cells]))
    ry = float(np.median([(c[1] + c[3] / 2.0) % ch for c in cells]))

    base_x = rx - cw / 2.0
    base_y = ry - ch / 2.0
    gx0, gy0 = bx + 1350.0, by + 80.0          # 已知布局下的位置，只当"挑候选"的参照

    best = None
    k0 = int(np.ceil((bx + bw + 20 - base_x) / cw))
    for m in range(k0, k0 + 12):
        x0 = base_x + m * cw
        if x0 < bx + bw + 20 or x0 + 5 * cw > W:
            continue
        r0 = int(np.ceil((150 - base_y) / ch))
        for q in range(r0, r0 + 14):
            y0 = base_y + q * ch
            if y0 < 150 or y0 + 5 * ch > H:
                continue
            d = abs(x0 - gx0) + abs(y0 - gy0)
            if best is None or d < best[0]:
                best = (d, x0, y0)
    if best is None:
        return None
    return best[1], best[2], cw, ch


def calibrate(img, verbose=False):
    """量出作答区、参考网格、块面尺寸 → dict。"""
    board = find_board(img)
    src = "检测"
    if board is None:
        board, src = BOARD_FIXED, "回退常量"
        if verbose:
            log("[标定] 没找到作答区，用回退常量 %s" % (BOARD_FIXED,))
    bx, by, bw, bh = board
    lw, lh = bw / 5.0, bh / 5.0

    grid = find_reference_grid(img, board)
    if grid is None:
        grid = (bx + 1350.0, by + 80.0, float(CELL_W), float(CELL_H))
        src += "+网格回退"
        if verbose:
            log("[标定] 参考网格没找到，用板区偏移兜底")
    # 参考格和块面是同一个比例（114/146 = 0.78），尺寸差太多说明读错了
    if abs(grid[2] / lw - 0.781) > 0.08 or abs(grid[3] / lh - 0.781) > 0.08:
        if verbose:
            log("[标定] 参考网格尺寸异常（%.1f×%.1f vs 块 %.1f×%.1f），改用偏移兜底"
                % (grid[2], grid[3], lw, lh))
        grid = (bx + 1350.0, by + 80.0, float(CELL_W), float(CELL_H))
        src += "+网格回退"

    if verbose:
        log("[标定] 作答区 %s（%s），块面 %.1f×%.1f，参考网格 (%.0f,%.0f) 格 %.1f×%.1f"
            % (tuple(int(v) for v in board), src, lw, lh,
               grid[0], grid[1], grid[2], grid[3]))
    return {"board": tuple(float(v) for v in board), "grid": grid,
            "lw": float(lw), "lh": float(lh), "src": src}


# ============================== 三、识别 ==============================
def detect(model, img):
    """YOLO 推理一次 → [{box, label, conf}]。"""
    r = model.predict(img, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    out = []
    for i in range(len(r.boxes)):
        x1, y1, x2, y2 = [float(v) for v in r.boxes.xyxy[i].tolist()]
        out.append({"box": (x1, y1, x2, y2),
                    "label": model.names[int(r.boxes.cls[i])],
                    "conf": float(r.boxes.conf[i])})
    return out


def split_boxes(boxes, lw, lh, cw, ch):
    """把 YOLO 框分成拼图块和参考格两类。"""
    pieces, cells = [], []
    for b in boxes:
        w = b["box"][2] - b["box"][0]
        h = b["box"][3] - b["box"][1]
        d_piece = abs(w - lw) / lw + abs(h - lh) / lh
        d_cell = abs(w - cw) / cw + abs(h - ch) / ch
        if min(d_piece, d_cell) > 0.20:      # 尺寸不伦不类，丢弃
            continue
        (pieces if d_piece <= d_cell else cells).append(b)
    return pieces, cells


def class_grid(cells, grid):
    """参考格的类别 → {(r,c): 类别}。"""
    gx, gy, cw, ch = grid
    out = {}
    for b in cells:
        cx = (b["box"][0] + b["box"][2]) / 2
        cy = (b["box"][1] + b["box"][3]) / 2
        c = int(round((cx - gx) / cw - 0.5))
        r = int(round((cy - gy) / ch - 0.5))
        if 0 <= r < 5 and 0 <= c < 5:
            if (r, c) not in out or b["conf"] > out[(r, c)][1]:
                out[(r, c)] = (b["label"], b["conf"])
    return {k: v[0] for k, v in out.items()}


def white_cells_of(img, grid):
    """参考网格里的纯白格 → [(r,c)]。"""
    gx, gy, cw, ch = grid
    out = []
    for r in range(5):
        for c in range(5):
            x0 = int(round(gx + c * cw)) + 3
            y0 = int(round(gy + r * ch)) + 3
            cell = img[y0:y0 + int(ch) - 6, x0:x0 + int(cw) - 6]
            if cell.size == 0:
                continue
            pure = (cell >= WHITE_PIX).all(axis=2).mean()
            std = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY).std()
            if pure >= PURE_MIN and std < STD_MAX:
                out.append((r, c))
    return out


def looks_white(img, cx, cy, lw, lh):
    """这个位置是不是一块纯白块。"""
    f = img[int(cy - lh / 2):int(cy + lh / 2), int(cx - lw / 2):int(cx + lw / 2)]
    if f.size == 0:
        return False
    pure = float((f >= WHITE_PIX).all(axis=2).mean())
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    colored = float(((hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 70)).mean())
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[12:-12, 12:-12]   # 内缩，避开面边框
    if g.size == 0:
        return False
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    edge = float(np.mean(cv2.magnitude(gx, gy)))
    return pure >= 0.75 and colored < 0.05 and edge < 40.0


def white_pieces_of(img, lw, lh, board, grid, expect=5, photo_boxes=None):
    """找托盘里的白块 → 每块的面中心 [(x,y)]。"""
    bx, by, bw, bh = board
    gx = grid[0]
    white = ((img[:, :, 0] >= WHITE_PIX) & (img[:, :, 1] >= WHITE_PIX)
             & (img[:, :, 2] >= WHITE_PIX)).astype(np.uint8)
    white[:int(by * 0.45), :] = 0                        # 顶部 UI
    white[:, int(gx - 40):] = 0                          # 参考栏
    white[int(by) - 6:int(by + bh) + 6,
          int(bx) - 6:int(bx + bw) + 6] = 0              # 作答区本身

    k = max(9, int(round(lw * 0.14)) | 1)
    er = cv2.erode(white, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(er, 8)
    comps = sorted([(float(st[i, 4]), float(cen[i][0]), float(cen[i][1]))
                    for i in range(1, n)], reverse=True)

    # 不够 expect 个就逐级放宽面积门槛（正常局不触发）
    min_area = 5000.0
    cand = []
    while True:
        cand = [c for c in comps if c[0] >= min_area]
        if len(cand) >= expect or min_area <= 2500.0:
            break
        min_area = max(2500.0, min_area * 0.75)
    if min_area < 5000.0:
        log("[白块] 面积门槛放宽到 %.0f 才凑够 %d 个" % (min_area, expect))

    def on_photo(x, y):
        for (x1, y1, x2, y2) in (photo_boxes or []):
            if abs(x - (x1 + x2) / 2) < lw * 0.5 and abs(y - (y1 + y2) / 2) < lh * 0.5:
                return True
        return False

    picked, dropped, dropped_photo = [], 0, []
    for sc, x, y in cand:
        if any((x - px) ** 2 + (y - py) ** 2 <= (lw * 0.6) ** 2 for _, px, py in picked):
            continue
        if not looks_white(img, x, y, lw, lh):
            dropped += 1
            continue
        if on_photo(x, y):          # YOLO 说那儿有照片块 → 硬判不是白块
            dropped_photo.append((x, y))
            continue
        picked.append((sc, x, y))
        if len(picked) >= expect:
            break
    if dropped:
        log("[白块] 纹理判据丢弃 %d 个候选" % dropped)
    if dropped_photo:
        log("[白块] YOLO 交叉校验剔除 %d 个非白块" % len(dropped_photo))
    if len(picked) < expect:
        log("[白块] 只找到 %d 个（期望 %d），可能被压住太多" % (len(picked), expect))
    return [(x, y) for _, x, y in picked]


# ============================== 四、搬运（难度二特有）==============================
def board_content_mask(img, cal, inset=STAGE_INSET):
    """作答区里「不是底色」的像素掩膜 → (mask, 左上角)。"""
    bx, by, bw, bh = cal["board"]
    x0, y0 = int(bx + inset), int(by + inset)
    x1, y1 = int(bx + bw - inset), int(by + bh - inset)
    sub = img[y0:y1, x0:x1]
    if sub.size == 0:
        return np.zeros((1, 1), np.uint8), (x0, y0)
    d = np.abs(sub.astype(np.int16) - BOARD_BGR).sum(2)
    return (d > 30).astype(np.uint8), (x0, y0)


def board_piece_ratio(img, cal, inset=STAGE_INSET):
    """作答区被碎片盖住的面积占比。"""
    m, _ = board_content_mask(img, cal, inset)
    return float(m.mean()) if m.size else 0.0


def deepest_point(img, cal, avoid=None, inset=STAGE_INSET):
    """在作答区里找一个离空地最远的点，当抓取点。"""
    m, (x0, y0) = board_content_mask(img, cal, inset)
    if m.size == 0 or float(m.mean()) < 0.01:
        return None
    if avoid:
        for (ax, ay) in avoid:
            cv2.circle(m, (int(ax - x0), int(ay - y0)), 34, 0, -1)
        if float(m.mean()) < 0.005:
            return None
    # 四周补一圈 0 再算距离：否则贴着 ROI 边界的点会因为"外面没算"而拿到假的高值
    mp = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(mp, cv2.DIST_L2, 5)
    _, mx, _, loc = cv2.minMaxLoc(dist)
    if mx < 12:                 # 最深处都不到 12 px → 没有像样的块
        return None
    return (x0 + loc[0] - 1, y0 + loc[1] - 1)


def stage_slots(cal, size=(2560, 1600)):
    """生成搬运落点（左右两带，按行出点）。"""
    bx, by, bw, bh = cal["board"]
    lw, lh = cal["lw"], cal["lh"]
    W, H = size

    xs_left = [bx - 8 - lw / 2 - i * SLOT_PITCH_X for i in range(3)]
    xs_right = [bx + bw + 8 + lw / 2 + i * SLOT_PITCH_X for i in range(3)]
    xs_left = [min(max(x, lw / 2 + 6), W - lw / 2 - 6) for x in xs_left]
    xs_right = [min(max(x, lw / 2 + 6), W - lw / 2 - 6) for x in xs_right]

    rows = int((H - 20 - STAGE_TOP) // SLOT_PITCH_Y) + 1
    ys = [STAGE_TOP + lh / 2 + j * SLOT_PITCH_Y for j in range(rows)]
    ys = [y for y in ys if y + lh / 2 < H - 6]

    order = [xs_left[0], xs_right[0], xs_left[1], xs_right[1],
             xs_left[2], xs_right[2]]
    out = []
    for y in ys:
        for x in order:
            out.append((float(x), float(y)))
    return out


def landed_piece(before, after, dst, lw, lh):
    """量出刚拖过去那块落在哪 → (x1,y1,x2,y2)；没搬动返回 None。"""
    pad_x, pad_y = int(lw * 1.25), int(lh * 1.35)
    x0 = max(0, int(dst[0]) - pad_x)
    x1 = min(after.shape[1], int(dst[0]) + pad_x)
    y0 = max(0, int(dst[1]) - pad_y)
    y1 = min(after.shape[0], int(dst[1]) + pad_y)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    a = before[y0:y1, x0:x1].astype(np.int16)
    b = after[y0:y1, x0:x1].astype(np.int16)
    # 只留「变了 + 现在是块」的像素，避免把腾出来的空地也算进来
    changed = np.abs(a - b).sum(2) > 40
    is_piece = np.abs(b - TRAY_BGR).sum(2) > 30
    m = (changed & is_piece).astype(np.uint8)
    if int(m.sum()) < 1500:                 # 几乎没有变化 → 这块没动
        return None
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    best = max(range(1, n), key=lambda i: st[i, 4])
    bx, by, bw, bh = st[best][:4]
    if bw < lw * 0.5 or bh < lh * 0.5:      # 太小的变化是噪点
        return None
    return (x0 + int(bx), y0 + int(by), x0 + int(bx) + int(bw), y0 + int(by) + int(bh))


def _overlap(box, boxes, margin=4):
    """这个框和已停好的块有没有重叠。"""
    x1, y1, x2, y2 = box
    for (a1, b1, a2, b2) in boxes:
        if x2 - margin <= a1 or a2 - margin <= x1 or y2 - margin <= b1 or b2 - margin <= y1:
            continue
        return True
    return False


def clear_board(pyautogui, cal, abort=None, max_moves=STAGE_MAX_MOVES):
    """把作答区搬空：搬一块 → 量位置 → 必要时摆正，直到空。"""
    img = grab()
    lw, lh = cal["lw"], cal["lh"]
    slots = stage_slots(cal, (img.shape[1], img.shape[0]))
    si, moved, fixed, bad = 0, 0, 0, []
    parked = []
    log("开始搬运：作答区碎片占比 %.2f" % board_piece_ratio(img, cal))

    while moved < max_moves:
        if stopping(abort):
            log("  F8 中断搬运")
            break
        ratio = board_piece_ratio(img, cal)
        if ratio <= STAGE_EXIT_RATIO:
            break
        p = deepest_point(img, cal, avoid=bad)
        if p is None:
            log("  找不到可抓的点，停止搬运（占比 %.2f）" % ratio)
            break
        if si >= len(slots):
            log("  落点用完了，停止搬运")
            break

        dst = slots[si]
        drag(pyautogui, p, dst)
        time.sleep(T_SETTLE)
        img2 = grab()
        box = landed_piece(img, img2, dst, lw, lh)
        if box is None:
            bad.append(p)
            img = img2
            log("  这一下没搬动 (%d,%d)，换个抓取点再试" % (p[0], p[1]))
            continue

        # 落歪了就再拖一次，把它摆到槽心
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        off = ((cx - dst[0]) ** 2 + (cy - dst[1]) ** 2) ** 0.5
        if off > CORRECT_TOL:
            drag(pyautogui, (cx, cy), dst)
            time.sleep(T_SETTLE)
            img3 = grab()
            box3 = landed_piece(img2, img3, dst, lw, lh)
            if box3 is not None:
                box = box3
                cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
                off = ((cx - dst[0]) ** 2 + (cy - dst[1]) ** 2) ** 0.5
            img2 = img3
            fixed += 1

        if _overlap(box, parked):
            log("   [!] 这块和已经停好的块有重叠，后面可能抓不到它")

        parked.append(box)
        si += 1
        moved += 1
        img = img2
        log("  第 %2d 块：(%d,%d) → (%.0f,%.0f)，偏差 %.0f px，作答区占比 %.2f"
            % (moved, p[0], p[1], dst[0], dst[1], off, board_piece_ratio(img2, cal)))

    log("搬运结束：共搬 %d 块（其中 %d 块摆正过一次），作答区碎片占比 %.2f"
        % (moved, fixed, board_piece_ratio(img, cal)))
    return img


# ============================== 五、配对 ==============================
def ncc(a, b):
    """两块小图的相关系数，只用来排序。"""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if ga.std() < 1e-3 or gb.std() < 1e-3:
        return -1.0
    return float(cv2.matchTemplate(ga, gb, cv2.TM_CCOEFF_NORMED)[0, 0])


def build_candidates(img, pieces, grid, wcells, n_white, cal):
    """给每个格子排一张「先试谁」的候选表。"""
    gx, gy, cw, ch = cal["grid"]
    lw, lh = cal["lw"], cal["lh"]
    out = {}
    for (r, c) in wcells:
        out[(r, c)] = [("white", i) for i in range(n_white)]
    for r in range(5):
        for c in range(5):
            if (r, c) in out or (r, c) not in grid:
                continue
            x0 = int(round(gx + c * cw))
            y0 = int(round(gy + r * ch))
            rc = cv2.resize(img[y0:y0 + int(ch), x0:x0 + int(cw)],
                            (int(lw), int(lh)), interpolation=cv2.INTER_AREA)
            sc = [(ncc(rc, p["face"]), i) for i, p in enumerate(pieces)
                  if p["label"] == grid[(r, c)]]
            sc.sort(reverse=True)
            out[(r, c)] = [("photo", i) for _, i in sc]
    return out


def play_order():
    """落格顺序：外圈（角→边）→ 内圈。"""
    corners = {(0, 0), (0, 4), (4, 0), (4, 4)}
    outer = [(r, c) for r in range(5) for c in range(5)
             if r in (0, 4) or c in (0, 4)]
    outer.sort(key=lambda p: 0 if p in corners else 1)
    inner = [(r, c) for r in range(5) for c in range(5)
             if not (r in (0, 4) or c in (0, 4))]
    return outer + inner


# ============================== 六、白格需要什么形状 ==============================
def _frac(mask, x1, x2, y1, y2):
    """mask 在指定矩形里的占比。"""
    h, w = mask.shape[:2]
    x1, x2 = max(0, int(x1)), min(w, int(x2))
    y1, y2 = max(0, int(y1)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    s = mask[y1:y2, x1:x2]
    return float(s.mean()) if s.size else 0.0


def side_strips(x0, y0, w, h):
    """返回四条边的外侧条带和内侧条带坐标。"""
    hw = w * MID / 2.0
    hh = h * MID / 2.0
    cx1, cx2 = x0 + w / 2.0 - hw, x0 + w / 2.0 + hw
    cy1, cy2 = y0 + h / 2.0 - hh, y0 + h / 2.0 + hh
    outer = {
        "top": (cx1, cx2, y0 - GAP - THK, y0 - GAP),
        "bottom": (cx1, cx2, y0 + h + GAP, y0 + h + GAP + THK),
        "left": (x0 - GAP - THK, x0 - GAP, cy1, cy2),
        "right": (x0 + w + GAP, x0 + w + GAP + THK, cy1, cy2),
    }
    inner = {
        "top": (cx1, cx2, y0 + GAP, y0 + GAP + THK),
        "bottom": (cx1, cx2, y0 + h - GAP - THK, y0 + h - GAP),
        "left": (x0 + GAP, x0 + GAP + THK, cy1, cy2),
        "right": (x0 + w - GAP - THK, x0 + w - GAP, cy1, cy2),
    }
    return outer, inner


def read_board_required(img, cal, filled=None):
    """读盘面：每个空格四条边分别需要什么形状。"""
    bx, by, bw, bh = cal["board"]
    cw, ch = bw / 5.0, bh / 5.0
    board_mask = (np.abs(img.astype(np.int16) - BOARD_BGR).sum(axis=2) < 30)
    piece_mask = ~board_mask

    if filled is None:
        filled = set()
        for r in range(5):
            for c in range(5):
                if cell_filled(img, cal, r, c):
                    filled.add((r, c))

    req = {}
    for r in range(5):
        for c in range(5):
            if (r, c) in filled:
                continue
            x0, y0 = bx + c * cw, by + r * ch
            outer, inner = side_strips(x0, y0, cw, ch)
            need = {}
            for s in SIDES:
                nr = r + (-1 if s == "top" else 1 if s == "bottom" else 0)
                nc = c + (-1 if s == "left" else 1 if s == "right" else 0)
                if not (0 <= nr < 5 and 0 <= nc < 5):
                    need[s] = "flat"
                    continue
                if (nr, nc) not in filled:
                    need[s] = None
                    continue
                x1, x2, y1, y2 = inner[s]
                need[s] = "notch" if _frac(piece_mask, x1, x2, y1, y2) > 0.35 else "tab"
            req[(r, c)] = need
    return req, filled


def white_features(img, x, y, lw, lh):
    """量一块白块四条边 → {side: (外侧占比, 内侧占比, 可不可信)}。"""
    white = ((img[:, :, 0] >= WHITE_PIX) & (img[:, :, 1] >= WHITE_PIX)
             & (img[:, :, 2] >= WHITE_PIX))
    x0, y0 = x - lw / 2.0, y - lh / 2.0
    outer, inner = side_strips(x0, y0, lw, lh)

    def far_rect(s):
        x1, x2, y1, y2 = outer[s]
        if s == "top":
            return x1, x2, y1 - 14, y1 - 2
        if s == "bottom":
            return x1, x2, y2 + 2, y2 + 14
        if s == "left":
            return x1 - 14, x1 - 2, y1, y2
        return x2 + 2, x2 + 14, y1, y2

    feat = {}
    for s in SIDES:
        x1, x2, y1, y2 = outer[s]
        o = _frac(white, x1, x2, y1, y2)
        x1, x2, y1, y2 = inner[s]
        i = _frac(white, x1, x2, y1, y2)
        fx1, fx2, fy1, fy2 = far_rect(s)
        far = _frac(white, fx1, fx2, fy1, fy2)
        known = not (far > 0.5 and o > 0.5)
        feat[s] = (o, i, known)
    return feat


def side_type(feat, s):
    """这块在这一条边上最像 tab / notch / flat。"""
    o, i, known = feat[s]
    if not known:
        return None
    best, name = 9.9, None
    for t, (eo, ei) in EXPECT.items():
        d = (o - eo) ** 2 + (i - ei) ** 2
        if d < best:
            best, name = d, t
    return name


def missing_flats(feat, need):
    """必需的平边里，这块有几条不是平边。"""
    miss = 0
    for s in SIDES:
        if need.get(s) != "flat":
            continue
        t = side_type(feat, s)
        if t is not None and t != "flat":
            miss += 1
    return miss


def shape_cost(feat, need, flat_w=FLAT_W):
    """这块和「这格需要的形状」差多少，越小越像。"""
    total, wsum = 0.0, 0.0
    for s in SIDES:
        want = need.get(s)
        if want is None:
            continue
        o, i, known = feat[s]
        if not known:
            continue
        eo, ei = EXPECT[want]
        w = flat_w if want == "flat" else 1.0
        total += w * ((o - eo) ** 2 + (i - ei) ** 2)
        wsum += w
    return total / wsum if wsum else 9.9


def rank_pieces(feats, need):
    """给这一格排出「先试谁」的顺序。"""
    return sorted(range(len(feats)),
                  key=lambda i: (missing_flats(feats[i], need),
                                 shape_cost(feats[i], need)))


def assign_whites(cells, feats, reqs):
    """穷举全排列，找总代价最小的「白格 → 白块」对应。"""
    cells = list(cells)
    n = len(cells)
    if n == 0 or len(feats) < n:
        return None
    scored = []
    for perm in itertools.permutations(range(len(feats)), n):
        cost = 0.0
        for k, cell in enumerate(cells):
            need = reqs.get(cell, {})
            f = feats[perm[k]]
            cost += shape_cost(f, need) + FLAT_PEN * missing_flats(f, need)
        scored.append((cost, perm))
    scored.sort(key=lambda x: x[0])
    best_cost, best_perm = scored[0]
    second = scored[1][0] if len(scored) > 1 else best_cost + 9.9
    margin = (second - best_cost) / max(best_cost, 1e-6)
    return {cells[k]: best_perm[k] for k in range(n)}, best_cost, margin


# ============================== 七、执行 ==============================
_last_pos = [None, None]


def move_to(x, y):
    """一步把光标移到位。"""
    ctypes.windll.user32.SetCursorPos(int(round(x)), int(round(y)))


def drag(pyautogui, src, dst, hold=None):
    """按下 → 移动 → 松开。"""
    hold = T_HOLD if hold is None else hold
    if _last_pos[0] != src[0] or _last_pos[1] != src[1]:
        move_to(src[0], src[1])
        time.sleep(0.015)
        _last_pos[0], _last_pos[1] = src[0], src[1]
    pyautogui.mouseDown()
    time.sleep(hold)
    move_to(dst[0], dst[1])
    time.sleep(T_BEFORE_UP)
    pyautogui.mouseUp()
    _last_pos[0], _last_pos[1] = dst[0], dst[1]


def cell_rect(cal, r, c):
    bx, by, bw, bh = cal["board"]
    lw, lh = bw / 5.0, bh / 5.0
    return bx + c * lw, by + r * lh, lw, lh


def cell_filled(img, cal, r, c, inset=12, empty_frac=0.5):
    """这一格里有没有东西（看板底蓝占比）。"""
    x, y, w, h = cell_rect(cal, r, c)
    p = img[int(y) + inset:int(y + h) - inset, int(x) + inset:int(x + w) - inset]
    if p.size == 0:
        return False
    frac = float((np.abs(p.astype(np.int16) - BOARD_BGR).sum(axis=2) < 30).mean())
    return frac < empty_frac


def cell_diff(a, b, cal, r, c, inset=12):
    """同一格在两帧之间的平均像素差。"""
    x, y, w, h = cell_rect(cal, r, c)
    pa = a[int(y) + inset:int(y + h) - inset,
           int(x) + inset:int(x + w) - inset].astype(np.int16)
    pb = b[int(y) + inset:int(y + h) - inset,
           int(x) + inset:int(x + w) - inset].astype(np.int16)
    if pa.size == 0 or pa.shape != pb.shape:
        return 999.0
    return float(np.abs(pa - pb).mean())


def is_game_over(img, base_mean, ratio=0.80):
    """整屏变暗（完成弹窗的黑色遮罩）= 拼完了。"""
    return float(img.mean()) < base_mean * ratio


def place_one(pyautogui, img, cal, src, dst, r, c, verify=True):
    """拖一次；verify=True 时再上拽 20 px 确认吸附。"""
    drag(pyautogui, src, dst)
    time.sleep(T_SETTLE)
    img = grab()
    if not verify:
        if cell_filled(img, cal, r, c):
            return True, img
        drag(pyautogui, dst, src)          # 没落上去，拖回原处，别让它躺在半路
        time.sleep(T_SETTLE)
        return False, grab()

    up = (dst[0], dst[1] - 20)             # 吸附的块会锁死，拽不动
    drag(pyautogui, dst, up)
    time.sleep(T_SETTLE)
    img2 = grab()
    d = cell_diff(img, img2, cal, r, c)
    if d < 3.0:
        return True, img2
    drag(pyautogui, up, src)               # 没吸附 → 放回去
    time.sleep(T_SETTLE)
    return False, grab()


# ============================== 八、主流程 ==============================
def play(pyautogui, model, img, abort=None):
    """一局：搬空作答区 → 拼照片格 → 拼白格 → 判完成。"""
    base_mean = float(img.mean())
    cal = calibrate(img, verbose=True)

    # ---- 难度二独有的第一步：把作答区搬空 ----
    ratio = board_piece_ratio(img, cal)
    if ratio > 0.03:
        log("作答区还有碎片（占比 %.2f），先搬到棋盘两侧" % ratio)
        img = clear_board(pyautogui, cal, abort)
        img = grab()
        cal = calibrate(img, verbose=True)  # 棋盘空了，标定会更准，重算一次
        base_mean = float(img.mean())
        log("作答区已空（占比 %.2f），开始拼图" % board_piece_ratio(img, cal))
        try:                                  # 存一张"摊开之后"的图，出问题可以离线查
            cv2.imwrite(os.path.join(OUT_DIR, "staged.png"), img)
            log("   （已存 %s）" % os.path.join(OUT_DIR, "staged.png"))
        except Exception as e:
            log("   （存 staged.png 失败：%r）" % (e,))
    else:
        log("作答区是空的，直接开始拼图")

    # ---- 识别 ----
    boxes = detect(model, img)
    lw, lh = cal["lw"], cal["lh"]
    cw, ch = cal["grid"][2], cal["grid"][3]
    pieces, cells = split_boxes(boxes, lw, lh, cw, ch)
    grid = class_grid(cells, cal["grid"])
    wcells = white_cells_of(img, cal["grid"])
    wpieces = white_pieces_of(img, lw, lh, cal["board"], cal["grid"],
                              expect=max(1, len(wcells)),
                              photo_boxes=[p["box"] for p in pieces])

    for p in pieces:
        x1, y1, x2, y2 = p["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        p["center"] = (cx, cy)
        p["face"] = cv2.resize(
            img[int(cy - lh / 2):int(cy + lh / 2),
                int(cx - lw / 2):int(cx + lw / 2)],
            (int(lw), int(lh)))

    log("识别完成：照片块 %d、参考格 %d、白格 %d、白块 %d"
        % (len(pieces), len(grid), len(wcells), len(wpieces)))

    cands = build_candidates(img, pieces, grid, wcells, len(wpieces), cal)
    order = play_order()
    pos = {("photo", i): p["center"] for i, p in enumerate(pieces)}
    for i, xy in enumerate(wpieces):
        pos[("white", i)] = xy

    counts = {}
    for (r, c) in order:
        k = len(cands.get((r, c), []))
        counts[k] = counts.get(k, 0) + 1
    log("候选数分布：%s（键=每格候选个数，值=格数）" % counts)
    lack = [(r, c) for (r, c) in order if not cands.get((r, c))]
    if lack:
        log("   [!] 这 %d 格一个候选都没有，永远拼不上：%s" % (len(lack), lack))

    done, tries = set(), 0

    # ---- 阶段一：先拼照片格（白格留空）----
    log("阶段一：先拼 %d 个照片格" % (25 - len(wcells)))
    for (r, c) in order:
        if stopping(abort):
            log("  F8 中断：停止阶段一")
            break
        if (r, c) in wcells:
            continue
        if cell_filled(img, cal, r, c):        # 中断重来时这格已经拼好了
            done.add((r, c))
            log("  · r%dc%d 已经有块了，跳过" % (r, c))
            continue
        x, y, w, h = cell_rect(cal, r, c)
        dst = (x + w / 2.0, y + h / 2.0)
        clist = [k for k in cands.get((r, c), []) if k[0] == "photo"]
        need_verify = True                 # 一律验证是否真的吸附
        for key in clist:
            if key not in pos:
                continue
            tries += 1
            ok, img = place_one(pyautogui, img, cal, pos[key], dst, r, c,
                                verify=need_verify)
            if ok:
                del pos[key]
                done.add((r, c))
                log("  ✔ r%dc%d ← %s%s" % (r, c, key[0] + str(key[1]),
                                           "" if need_verify else "  [免验证]"))
                break
            log("  ✘ r%dc%d 放不上，换下一块" % (r, c))
    log("阶段一结束：已拼 %d 格，拖拽 %d 次" % (len(done), tries))
    if stopping(abort):
        log("  F8 中断：跳过阶段二")
        return len(done), tries

    # ---- 阶段二：读盘面形状 → 直接拼白块 ----
    img = grab()
    reqs, _ = read_board_required(img, cal)
    log("阶段二：读盘面，白格需要的形状：")
    for (r, c) in wcells:
        need = reqs.get((r, c), {})
        log("  r%dc%d  %s" % (r, c, {s: need.get(s) for s in SIDES}))

    feats = [white_features(img, x, y, lw, lh) for (x, y) in wpieces]
    res = assign_whites(wcells, feats, reqs)
    plan = {}
    if res is None:
        log("  形状匹配失败（白块不够/需求未知），退回逐个试")
    else:
        plan, cost, margin = res
        log("  形状匹配完成，总代价 %.3f，领先幅度 %.2f" % (cost, margin))
        for (r, c), idx in plan.items():
            log("    r%dc%d ← 白块%d" % (r, c, idx))

    for (r, c) in order:
        if stopping(abort):
            log("  F8 中断：停止阶段二")
            break
        if (r, c) not in wcells or (r, c) in done:
            continue
        if cell_filled(img, cal, r, c):
            done.add((r, c))
            log("  · r%dc%d 已经有块了，跳过" % (r, c))
            continue
        x, y, w, h = cell_rect(cal, r, c)
        dst = (x + w / 2.0, y + h / 2.0)
        first = plan.get((r, c))
        order_idx = rank_pieces(feats, reqs.get((r, c), {}))
        if first is not None and order_idx and order_idx[0] != first:
            order_idx = [first] + [i for i in order_idx if i != first]
        for i in order_idx:
            key = ("white", i)
            if key not in pos:
                continue
            tries += 1
            ok, img = place_one(pyautogui, img, cal, pos[key], dst, r, c)
            if ok:
                del pos[key]
                done.add((r, c))
                log("  ✔ r%dc%d ← 白块%d %s" % (r, c, i,
                                               "（按形状直放）" if i == first else "（试放）"))
                break
            log("  ✘ r%dc%d 白块%d 放不上" % (r, c, i))

    # ---- 兜底重试：没拼上的格再补两轮 ----
    for _round in range(2):
        if stopping(abort):
            break
        left = [(r, c) for (r, c) in order if (r, c) not in done]
        if not left or not pos:
            break
        for (r, c) in left:
            if (r, c) in done:
                continue
            x, y, w, h = cell_rect(cal, r, c)
            dst = (x + w / 2.0, y + h / 2.0)
            if (r, c) in wcells:
                keys = [("white", i) for i in rank_pieces(feats, reqs.get((r, c), {}))]
            else:
                keys = [k for k in cands.get((r, c), []) if k[0] == "photo"]
            for key in keys:
                if key not in pos:
                    continue
                tries += 1
                ok, img = place_one(pyautogui, img, cal, pos[key], dst, r, c)
                if ok:
                    del pos[key]
                    done.add((r, c))
                    log("  ✔ r%dc%d ← %s（重试成功）" % (r, c, key[0] + str(key[1])))
                    break
                log("  ✘ r%dc%d %s 重试没成" % (r, c, key[0] + str(key[1])))

    # ---- 收尾：一定要等到完成弹窗 ----
    img = grab()
    on_board = sum(1 for r in range(5) for c in range(5) if cell_filled(img, cal, r, c))
    if is_game_over(img, base_mean):
        log("✔ 检测到完成弹窗：%d/25 全部拼完（共拖拽 %d 次）" % (on_board, tries))
    else:
        missing = [(r, c) for (r, c) in order if not cell_filled(img, cal, r, c)]
        log("✘ 没等到完成弹窗：盘面已填 %d/25，还差 %d 格 %s"
            % (on_board, len(missing), missing))
        log("   手里还剩 %d 块：%s" % (len(pos), [k[0] + str(k[1]) for k in pos]))
    return on_board, tries


def wait_for_ready(model, abort=None):
    """等棋盘生成出来。"""
    stable, prev = 0, None
    last_print = 0.0
    while True:
        if stopping(abort):
            log("等待被 F8 中断")
            return None
        img = grab()
        if find_warning_banner(img):
            if time.time() - last_print > 5:
                log("   [!] 页面是『已作废』状态（红条），请 Ctrl+R 刷新后重新选 25 格")
                last_print = time.time()
            stable, prev = 0, None
            time.sleep(0.5)
            continue
        if find_start_button(img) is not None:
            stable, prev = 0, None
            if time.time() - last_print > 5:
                log("   还在开始面板：点「开始这一部分」，或按 F8 重来")
                last_print = time.time()
            time.sleep(0.3)
            continue

        cal = calibrate(img)
        boxes = detect(model, img)
        lw, lh = cal["lw"], cal["lh"]
        cw, ch = cal["grid"][2], cal["grid"][3]
        pieces, cells = split_boxes(boxes, lw, lh, cw, ch)
        ratio = board_piece_ratio(img, cal)
        key = (len(cells), len(pieces), round(ratio, 2))
        ready = (len(cells) >= 15 and (len(pieces) >= 20 or ratio > 0.05))
        if ready:
            stable = stable + 1 if key == prev else 1
            if stable >= 2:
                return img
        else:
            stable = 0
        prev = key
        if time.time() - last_print > 3:
            log("   等棋盘… 参考格 %d，块 %d，作答区占比 %.2f"
                % (len(cells), len(pieces), ratio))
            last_print = time.time()
        time.sleep(0.15)


def find_warning_banner(img):
    """顶部红条（本局成绩作废）→ True。"""
    top = img[:300, :]
    b = top[:, :, 0].astype(np.int16)
    g = top[:, :, 1].astype(np.int16)
    r = top[:, :, 2].astype(np.int16)
    red = ((r > 140) & (g < 90) & (b < 90))
    if red.size == 0:
        return False
    return bool(red.mean(axis=1).max() > 0.6)


def click_start_once(pyautogui):
    """连拍两帧确认按钮没在动，再点。"""
    img1 = grab()
    hit1 = find_start_button(img1)
    if hit1 is None:
        return False
    time.sleep(0.15)
    hit2 = find_start_button(grab())
    if hit2 is None:
        return False
    (x1, y1), _ = hit1
    (x2, y2), _ = hit2
    if abs(x1 - x2) > 5 or abs(y1 - y2) > 5:
        log("按钮位置还在变（页面在动），先不点")
        return False
    log("找到开始按钮 (%d,%d)，点击" % (x2, y2))
    pyautogui.click(x2, y2)
    return True


def start_hotkey_listener(trigger):
    """全局热键 F8：开始 / 停止。"""
    from ctypes import wintypes

    WM_HOTKEY = 0x0312

    def run():
        user32 = ctypes.windll.user32
        if not user32.RegisterHotKey(None, 1, 0, VK_START):
            log("[!] F8 注册失败（被别的程序占用了）")
            return
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            if msg.message == WM_HOTKEY:
                trigger.set()

    threading.Thread(target=run, daemon=True).start()


# ============================== 九、离线看一张图 ==============================
def report(path):
    """看一张图：标定 + 识别 + 搬运计划，不动鼠标。"""
    img = cv2.imread(path)
    if img is None:
        print("读不到图：", path)
        return
    log("图片 %s  %dx%d" % (path, img.shape[1], img.shape[0]))
    cal = calibrate(img, verbose=True)
    ratio = board_piece_ratio(img, cal)
    log("作答区碎片占比 %.3f → %s" % (ratio, "需要先搬运" if ratio > 0.03 else "已经是空的"))
    if ratio > 0.03:
        p = deepest_point(img, cal)
        slots = stage_slots(cal, (img.shape[1], img.shape[0]))
        log("第一个抓取点 %s；落点前 6 个：%s"
            % (p, [(round(x), round(y)) for x, y in slots[:6]]))
        log("落点总数 %d" % len(slots))

    from ultralytics import YOLO
    model = YOLO(MODEL)
    boxes = detect(model, img)
    pieces, cells = split_boxes(boxes, cal["lw"], cal["lh"],
                                cal["grid"][2], cal["grid"][3])
    log("YOLO：块级框 %d 个、格级框 %d 个" % (len(pieces), len(cells)))
    bx, by, bw, bh = cal["board"]
    inside = sum(1 for b in pieces
                 if bx <= (b["box"][0] + b["box"][2]) / 2 <= bx + bw
                 and by <= (b["box"][1] + b["box"][3]) / 2 <= by + bh)
    log("其中 %d 个在作答区内（初始居中时应该接近 30）" % inside)

    grid = class_grid(cells, cal["grid"])
    wcells = white_cells_of(img, cal["grid"])
    log("参考格类别 %d 格，白格 %s" % (len(grid), wcells))
    if ratio > 0.03:
        log("白块先不找：它们这会儿都在作答区里，搬空之后才在托盘上")
        return
    wpieces = white_pieces_of(img, cal["lw"], cal["lh"], cal["board"],
                              cal["grid"], expect=max(1, len(wcells)),
                              photo_boxes=[p["box"] for p in pieces])
    log("白块 %d 个：%s" % (len(wpieces), [(int(x), int(y)) for x, y in wpieces]))


def plan_report(path):
    """看一张摊开后的图：配对计划，不动鼠标。"""
    img = cv2.imread(path)
    if img is None:
        print("读不到图：", path)
        return
    log("图片 %s  %dx%d" % (path, img.shape[1], img.shape[0]))
    cal = calibrate(img, verbose=True)
    log("作答区碎片占比 %.3f（应该接近 0）" % board_piece_ratio(img, cal))

    from ultralytics import YOLO
    model = YOLO(MODEL)
    boxes = detect(model, img)
    lw, lh = cal["lw"], cal["lh"]
    cw, ch = cal["grid"][2], cal["grid"][3]
    pieces, cells = split_boxes(boxes, lw, lh, cw, ch)
    grid = class_grid(cells, cal["grid"])
    wcells = white_cells_of(img, cal["grid"])
    wpieces = white_pieces_of(img, lw, lh, cal["board"], cal["grid"],
                              expect=max(1, len(wcells)),
                              photo_boxes=[p["box"] for p in pieces])
    for p in pieces:
        x1, y1, x2, y2 = p["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        p["center"] = (cx, cy)
        p["face"] = cv2.resize(img[int(cy - lh / 2):int(cy + lh / 2),
                                   int(cx - lw / 2):int(cx + lw / 2)],
                               (int(lw), int(lh)))
    log("认出：照片块 %d、参考格类别 %d、白格 %d、白块 %d"
        % (len(pieces), len(grid), len(wcells), len(wpieces)))
    if pieces:
        lab = {}
        for p in pieces:
            lab[p["label"]] = lab.get(p["label"], 0) + 1
        log("块的类别：%s" % lab)

    cands = build_candidates(img, pieces, grid, wcells, len(wpieces), cal)
    empty = []
    log("每格的候选数（外圈→内圈的顺序）：")
    for (r, c) in play_order():
        k = len(cands.get((r, c), []))
        kind = "白" if (r, c) in wcells else (grid.get((r, c), "?"))
        if k == 0:
            empty.append((r, c))
        log("  r%dc%d  %-10s 候选 %d" % (r, c, kind, k))
    if empty:
        log("!! 这 %d 格一个候选都没有，拼不动：%s" % (len(empty), empty))
    else:
        log("所有格子都有候选，可以正常拼")


def main():
    if "--plan" in sys.argv:
        i = sys.argv.index("--plan")
        if i + 1 < len(sys.argv):
            plan_report(sys.argv[i + 1])
        else:
            print("用法：python level2.py --plan 截图.png")
        return
    if "--image" in sys.argv:
        i = sys.argv.index("--image")
        if i + 1 < len(sys.argv):
            report(sys.argv[i + 1])
        else:
            print("用法：python level2.py --image 截图.png")
        return

    dpi_aware()
    from ultralytics import YOLO
    log("加载模型 …")
    model = YOLO(MODEL)
    model.predict(np.zeros((320, 320, 3), np.uint8), imgsz=IMGSZ, verbose=False)
    log("模型就绪")

    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0        # 节奏由 T_HOLD/T_SETTLE 控制，这里再加就是白等

    trigger = threading.Event()
    start_hotkey_listener(trigger)
    log("按 F8 开始；运行中再按 F8 = 立刻停下")

    try:
        while True:
            trigger.wait()
            trigger.clear()
            try:
                img = grab()
                if find_warning_banner(img):
                    log("[!] 页面有红条，请 Ctrl+R 刷新后重来")
                    continue
                if not click_start_once(pyautogui):
                    log("没有可点的开始按钮，直接等棋盘")
                log("等待棋盘生成 …")
                img = wait_for_ready(model, trigger)
                if img is not None:
                    play(pyautogui, model, img, trigger)
            except Exception as e:
                log("[错误] %r" % (e,))
            if trigger.is_set():          # 这一局是被 F8 停下的 → 别自动重开
                trigger.clear()
                log("\n已停下：按 F8 重新开始\n")
                continue
            log("\n再按 F8 可以重来一次\n")
    except KeyboardInterrupt:
        log("\nCtrl+C 退出")
    finally:
        try:
            pyautogui.mouseUp()           # 万一停的时候还按着左键，松开
        except Exception:
            pass

    log("程序结束")


if __name__ == "__main__":
    main()

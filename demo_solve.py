# -*- coding: utf-8 -*-
r"""拼图考核自动化解题程序（PART 3「照片 + 纯白混合」+ 5×5）

用法：
    python demo_solve.py
        按 F8 开始：自动点「开始这一部分」→ 等棋盘生成 → 截屏 → 识别 → 逐格拖拽
    python demo_solve.py --dry
        只识别、出报告，不碰鼠标
    python demo_solve.py --save
        额外保存原始截图和调试图（默认不写，避免每次拖慢）
    python demo_solve.py --image 某张截图.png
        离线分析一张图片（不操作鼠标，调试用）

输出目录：<本文件目录>\out\
    live_01_detect.png   识别标注（--save 时才有）
    live_02_solved.png   拼好后的效果图（--save 时才有）
    live_report.txt      每块 → 每格的对应表
    play_log.txt         逐次拖拽日志
"""
import ctypes
import os
import sys
import threading
import time

import cv2
import numpy as np

# ============================== 路径与常量 ==============================
# 所有路径都从"本文件所在目录"推出来（运行时仍是绝对路径，但换台电脑/换目录也能跑）：
#   <本文件目录>\models\best.pt   模型
#   <本文件目录>\out\             输出目录（自动创建）
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE, "models", "best.pt")
OUT_DIR = os.path.join(BASE, "out")

IMGSZ, CONF = 1280, 0.20
WHITE_PIX, PURE_MIN, STD_MAX = 250, 0.98, 2.0
BOARD_BGR = np.array([42, 23, 15], np.int16)      # 板区底色 #0f172a
CELL_W, CELL_H = 114, 86                          # 参考格基准尺寸（只用于筛选框）
PIECE_W, PIECE_H = 146, 109                       # 块面基准尺寸（只用于筛选框）

# ===== 全屏布局基准（2560×1600）=====
# 浏览器保持全屏（无地址栏）时这组常量是准的；实际以 calibrate() 的检测结果为准，
# 这一组只是检测失败时的兜底。
FULL_SCREEN = (2560, 1600)
BOARD_FIXED = (607, 648, 730, 547)                # 板区 x,y,w,h
GRID_FIXED = (1957.0, 728.0, 114.0, 85.0)         # 参考网格 x0,y0,格宽,格高

# 部分颜色定义（"开始这一部分"按钮是靛蓝实心大块）
BTN_AREA_MIN = 20000

os.makedirs(OUT_DIR, exist_ok=True)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ============================== 基础工具 ==============================
def dpi_aware():
    """必须最先做：否则截图是物理像素、鼠标按逻辑像素，点击位置全错"""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def grab():
    import mss
    with mss.mss() as s:
        shot = np.array(s.grab(s.monitors[1]))
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


def find_start_button(img):
    """返回「开始这一部分」按钮中心 (x,y)；找不到返回 None"""
    b, g, r = cv2.split(img.astype(np.int16))
    mask = ((b > 150) & (r > 60) & (r < 150) & (g < 120) & ((b - g) > 80)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
    best, box = 0, None
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < BTN_AREA_MIN or h > 120 or w < 200:      # 按钮是"宽而扁"的实心块
            continue
        if a > best:
            best, box = a, (int(x), int(y), int(w), int(h))
    if box is None:
        return None
    return (box[0] + box[2] // 2, box[1] + box[3] // 2), box


def find_board(img):
    """板区深蓝矩形 → (x,y,w,h)；找不到返回 None"""
    d = np.abs(img.astype(np.int16) - BOARD_BGR).sum(2)
    m = (d < 10).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    best = None
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 100000:
            continue
        ar = w / float(h)
        if 1.15 < ar < 1.55 and 500 < w < 1200:          # 730×547 ≈ 1.33
            if best is None or a > best[4]:
                best = (int(x), int(y), int(w), int(h), int(a))
    return best[:4] if best else None


# ============================== 标定 ==============================
def calibrate(img, yolo_boxes):
    """从 YOLO 框 + 白格块，自动拟合 板区 / 参考网格 的原点与格尺寸。

    返回 dict：board=(x,y,w,h)  grid=(x0,y0,cw,ch)  lw,lh
    """
    board = find_board(img)
    src = "检测"
    if board is None:                       # 全屏下不可能走到这里，只作保险
        board, src = BOARD_FIXED, "回退全屏基准"
        print("[标定] 没找到板区，使用全屏基准常量")

    bx, by, bw, bh = board
    lw, lh = bw / 5.0, bh / 5.0                     # 板格 = 块面尺寸

    # ===== 参考网格的几何先验 =====
    # 参考网格相对板区的偏移固定为 (1350, 80)：页面缩放/多一条提示条时，
    # 板区和网格会一起平移，这个相对偏移不变，所以拟合失败时用它兜底。
    GRID_DX, GRID_DY = 1350.0, 80.0

    def by_offset():
        return (bx + GRID_DX, by + GRID_DY, float(CELL_W), float(CELL_H))

    # 1) 参考格候选：尺寸像格子 + 必须在板区右侧（防止顶部误检把整张网格拉偏）
    right_min = bx + bw + 30
    cells = [b for b in yolo_boxes
             if abs((b[2] - b[0]) - CELL_W) < 16 and abs((b[3] - b[1]) - CELL_H) < 16
             and b[0] > right_min]
    # 2) 参考栏里的纯白格（尺寸同样接近格子）
    white = ((img[:, :, 0] >= WHITE_PIX) & (img[:, :, 1] >= WHITE_PIX)
             & (img[:, :, 2] >= WHITE_PIX)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(white, 8)
    wcells = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if (100 <= w <= 130 and 75 <= h <= 100 and a > 5000
                and x > right_min and by - 150 < y < by + bh + 200):
            wcells.append((float(x), float(y), float(w), float(h)))
    if len(cells) < 8:
        print("[标定] 参考格检出太少（%d），改用板区偏移兜底" % len(cells))
        return {"board": board, "grid": by_offset(), "lw": lw, "lh": lh,
                "src": src + "+偏移兜底"}

    allx = [c[0] for c in cells] + [w[0] for w in wcells]
    ally = [c[1] for c in cells] + [w[1] for w in wcells]
    allw = [c[2] - c[0] for c in cells] + [w[2] for w in wcells]
    allh = [c[3] - c[1] for c in cells] + [w[3] for w in wcells]
    cw, ch = float(np.median(allw)), float(np.median(allh))

    # 3) 用"中心 → 列号"自洽拟合原点（比取 min(x1) 抗漏检）
    cxs = [c[0] + cw / 2 for c in cells] + [w[0] + w[2] / 2 for w in wcells]
    cys = [c[1] + ch / 2 for c in cells] + [w[1] + w[3] / 2 for w in wcells]
    minx, miny = min(cxs), min(cys)
    ixs = [round((x - minx) / cw) for x in cxs]
    iys = [round((y - miny) / ch) for y in cys]
    x0 = float(np.median([x - i * cw for x, i in zip(cxs, ixs)])) - cw / 2
    y0 = float(np.median([y - i * ch for y, i in zip(cys, iys)])) - ch / 2

    # 4) 校验：拟合结果必须能把所有格子落进 0..4，且跨度够（防止整体错行）
    fc = {int(round((x - x0) / cw - 0.5)) for x in cxs}
    fr = {int(round((y - y0) / ch - 0.5)) for y in cys}
    if not (fc <= set(range(5)) and fr <= set(range(5))
            and (max(fc) - min(fc)) >= 2 and (max(fr) - min(fr)) >= 2):
        print("[标定] 网格拟合越界（列 %s / 行 %s），改用板区偏移兜底"
              % (sorted(fc), sorted(fr)))
        return {"board": board, "grid": by_offset(), "lw": lw, "lh": lh,
                "src": src + "+偏移兜底"}

    return {"board": board, "grid": (x0, y0, cw, ch), "lw": lw, "lh": lh, "src": src}


# ============================== 识别 ==============================
def detect(model, img):
    r = model.predict(img, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    out = []
    for i in range(len(r.boxes)):
        x1, y1, x2, y2 = [float(v) for v in r.boxes.xyxy[i].tolist()]
        out.append({"box": (x1, y1, x2, y2),
                    "label": model.names[int(r.boxes.cls[i])],
                    "conf": float(r.boxes.conf[i])})
    return out


def split_boxes(boxes, lw, lh):
    """按"离哪个基准尺寸更近"分类，比给两个区间更稳（两档只差 22%）"""
    cell_w, cell_h = lw * 0.781, lh * 0.781        # 参考格 = 块面 × 0.781（114/146）
    pieces, cells = [], []
    for b in boxes:
        w = b["box"][2] - b["box"][0]
        h = b["box"][3] - b["box"][1]
        d_piece = abs(w - lw) / lw + abs(h - lh) / lh
        d_cell = abs(w - cell_w) / cell_w + abs(h - cell_h) / cell_h
        if min(d_piece, d_cell) > 0.20:
            continue                                # 尺寸不伦不类，丢弃
        (pieces if d_piece <= d_cell else cells).append(b)
    return pieces, cells


def class_grid(cells, grid):
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
    """判断这个位置是不是一块"纯白块"。

    只看白占比不够：白底照片（白杯子、白椅子）也能到 0.8 以上。所以三条一起判：
        ① 纯白像素占比 ≥ 0.75（被邻块压住时也会掉，所以门槛不能太高）
        ② 彩色像素 < 5%（排除有颜色的主体）
        ③ 内区边缘能量 < 40（纯白块内部几乎没有纹理，照片一定有）
    ② 挡不住白杯子/白椅子——它们本身没颜色，靠的是 ③。
    """
    f = img[int(cy - lh / 2):int(cy + lh / 2), int(cx - lw / 2):int(cx + lw / 2)]
    if f.size == 0:
        return False
    pure = float((f >= WHITE_PIX).all(axis=2).mean())
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    colored = float(((hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 70)).mean())
    # 内缩 12 px 再算梯度，避开面边框本身；纯白面内部梯度接近 0，照片块一定有纹理
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[12:-12, 12:-12]
    if g.size == 0:
        return False
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    edge = float(np.mean(cv2.magnitude(gx, gy)))
    return pure >= 0.75 and colored < 0.05 and edge < 40.0


def white_pieces_of(img, lw, lh, board, grid, expect=5, photo_boxes=None):
    """找托盘里的白块，返回每块的面中心坐标。

    做法（腐蚀分离）：把纯白区域按略大于 tab 宽度的核（≈lw×0.14）腐蚀一次。
    每块白块的面会剩下一个 ~105×68 的核心；两块白块之间靠 tab 相连的十几像素桥会被腐蚀断，
    所以"每块白块 = 一个连通域"。照片块的白背景腐蚀后只剩碎片，过不了面积门槛。

    之后再做两道过滤：
      · looks_white —— 排除"白底照片"（内部有纹理/彩色）；
      · photo_boxes（YOLO 检出的照片框）—— 白块 YOLO 检不出来，
        所以位置和某张照片块重合的候选一定是误判。
    """
    bx, by, bw, bh = board
    gx = grid[0]
    white = ((img[:, :, 0] >= WHITE_PIX) & (img[:, :, 1] >= WHITE_PIX)
             & (img[:, :, 2] >= WHITE_PIX)).astype(np.uint8)
    white[:int(by * 0.45), :] = 0                    # 顶部 UI
    white[:, int(gx - 40):] = 0                      # 参考栏（用标定出来的网格左界，别猜）
    white[by - 6:by + bh + 6, bx - 6:bx + bw + 6] = 0  # 板区本身

    k = max(9, int(round(lw * 0.14)) | 1)            # 146 → 21 px，略大于 tab 宽度
    er = cv2.erode(white, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(er, 8)
    comps = sorted([(float(st[i, 4]), float(cen[i][0]), float(cen[i][1]))
                    for i in range(1, n)], reverse=True)

    # ---- 面积门槛 + 逐级放宽 ----
    # 正常一个白块的面腐蚀后还有 1 万像素上下；照片块的白背景被主体切断，只剩 1600~1900
    # 的碎片，所以 5000 这条线就能把两者分开。
    # 但白块被邻块压住时面积会掉下来，偶尔会掉到 5000 以下 → 少一块白块 → 那个白格
    # 的候选里没有正确的那块 → 整局卡在 24/25。所以这里加一道兜底：
    # 白块数必须等于白格数（expect），不够就把门槛按 0.75 倍逐级放宽重筛，
    # 最低 2500（仍高于照片碎片的 1600~1900）。正常局一次就够，不会触发。
    min_area = 5000.0
    cand = []
    while True:
        cand = [c for c in comps if c[0] >= min_area]
        if len(cand) >= expect or min_area <= 2500.0:
            break
        min_area = max(2500.0, min_area * 0.75)
    if min_area < 5000.0:
        print("[白块] 面积门槛放宽到 %.0f 才凑够 %d 个" % (min_area, expect))

    def on_photo(x, y):
        for (x1, y1, x2, y2) in (photo_boxes or []):
            if abs(x - (x1 + x2) / 2) < lw * 0.5 and abs(y - (y1 + y2) / 2) < lh * 0.5:
                return True
        return False

    picked, dropped, dropped_photo = [], 0, []
    for sc, x, y in cand:
        if any((x - px) ** 2 + (y - py) ** 2 <= (lw * 0.6) ** 2 for _, px, py in picked):
            continue
        if not looks_white(img, x, y, lw, lh):       # 白占比够但内部有纹理 → 照片
            dropped += 1
            continue
        if on_photo(x, y):                           # YOLO 说那儿有照片块 → 硬判
            dropped_photo.append((x, y))
            continue
        picked.append((sc, x, y))
        if len(picked) >= expect:
            break
    if dropped:
        print("[白块] 纹理判据丢弃 %d 个候选" % dropped)
    if dropped_photo:
        print("[白块] YOLO 交叉校验剔除 %d 个非白块：%s"
              % (len(dropped_photo), [(int(x), int(y)) for x, y in dropped_photo]))
    if len(picked) < expect:
        print("[白块] 只找到 %d 个（期望 %d）——可能被压住太多" % (len(picked), expect))
    return [(x, y) for _, x, y in picked]


def ncc(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if ga.std() < 1e-3 or gb.std() < 1e-3:
        return -1.0
    return float(cv2.matchTemplate(ga, gb, cv2.TM_CCOEFF_NORMED)[0, 0])


def assign(img, pieces, grid, wcells, wpieces, cal):
    gx, gy, cw, ch = cal["grid"]
    put, used, plan = {}, set(), []

    # 白格 ← 白块（视觉等价，按最近距离省行程）
    pool = list(range(len(wpieces)))
    for (r, c) in wcells:
        tx, ty = gx + (c + 0.5) * cw, gy + (r + 0.5) * ch
        if not pool:
            break
        k = min(pool, key=lambda i: (wpieces[i][0] - tx) ** 2 + (wpieces[i][1] - ty) ** 2)
        pool.remove(k)
        put[(r, c)] = ("white", k)

    # 照片格 ← 同类块；同类多块时按参考格 NCC 择优
    pairs = []
    for r in range(5):
        for c in range(5):
            if (r, c) in put or (r, c) not in grid:
                continue
            x0 = int(round(gx + c * cw)); y0 = int(round(gy + r * ch))
            rc = cv2.resize(img[y0:y0 + int(ch), x0:x0 + int(cw)],
                            (int(cal["lw"]), int(cal["lh"])), interpolation=cv2.INTER_AREA)
            for i, p in enumerate(pieces):
                if p["label"] != grid[(r, c)]:
                    continue
                pairs.append((ncc(rc, p["face"]), (r, c), i))
    pairs.sort(reverse=True)
    for sc, cell, i in pairs:
        if cell in put or i in used:
            continue
        put[cell] = ("photo", i)
        used.add(i)
    return put, [i for i in range(len(pieces)) if i not in used]


# ============================== 出图 ==============================
def render_solved(pieces, put, cal, zoom=2):
    """把每块贴回它该在的格子。没配上的格子画灰底问号——用来区分
    『算失败』和『算出来就是黑的』，别再出现整张纯黑的图。"""
    bx, by, bw, bh = cal["board"]
    canvas = np.zeros((bh, bw, 3), np.uint8)
    ys = [int(round(k * bh / 5)) for k in range(6)]
    xs = [int(round(k * bw / 5)) for k in range(6)]
    for r in range(5):
        for c in range(5):
            tile = np.full((ys[r + 1] - ys[r], xs[c + 1] - xs[c], 3), 55, np.uint8)
            if (r, c) not in put:
                cv2.putText(tile, "?", (tile.shape[1] // 2 - 20, tile.shape[0] // 2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3, cv2.LINE_AA)
                canvas[ys[r]:ys[r + 1], xs[c]:xs[c + 1]] = tile
    for (r, c), (kind, idx) in put.items():
        tile = np.full((ys[r + 1] - ys[r], xs[c + 1] - xs[c], 3), 255, np.uint8)
        if kind == "photo":
            tile = cv2.resize(pieces[idx]["face"], (xs[c + 1] - xs[c], ys[r + 1] - ys[r]))
        canvas[ys[r]:ys[r + 1], xs[c]:xs[c + 1]] = tile
    big = cv2.resize(canvas, (bw * zoom, bh * zoom), interpolation=cv2.INTER_NEAREST)
    for k in range(1, 5):
        cv2.line(big, (k * bw * zoom // 5, 0), (k * bw * zoom // 5, bh * zoom), (0, 0, 0), 2)
        cv2.line(big, (0, k * bh * zoom // 5), (bw * zoom, k * bh * zoom // 5), (0, 0, 0), 2)
    return big


def render_detect(img, pieces, grid, wcells, wpieces, cal):
    vis = img.copy()
    gx, gy, cw, ch = cal["grid"]
    bx, by, bw, bh = cal["board"]
    for p in pieces:
        x1, y1, x2, y2 = [int(v) for v in p["box"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(vis, p["label"], (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
    for (r, c) in grid:
        x0 = int(round(gx + c * cw)); y0 = int(round(gy + r * ch))
        cv2.rectangle(vis, (x0, y0), (x0 + int(cw), y0 + int(ch)), (0, 255, 0), 2)
    for (r, c) in wcells:
        x0 = int(round(gx + c * cw)); y0 = int(round(gy + r * ch))
        cv2.rectangle(vis, (x0, y0), (x0 + int(cw), y0 + int(ch)), (255, 0, 0), 3)
    for (x, y) in wpieces:
        cv2.circle(vis, (int(x), int(y)), 10, (255, 0, 255), 2)
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
    return vis


# ============================== 分析主流程 ==============================
def analyze(model, img, tag="live"):
    t0 = time.time()
    boxes = detect(model, img)
    cal = calibrate(img, [b["box"] for b in boxes])
    gx, gy, cw, ch = cal["grid"]
    bx, by, bw, bh = cal["board"]
    lw, lh = cal["lw"], cal["lh"]

    pieces_raw, cells_raw = split_boxes(boxes, lw, lh)
    grid = class_grid(cells_raw, cal["grid"])
    wcells = white_cells_of(img, cal["grid"])
    wpieces = white_pieces_of(img, lw, lh, cal["board"], cal["grid"],
                              expect=max(1, len(wcells)),
                              photo_boxes=[p["box"] for p in pieces_raw])

    # 网格尺寸校验：万一选成 3×3/4×4，这里能立刻发现
    cols = sorted({int(round(((b["box"][0] + b["box"][2]) / 2 - gx) / cw - 0.5)) for b in cells_raw})
    col_span = (min(cols), max(cols)) if cols else (0, 0)
    if col_span != (0, 4):
        print("[警告] 参考格列号范围 %s，不是 0..4 —— 这一局可能不是 5×5" % (col_span,))

    for p in pieces_raw:
        x1, y1, x2, y2 = p["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        p["center"] = (cx, cy)
        p["face"] = cv2.resize(
            img[int(cy - lh / 2):int(cy + lh / 2), int(cx - lw / 2):int(cx + lw / 2)],
            (int(lw), int(lh)))

    put, left = assign(img, pieces_raw, grid, wcells, wpieces, cal)
    dt = time.time() - t0

    L = []
    L.append("标定[%s]：板区 (%d,%d) %dx%d   块面 %.0fx%.0f   参考格 (%.0f,%.0f) %.0fx%.0f"
             % (cal.get("src", "检测"), bx, by, bw, bh, lw, lh, gx, gy, cw, ch))
    L.append("YOLO：块 %d 个，参考格类别 %d 个；OpenCV：白格 %d，白块 %d；耗时 %.2f s"
             % (len(pieces_raw), len(grid), len(wcells), len(wpieces), dt))
    missing = [(r, c) for r in range(5) for c in range(5) if (r, c) not in grid]
    L.append("参考图缺的格 %s  ↔  白格 %s   一致: %s"
             % (missing, wcells, sorted(missing) == sorted(wcells)))
    L.append("")
    L.append("配对结果（块 → 格）：")
    for r in range(5):
        row = []
        for c in range(5):
            if (r, c) in put:
                kind, idx = put[(r, c)]
                row.append(("白块" if kind == "white" else pieces_raw[idx]["label"])[:8].ljust(8))
            else:
                row.append("--".ljust(8))
        L.append("  r%d " % r + " ".join(row))

    # 落点计划（暂时只打印，不执行）
    L.append("")
    L.append("落点计划（鼠标执行留到下一步）：")
    for r in range(5):
        for c in range(5):
            if (r, c) not in put:
                continue
            kind, idx = put[(r, c)]
            tx, ty = bx + (c + 0.5) * lw, by + (r + 0.5) * lh
            if kind == "white":
                sx, sy = wpieces[idx]
                L.append("  r%dc%d  %-9s (%4.0f,%4.0f) → (%4.0f,%4.0f)" % (r, c, "白块", sx, sy, tx, ty))
            else:
                sx, sy = pieces_raw[idx]["center"]
                L.append("  r%dc%d  %-9s (%4.0f,%4.0f) → (%4.0f,%4.0f)"
                         % (r, c, pieces_raw[idx]["label"], sx, sy, tx, ty))
    L.append("")
    L.append("未使用块（干扰块）%d 个：" % len(left))
    for i in left:
        p = pieces_raw[i]
        L.append("   %-10s conf=%.2f center=(%.0f,%.0f)" % (p["label"], p["conf"], *p["center"]))
    L.append("缺块的格：%s" % [k for k in [(r, c) for r in range(5) for c in range(5)] if k not in put])

    report = "\n".join(L)
    print(report)
    if "--save" in sys.argv:                  # 出图是为了人工核对；实战时不必每次写 4 张 1.7MB PNG
        cv2.imwrite(os.path.join(OUT_DIR, "%s_raw.png" % tag), img)
        cv2.imwrite(os.path.join(OUT_DIR, "%s_01_detect.png" % tag),
                    render_detect(img, pieces_raw, grid, wcells, wpieces, cal))
        cv2.imwrite(os.path.join(OUT_DIR, "%s_02_solved.png" % tag),
                    render_solved(pieces_raw, put, cal))
    open(os.path.join(OUT_DIR, "%s_report.txt" % tag), "w", encoding="utf-8").write(report)
    print("\n输出目录:", OUT_DIR)
    return put, {"cal": cal, "pieces": pieces_raw, "grid": grid,
                 "wcells": wcells, "wpieces": wpieces}


# ============================== F8 热键 ==============================
def start_hotkey_listener(trigger: threading.Event):
    """用 Win32 RegisterHotKey 监听 F8（不依赖 keyboard 库）"""
    import ctypes
    from ctypes import wintypes

    WM_HOTKEY, VK_F8 = 0x0312, 0x77

    def run():
        user32 = ctypes.windll.user32
        if not user32.RegisterHotKey(None, 1, 0, VK_F8):
            print("[!] F8 注册失败（可能被别的程序占用）")
            return
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            if msg.message == WM_HOTKEY:
                trigger.set()

    t = threading.Thread(target=run, daemon=True)
    t.start()


def wait_for_ready(model, abort_event=None):
    """一直等，直到棋盘真的生成（不设超时、不自动刷新）。

    就绪判据（三条都满足）：
      · 开始按钮已消失
      · 参考格（114×86 级别）≥15 个，且照片块（146×109 级别）≥20 个
      · 连续 2 帧计数一致（挡掉刷新过程中的半成品帧）
    等待期间再按一次 F8 可以中断。
    """
    stable, prev = 0, None
    last_print = 0.0
    last = ""
    while True:
        if abort_event is not None and abort_event.is_set():
            print("等待被 F8 中断")
            return None
        img = grab()
        if find_warning_banner(img):
            if time.time() - last_print > 5:
                print("   [!] 页面是『已作废』状态（红条），请 Ctrl+R 刷新后重新选 25 格")
                last_print = time.time()
            stable, prev = 0, None
            time.sleep(0.5)
            continue
        if find_start_button(img) is not None:      # 还停在开始面板（可能刚刷新过）
            stable, prev = 0, None
            if time.time() - last_print > 5:
                print("   还在开始面板：点「开始这一部分」，或按 F8 重来")
                last_print = time.time()
            time.sleep(0.3)
            continue
        boxes = detect(model, img)
        n_cell = sum(1 for b in boxes
                     if abs((b["box"][2] - b["box"][0]) - CELL_W) < 16
                     and abs((b["box"][3] - b["box"][1]) - CELL_H) < 16)
        n_piece = sum(1 for b in boxes
                      if abs((b["box"][2] - b["box"][0]) - PIECE_W) < 22
                      and abs((b["box"][3] - b["box"][1]) - PIECE_H) < 22)
        last = "参考格 %d，块 %d" % (n_cell, n_piece)
        key = (n_cell, n_piece)
        if n_cell >= 15 and n_piece >= 20:
            stable = stable + 1 if key == prev else 1
            if stable >= 2:                          # 连续 2 帧不变即可（3 帧会多等 ~0.5 s）
                return img
        else:
            stable = 0
        prev = key
        if time.time() - last_print > 3:
            print("   等棋盘… %s" % last)
            last_print = time.time()
        time.sleep(0.15)


def find_warning_banner(img):
    """检测顶部那条红警告条（'页面状态被外部修改，本局成绩作废'）。
    这种局无论怎么点都不会计分，要先刷新页面。"""
    top = img[:300, :]
    b = top[:, :, 0].astype(np.int16)
    g = top[:, :, 1].astype(np.int16)
    r = top[:, :, 2].astype(np.int16)
    red = ((r > 140) & (g < 90) & (b < 90))
    if red.size == 0:
        return False
    return bool(red.mean(axis=1).max() > 0.6)


def click_start_once(pyautogui):
    """连拍两帧确认按钮没在动，再点。防止页面还在渲染时点到别处。"""
    img1 = grab()
    hit1 = find_start_button(img1)
    if hit1 is None:
        return False
    time.sleep(0.15)
    img2 = grab()
    hit2 = find_start_button(img2)
    if hit2 is None:
        return False
    (x1, y1), _ = hit1
    (x2, y2), _ = hit2
    if abs(x1 - x2) > 5 or abs(y1 - y2) > 5:
        print("按钮位置还在变（页面在动），先不点")
        return False
    print("找到开始按钮 (%d,%d)，点击" % (x2, y2))
    pyautogui.click(x2, y2)
    return True


# ============================== 执行层：鼠标拖拽 ==============================
def cell_rect(cal, r, c):
    bx, by, bw, bh = cal["board"]
    lw, lh = bw / 5.0, bh / 5.0
    return bx + c * lw, by + r * lh, lw, lh


def cell_filled(img, cal, r, c, inset=12, empty_frac=0.5):
    """判断这一格是不是已经有东西。

    空格是纯板底蓝 BGR(42,23,15)，板底蓝占比接近 1；贴上图片后会掉到 0.2 以下。
    用"颜色占比"而不是"灰度标准差"：邻块的 tab 偶尔蹭进格子 1% 的面积，
    就能把标准差从 0 抬到 16 以上，用标准差会误判成"已填"。
    注意：它只判"格子里有东西"，分不清"吸附成功"和"块被丢在那儿"，
    后者要靠 cell_diff() 那一拽来区分。
    """
    x, y, w, h = cell_rect(cal, r, c)
    p = img[int(y) + inset:int(y + h) - inset, int(x) + inset:int(x + w) - inset]
    if p.size == 0:
        return False
    frac = float((np.abs(p.astype(np.int16) - BOARD_BGR).sum(axis=2) < 30).mean())
    return frac < empty_frac


LOG_PATH = os.path.join(OUT_DIR, "play_log.txt")


def log(msg):
    """同时打到控制台和 play_log.txt，方便事后复盘。"""
    print(msg)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def reset_log():
    try:
        open(LOG_PATH, "w", encoding="utf-8").close()
    except Exception:
        pass


def is_game_over(img, base_mean, ratio=0.80):
    """拼完了会弹「拼图完成」窗，整屏盖一层黑遮罩 → 平均亮度掉 20% 以上。
    比去找那四个字稳，也不依赖窗口布局。"""
    return float(img.mean()) < base_mean * ratio


def cell_diff(a, b, cal, r, c, inset=12):
    """同一格在两帧之间的平均像素差。用来判断"拽一下有没有把它拽动"。"""
    x, y, w, h = cell_rect(cal, r, c)
    sy = slice(int(y) + inset, int(y + h) - inset)
    sx = slice(int(x) + inset, int(x + w) - inset)
    pa = a[sy, sx].astype(np.int16)
    pb = b[sy, sx].astype(np.int16)
    if pa.size == 0 or pa.shape != pb.shape:
        return 999.0
    return float(np.abs(pa - pb).mean())


def build_candidates(img, pieces, grid, wcells, wpieces, cal):
    """给每个格子排一张候选表 {格: [(kind, idx), ...]}。

    白格：白块外观完全相同，只能挨个试，所以把全部白块都列进去。
    照片格：只列同类块，按与参考格的像素相似度从高到低排（相似度只用于排序，不作定案）。
    """
    gx, gy, cw, ch = cal["grid"]
    lw, lh = cal["lw"], cal["lh"]
    out = {}
    for (r, c) in wcells:
        out[(r, c)] = [("white", i) for i in range(len(wpieces))]
    for r in range(5):
        for c in range(5):
            if (r, c) in out or (r, c) not in grid:
                continue
            x0 = int(round(gx + c * cw)); y0 = int(round(gy + r * ch))
            rc = cv2.resize(img[y0:y0 + int(ch), x0:x0 + int(cw)],
                            (int(lw), int(lh)), interpolation=cv2.INTER_AREA)
            sc = [(ncc(rc, p["face"]), i) for i, p in enumerate(pieces)
                  if p["label"] == grid[(r, c)]]
            sc.sort(reverse=True)
            out[(r, c)] = [("photo", i) for _, i in sc]
    return out


def play_order():
    """外圈（角→边）→ 内圈。
    checkSnap 要求"本块在外圈，或四邻已有一块放好"，这个顺序永远满足。"""
    corners = {(0, 0), (0, 4), (4, 0), (4, 4)}
    outer = [(r, c) for r in range(5) for c in range(5) if r in (0, 4) or c in (0, 4)]
    outer.sort(key=lambda p: 0 if p in corners else 1)
    inner = [(r, c) for r in range(5) for c in range(5) if not (r in (0, 4) or c in (0, 4))]
    return outer + inner


def drag(pyautogui, src, dst, hold=0.03, dur=0.12):
    """dur = 移动时长。验证用的那一小段可以传更短的值（距离近，不需要 0.12）。"""
    pyautogui.moveTo(src[0], src[1], duration=0.05)
    time.sleep(0.02)
    pyautogui.mouseDown()
    time.sleep(hold)
    pyautogui.moveTo(dst[0], dst[1], duration=dur)    # 分步移动，别瞬移（怕丢 pointermove）
    time.sleep(0.03)
    pyautogui.mouseUp()


def execute_play(model, img, pyautogui, abort=None, state=None):
    """按候选表逐格拖。
    验证方式：丢下去后往上拽一小段 —— 吸附成功的块锁死拽不动，没吸附的会被拽动。
    state 传 analyze() 的返回值时，直接复用它的识别结果，省掉一次 YOLO（约 0.3 s）。"""
    reset_log()
    if state is not None:
        cal = state["cal"]
        pieces = state["pieces"]
        grid = state["grid"]
        wcells = state["wcells"]
        wpieces = state["wpieces"]
        lw, lh = cal["lw"], cal["lh"]
    else:
        boxes = detect(model, img)
        cal = calibrate(img, [b["box"] for b in boxes])
        lw, lh = cal["lw"], cal["lh"]
        pieces, cells = split_boxes(boxes, lw, lh)
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

    cands = build_candidates(img, pieces, grid, wcells, wpieces, cal)
    pos = {("photo", i): p["center"] for i, p in enumerate(pieces)}
    for i, xy in enumerate(wpieces):
        pos[("white", i)] = xy

    done = tries = 0
    base_mean = float(img.mean())          # 开局整屏平均亮度，用来判"完成弹窗"
    finished = False
    for (r, c) in play_order():
        if abort is not None and abort.is_set():
            log("   按 F8 中断，停止拖拽")
            break
        if (r, c) not in cands:
            continue
        img = grab()
        if is_game_over(img, base_mean):
            log("   检测到完成弹窗（整屏变暗），停止拖拽")
            finished = True
            break
        if cell_filled(img, cal, r, c):
            done += 1
            continue
        x, y, w, h = cell_rect(cal, r, c)
        dst = (x + w / 2.0, y + h / 2.0)
        # 同类有多块（或白格）才有歧义，必须"拽一下"验证；
        # 该类只有这一块时丢上去就对，省掉一整套验证拖拽。
        need_verify = len(cands[(r, c)]) > 1
        for key in cands[(r, c)]:
            if key not in pos:
                continue
            if is_game_over(img, base_mean):        # 每拖一次前再看一眼，别在弹窗上乱拖
                log("   检测到完成弹窗，停止拖拽")
                finished = True
                break
            src = pos[key]
            tries += 1
            log("  拖 %-12s (%4.0f,%4.0f) → r%dc%d (%4.0f,%4.0f)%s"
                % (key[0] + str(key[1]), src[0], src[1], r, c, dst[0], dst[1],
                   "" if need_verify else "  [免验证]"))
            drag(pyautogui, src, dst)
            time.sleep(0.06)
            img = grab()

            if not need_verify:
                if cell_filled(img, cal, r, c):
                    del pos[key]
                    done += 1
                    log("     ✔ 落位（同类只有这一块，免验证）")
                    break
                log("     ✘ 没落上去，换下一块")
                continue

            # ---- 验证：把刚丢下去的块往上拽一小段，看格子有没有被拽动 ----
            # 吸附成功的块是锁死的，拽不动，格内像素几乎不变（实测恰好 0.0）；
            # 没吸附的一拽就动，格内内容会少掉一截，像素差随位移等比增大。
            # 位移取 20 px：上限是 89 px（相邻格心 109.4 − 吸附半径 20，再大可能误吸到上一格），
            # 下限约 12 px（再小既可能被游戏当成"点击"，深色块的像素差也会逼近阈值）。
            up = (dst[0], dst[1] - 20)
            drag(pyautogui, dst, up, dur=0.03)
            time.sleep(0.06)
            img2 = grab()
            d = cell_diff(img, img2, cal, r, c)
            if d < 3.0:
                del pos[key]
                done += 1
                log("     ✔ 吸附成功（拽不动，格内 diff=%.1f）" % d)
                img = img2
                break
            log("     ✘ 没吸附（拽走了，diff=%.1f），放回原处换下一块" % d)
            pos[key] = up
            drag(pyautogui, up, src)              # 放回原位，保持托盘整洁
            pos[key] = src
            time.sleep(0.05)
            img = grab()
        if finished:
            break
    img = grab()
    n_fill = sum(1 for rr in range(5) for cc in range(5)
                 if cell_filled(img, cal, rr, cc))
    log("执行结束：程序记账 %d 格｜板面实测 %d/25｜共拖 %d 次%s"
        % (done, n_fill, tries, "（检测到完成弹窗）" if finished else ""))
    return img


# ============================== 主程序 ==============================
def main():
    dpi_aware()
    from ultralytics import YOLO

    print("加载模型 …")
    model = YOLO(MODEL)
    model.predict(np.zeros((320, 320, 3), np.uint8), imgsz=IMGSZ, verbose=False)
    print("模型就绪")

    # ---- 离线模式 ----
    if "--image" in sys.argv:
        path = sys.argv[sys.argv.index("--image") + 1]
        img = cv2.imread(path)
        assert img is not None, "读不到图：" + path
        analyze(model, img, tag="offline")
        return

    # ---- 在线模式：F8 触发 ----
    import pyautogui
    pyautogui.FAILSAFE = False
    # pyautogui 默认每次调用后 sleep 0.1 s。一次拖拽有 4 次调用（moveTo/mouseDown/
    # moveTo/mouseUp），27 次拖拽光这个就白等 ~11 s。压到 0.01 s。
    pyautogui.PAUSE = 0.01
    trigger = threading.Event()
    start_hotkey_listener(trigger)
    print("按 F8 开始（点击「开始这一部分」→ 等棋盘 → 截屏识别）")

    while True:
        trigger.wait()
        trigger.clear()
        try:
            img = grab()
            if find_warning_banner(img):
                print("[!] 页面顶部有红条：本局已被判『状态被外部修改』，不计分。")
                print("    请手动 Ctrl+R 刷新，重新选好第 3 部分 + 25 格后再按 F8。")
                print("\n再按 F8 可以重来一次\n")
                continue
            if not click_start_once(pyautogui):
                print("没有可点的开始按钮，直接等棋盘就绪")
            print("等待棋盘生成 …（按 F8 可中断）")
            img = wait_for_ready(model, trigger)
            if img is None:
                continue
            print("棋盘就绪，开始识别 …")
            _put, state = analyze(model, img, tag="live")
            if "--dry" in sys.argv:
                print("（--dry：只出报告，不动鼠标）")
            else:
                print("开始拖拽 …（再按 F8 可中断）")
                execute_play(model, img, pyautogui, trigger, state)
        except Exception as e:
            print("[错误]", repr(e))
        print("\n再按 F8 可以重来一次\n")


if __name__ == "__main__":
    main()

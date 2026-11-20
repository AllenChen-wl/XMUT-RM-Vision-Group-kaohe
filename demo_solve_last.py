# -*- coding: utf-8 -*-
r"""拼图考核自动化解题（PART 3「照片 + 纯白混合」+ 5×5）

先拼 20 个照片格，再从盘面读出每个白格需要的边形状，按形状把白块直接放上去；
落块时仍"拽一下"验证，判错就退回逐个试。

用法：
    python demo_solve_last.py                  按 F8 开始
    python demo_solve_last.py --selftest       合成图自检（不需要游戏和 GPU）
    python demo_solve_last.py --board 图.png   只看一张截图的形状分析

依赖：models\best.pt（与本文件同级的 models\ 目录里）
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

BASE = os.path.dirname(os.path.abspath(__file__))

MODEL = os.path.join(BASE, "models", "best.pt")

IMGSZ, CONF = 1280, 0.20

WHITE_PIX, PURE_MIN, STD_MAX = 250, 0.98, 2.0

BOARD_BGR = np.array([42, 23, 15], np.int16)      # 板区底色 #0f172a

CELL_W, CELL_H = 114, 86                          # 参考格基准尺寸（只用于筛选框）

PIECE_W, PIECE_H = 146, 109                       # 块面基准尺寸（只用于筛选框）

BOARD_FIXED = (607, 648, 730, 547)                # 板区 x,y,w,h

BTN_AREA_MIN = 20000

# ===== 鼠标节奏（调小更快，太小会误判）=====
T_HOLD = 0.025        # 按下鼠标后停一下
T_SETTLE = 0.05       # 动作后等界面刷新，再截图
T_BEFORE_UP = 0.025   # 松手前停一下

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

    # 参考网格相对板区的固定偏移，晶格拟合失败时用它兜底
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

    # 面积门槛：白块 ≥5000，照片碎片只有 1600~1900；
    # 不够 5 个就按 0.75 倍逐级放宽到 2500 重筛
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

def start_hotkey_listener(trigger: threading.Event):
    """用 Win32 RegisterHotKey 监听 F8（不依赖 keyboard 库）"""
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

def log(msg):
    """打一行进度到控制台。"""
    print(msg)

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

_last_pos = [None, None]        # 记住光标上一次停在哪，省掉"原地再移一次"

# 用 Win32 的 SetCursorPos 直接移动光标（pyautogui 的 moveTo 有 0.1 秒下限、只走 2 步）


def move_to(x, y):
    """跳跃式：一步到位（对比版，和 ce.py 的"分小步"互为对照）。

    ce.py 走 8~14 小步、每步停 4 ms，总共 ~40~60 ms；
    这里直接 SetCursorPos 瞬移，耗时几乎为 0。
    好处：快。
    风险：浏览器可能只收到 1 个 mousemove 事件，HTML5 拖动有可能"跟不上鼠标"，
          表现为块没被拖起来、或者落在半路。跑的时候重点看日志里"放不上"的次数。
    """
    ctypes.windll.user32.SetCursorPos(int(round(x)), int(round(y)))


def drag(pyautogui, src, dst, hold=None):
    """一次"按下 → 移动 → 松开"。

    移动用自己写的 move_to（比 pyautogui 快约 3 倍、事件还更多）；
    按下/松开仍用 pyautogui（它会补齐 pointerdown / pointerup 的细节）。
    光标已经在 src 上时跳过移过去那一步——上拽验证本来就是原地。
    """
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


SIDES = ("top", "right", "bottom", "left")

# 判形状时用到的几何（单位 px，按真实块面 146x109 调出来的）
GAP = 4          # 条带离格子/块面的距离
THK = 12         # 条带厚度（凸起约 14 px，取 12 够用）
MID = 0.40       # 只取边长的中间 40%

# 三种形状的参考值：(外侧白占比, 内侧白占比)
EXPECT = {
    "tab": (1.0, 1.0),
    "notch": (0.0, 0.0),
    "flat": (0.0, 1.0),
}


# ============================== 通用小工具 ==============================
def _frac(mask, x1, x2, y1, y2):
    """mask 在指定矩形里的占比（矩形越界会自动裁掉）。"""
    h, w = mask.shape[:2]
    x1, x2 = max(0, int(x1)), min(w, int(x2))
    y1, y2 = max(0, int(y1)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    s = mask[y1:y2, x1:x2]
    return float(s.mean()) if s.size else 0.0


def side_strips(x0, y0, w, h):
    """给一个矩形，返回四条边"外侧条带"和"内侧条带"的坐标。"""
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


# ============================== 一、读盘面：白格需要什么形状 ==============================
def read_board_required(img, cal, filled=None):
    """返回 {(r,c): {side: 'flat'|'tab'|'notch'|None}}，只对空格计算。

    判据：
      · 这条边在盘面边界上      → flat
      · 隔壁没拼                → None（未知）
      · 空格这条边内侧有块像素  → notch（隔壁的凸起顶进来了）
      · 否则                    → tab（内部边不可能是平边）
    """
    bx, by, bw, bh = cal["board"]
    cw, ch = bw / 5.0, bh / 5.0
    board_mask = (np.abs(img.astype(np.int16) - BOARD_BGR).sum(axis=2) < 30)
    piece_mask = ~board_mask                       # True = 有块（非板底色）

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
                    need[s] = "flat"                     # 盘面边界
                    continue
                if (nr, nc) not in filled:
                    need[s] = None                       # 隔壁还空着，不知道
                    continue
                x1, x2, y1, y2 = inner[s]
                need[s] = "notch" if _frac(piece_mask, x1, x2, y1, y2) > 0.35 else "tab"
            req[(r, c)] = need
    return req, filled


# ============================== 二、量白块：它的四条边长什么样 ==============================
def white_features(img, x, y, lw, lh):
    """给一个白块（面心 x,y），返回 {side: (外侧白占比, 内侧白占比, 这条边可不可信)}。

    为什么要第三个值：白块在托盘里可能**和另一块并排贴着**，这时"外侧一片白"并不是凸起，
    而是隔壁那块的面。判据是"凸起只有 15 px 深"——所以再看一条**更远**的带（22~34 px）：
    那一带还是白 → 说明是隔壁的块，这条边判不了，直接弃用。
    """
    white = ((img[:, :, 0] >= WHITE_PIX) & (img[:, :, 1] >= WHITE_PIX)
             & (img[:, :, 2] >= WHITE_PIX))
    x0, y0 = x - lw / 2.0, y - lh / 2.0
    outer, inner = side_strips(x0, y0, lw, lh)

    def far_rect(s):
        """比条带更远的一条带，用来判断"外面到底是凸起还是隔壁的块"。"""
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
        known = not (far > 0.5 and o > 0.5)      # 外面还是一片白 → 是隔壁的块，不是凸起
        feat[s] = (o, i, known)
    return feat


FLAT_W = 2.5        # 平边的权重：盘面边界上的平边是"绝对可靠"的，白块剪影判平边有噪声
FLAT_PEN = 0.8      # 指派时，缺一条"必需平边"额外扣的分


def side_type(feat, s):
    """这块在这一条边上最像什么形状（'tab'/'notch'/'flat'），判不准返回 None。"""
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
    """这一格"必须有平边"的那几条边里，这块有几条不是平边（0 最好）。"""
    miss = 0
    for s in SIDES:
        if need.get(s) != "flat":
            continue
        t = side_type(feat, s)
        if t is not None and t != "flat":
            miss += 1
    return miss


def shape_cost(feat, need, flat_w=FLAT_W):
    """白块 feat 与"这格需要的形状"差多少（越小越像），未知边不参与比较。

    平边的误差乘上 FLAT_W：因为"这格必须平边"是**从盘面边界直接读出来的**，
    而"白块这一边是不是平"是靠剪影量的；量错的代价要放大，才不会一票投给错的块。
    """
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
    """给这一格排一个"先试谁"的顺序：先看缺几条必需平边，再看细节像不像。

    返回的是候选下标列表，**包含全部白块**——所以最坏情况还是把 5 个都试一遍，
    只是"第一个就中"的概率变高了，不可能比乱序更差。
    """
    return sorted(range(len(feats)),
                  key=lambda i: (missing_flats(feats[i], need), shape_cost(feats[i], need)))


def assign_whites(cells, feats, reqs):
    """在 5×5（或更少）范围里找总代价最小的"格 → 白块"一一对应，直接穷举全排列。

    返回 (对应表, 最优代价, 领先幅度)：
      领先幅度 = (第二好 - 最好) / 最好，越大说明越不容易搞错。
    """
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


# ============================== 三、执行：先照片，再白块 ==============================
def _grab():
    return grab()


def _place_one(pyautogui, img, cal, src, dst, r, c, verify=True):
    """拖一次；verify=True 再"往上拽一下"确认吸附。返回 (是否成功, 新截图)。

    verify=False 是"免验证"快路：只有这一格在该类别里**只剩一块**时才用——
    那种情况丢上去必然是对的，没必要再做一整套验证拖拽（省一次拖拽 + 一张截图）。
    """
    drag(pyautogui, src, dst)
    time.sleep(T_SETTLE)
    img = _grab()
    if not verify:
        if cell_filled(img, cal, r, c):
            return True, img
        # 没落上去：把块拖回原处。否则它会躺在半路，下次按老坐标去抓就抓空了。
        drag(pyautogui, dst, src)
        time.sleep(T_SETTLE)
        return False, _grab()
    up = (dst[0], dst[1] - 20)
    drag(pyautogui, dst, up)
    time.sleep(T_SETTLE)
    img2 = _grab()
    d = cell_diff(img, img2, cal, r, c)
    if d < 3.0:
        return True, img2
    drag(pyautogui, up, src)           # 没吸附，放回原处
    time.sleep(T_SETTLE)
    return False, _grab()


def _aborted(abort):
    """按过 F8 没有（没有 event 时永远 False）。"""
    return abort is not None and abort.is_set()

def play(pyautogui, model, img, abort=None):
    base_mean = float(img.mean())        # 开局亮度基准，末尾用它判断"完成弹窗"的黑色遮罩
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

    log("识别完成：照片块 %d、白格 %d、白块 %d" % (len(pieces), len(wcells), len(wpieces)))

    cands = build_candidates(img, pieces, grid, wcells, wpieces, cal)
    order = play_order()
    pos = {("photo", i): p["center"] for i, p in enumerate(pieces)}
    for i, xy in enumerate(wpieces):
        pos[("white", i)] = xy

    done = set()
    tries = 0

    # ---------- 阶段一：只拼照片格 ----------
    log("阶段一：先拼 %d 个照片格（白格留空）" % (25 - len(wcells)))
    for (r, c) in order:
        if _aborted(abort):
            log("  F8 中断：停止阶段一")
            break
        if (r, c) in wcells:
            continue
        x, y, w, h = cell_rect(cal, r, c)
        dst = (x + w / 2.0, y + h / 2.0)
        clist = [k for k in cands.get((r, c), []) if k[0] == "photo"]
        # 该类别只剩这一块时，丢上去必然是对的 → 免掉"拽一下"验证（省一次拖拽+一张截图）。
        # 有歧义（同类多块）才验证，否则会把对的块误判成没吸附。
        need_verify = len(clist) > 1
        for key in clist:
            if key not in pos:
                continue
            tries += 1
            ok, img = _place_one(pyautogui, img, cal, pos[key], dst, r, c,
                                 verify=need_verify)
            if ok:
                del pos[key]
                done.add((r, c))
                log("  ✔ r%dc%d ← %s%s" % (r, c, key[0] + str(key[1]),
                                           "" if need_verify else "  [免验证]"))
                break
            log("  ✘ r%dc%d 放不上，换下一块" % (r, c))
    log("阶段一结束：已拼 %d 格，拖拽 %d 次" % (len(done), tries))
    if _aborted(abort):
        log("  F8 中断：跳过阶段二")
        return len(done), tries

    # ---------- 阶段二：读盘面 + 直接拼白块 ----------
    img = _grab()
    reqs, _ = read_board_required(img, cal)
    log("阶段二：读盘面，白格需求形状如下")
    for (r, c) in wcells:
        need = reqs.get((r, c), {})
        log("  r%dc%d  %s" % (r, c, {s: need.get(s) for s in SIDES}))

    feats = [white_features(img, x, y, lw, lh) for (x, y) in wpieces]
    res = assign_whites(wcells, feats, reqs)
    if res is None:
        log("  形状匹配失败（白块不够或需求未知），退回逐个试")
        plan = {}
    else:
        assign, cost, margin = res
        plan = assign
        log("  形状匹配完成，总代价 %.3f" % cost)
        for (r, c), idx in assign.items():
            log("    r%dc%d ← 白块%d" % (r, c, idx))

    for (r, c) in order:
        if _aborted(abort):
            log("  F8 中断：停止阶段二")
            break
        if (r, c) not in wcells or (r, c) in done:
            continue
        x, y, w, h = cell_rect(cal, r, c)
        dst = (x + w / 2.0, y + h / 2.0)
        first = plan.get((r, c))
        # 候选顺序 = 按"缺平边数 + 形状代价"排出来的顺序（包含全部白块，最多还是试 5 次）
        order_idx = rank_pieces(feats, reqs.get((r, c), {}))
        if first is not None and order_idx and order_idx[0] != first:
            order_idx = [first] + [i for i in order_idx if i != first]
        for i in order_idx:
            key = ("white", i)
            if key not in pos:
                continue
            tries += 1
            ok, img = _place_one(pyautogui, img, cal, pos[key], dst, r, c)
            if ok:
                del pos[key]
                done.add((r, c))
                tag = "（按形状直放）" if i == first else "（退回试放）"
                log("  ✔ r%dc%d ← 白块%d %s" % (r, c, i, tag))
                break
            log("  ✘ r%dc%d 白块%d 放不上" % (r, c, i))

    # 兜底重试：对没拼上的格子再补试两轮（全拼上的局不会进这里）
    for _round in range(2):
        if _aborted(abort):
            log("  F8 中断：跳过补试")
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
                # 重试时一定做验证：上一轮已经错过一次，不能再凭"格子里有东西"判断
                ok, img = _place_one(pyautogui, img, cal, pos[key], dst, r, c)
                if ok:
                    del pos[key]
                    done.add((r, c))
                    log("  ✔ r%dc%d ← %s （重试成功）" % (r, c, key[0] + str(key[1])))
                    break
                log("  ✘ r%dc%d %s 重试没成" % (r, c, key[0] + str(key[1])))

    # 末尾复核：出现完成弹窗（整屏亮度掉 20%）才算真拼完，否则报出还差哪几格
    img = _grab()
    on_board = sum(1 for r in range(5) for c in range(5) if cell_filled(img, cal, r, c))
    finished = is_game_over(img, base_mean)

    if finished:
        log("✔ 检测到完成弹窗：%d/25 全部拼完（共拖拽 %d 次）" % (on_board, tries))
    else:
        missing = [(r, c) for (r, c) in order if not cell_filled(img, cal, r, c)]
        left_keys = list(pos.keys())
        log("✘ 没等到完成弹窗：盘面已填 %d/25，还差 %d 格 %s"
            % (on_board, len(missing), missing))
        log("   手里还剩 %d 块：%s" % (len(left_keys),
                                     [k[0] + str(k[1]) for k in left_keys]))
        if on_board >= 25:
            log("   ↑ 盘面看着满了但没完成：多半有块**躺在那儿没吸附**（不是空格），"
                "需要人工看一眼，或把该格重放一次")

    return on_board, tries


# ============================== 四、离线：只看一张图的形状 ==============================
def board_report(path):
    img = cv2.imread(path)
    if img is None:
        print("读不到图：", path)
        return
    from ultralytics import YOLO
    model = YOLO(MODEL)
    boxes = detect(model, img)
    cal = calibrate(img, [b["box"] for b in boxes])
    lw, lh = cal["lw"], cal["lh"]
    pieces, cells = split_boxes(boxes, lw, lh)
    wcells = white_cells_of(img, cal["grid"])
    wpieces = white_pieces_of(img, lw, lh, cal["board"], cal["grid"],
                                 expect=max(1, len(wcells)),
                                 photo_boxes=[p["box"] for p in pieces])
    reqs, filled = read_board_required(img, cal)
    print("已拼格数 %d，白格 %s" % (len(filled), wcells))
    for (r, c) in wcells:
        need = reqs.get((r, c), {})
        print("  白格 r%dc%d 需要：%s" % (r, c, {s: need.get(s) for s in SIDES}))
    print("\n白块实测特征（外侧白占比 / 内侧白占比）：")
    for i, (x, y) in enumerate(wpieces):
        f = white_features(img, x, y, lw, lh)
        print("  白块%d (%4d,%4d)  %s" % (i, x, y,
              {s: (round(f[s][0], 2), round(f[s][1], 2)) for s in SIDES}))
    feats = [white_features(img, x, y, lw, lh) for (x, y) in wpieces]
    res = assign_whites(wcells, feats, reqs)
    print("\n匹配结果：", res)


# ============================== 五、自检：合成图验证算法 ==============================
def selftest(seed=7, noise=0.0, spacing=40, verbose=True):
    """合成一张"照片块已拼好、白格空着、托盘有 5 个白块"的图，验证形状匹配能否还原正确配对。

    noise    : 给图片加多少高斯噪声（0~20，模拟录屏压缩）
    spacing  : 托盘里白块之间的间距（越小越容易互相贴住，真实情况可能只有十几 px）
    """
    W, H = 1500, 1000
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[:] = (6, 9, 17)                              # 托盘底色（BGR）
    cw, ch = 146, 109
    bx, by = 60, 60
    cv2.rectangle(canvas, (bx, by), (bx + cw * 5, by + ch * 5),
                  tuple(int(v) for v in BOARD_BGR), -1)

    rng = np.random.RandomState(seed)
    whites = [(0, 0), (0, 2), (2, 2), (4, 1), (4, 4)]
    filled = [(r, c) for r in range(5) for c in range(5) if (r, c) not in whites]

    # 每条内部边随机决定"谁凸起"：owner[(a,b)] = 拥有凸起的那一格
    owner = {}
    for r in range(5):
        for c in range(5):
            for (dr, dc, s) in ((-1, 0, "top"), (0, 1, "right")):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < 5 and 0 <= nc < 5):
                    continue
                a, b = (r, c), (nr, nc)
                owner[frozenset((a, b))] = a if rng.rand() < 0.5 else b

    def tab_rects(cell):
        """这一格"拥有凸起"的边 → 画出去的小矩形。"""
        r, c = cell
        x0, y0 = bx + c * cw, by + r * ch
        out = []
        for s in SIDES:
            nr = r + (-1 if s == "top" else 1 if s == "bottom" else 0)
            nc = c + (-1 if s == "left" else 1 if s == "right" else 0)
            if not (0 <= nr < 5 and 0 <= nc < 5):
                continue
            if owner.get(frozenset((cell, (nr, nc)))) is not cell:
                continue
            t = 15
            a, b = int(cw * 0.30), int(cw * 0.70)
            if s == "top":
                out.append((x0 + a, y0 - t, x0 + b, y0))
            elif s == "bottom":
                out.append((x0 + a, y0 + ch, x0 + b, y0 + ch + t))
            else:
                a, b = int(ch * 0.30), int(ch * 0.70)
                if s == "left":
                    out.append((x0 - t, y0 + a, x0, y0 + b))
                else:
                    out.append((x0 + cw, y0 + a, x0 + cw + t, y0 + b))
        return out

    # 画已拼的照片格：实心 + 自己的凸起
    for cell in filled:
        r, c = cell
        x0, y0 = bx + c * cw, by + r * ch
        color = (90 + (r * 37 + c * 53) % 120, 110 + (r * 23 + c * 41) % 110,
                 70 + (r * 61 + c * 29) % 130)
        cv2.rectangle(canvas, (x0, y0), (x0 + cw, y0 + ch), color, -1)
        for (x1, y1, x2, y2) in tab_rects(cell):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)

    # 计算每个白格需要的形状（真值）
    truth = {}
    for (r, c) in whites:
        need = {}
        for s in SIDES:
            nr = r + (-1 if s == "top" else 1 if s == "bottom" else 0)
            nc = c + (-1 if s == "left" else 1 if s == "right" else 0)
            if not (0 <= nr < 5 and 0 <= nc < 5):
                need[s] = "flat"
                continue
            n = (nr, nc)
            if n not in filled:
                need[s] = None
                continue
            need[s] = "notch" if owner.get(frozenset(((r, c), n))) is n else "tab"
        truth[(r, c)] = need

    # 在托盘里画 5 个白块（按真值形状）
    tray_y = by + ch * 5 + 60
    wpts = []
    for k, (r, c) in enumerate(whites):
        x0 = 60 + k * (cw + spacing)
        y0 = tray_y + (k % 2) * 30
        cv2.rectangle(canvas, (x0, y0), (x0 + cw, y0 + ch), (255, 255, 255), -1)
        need = truth[(r, c)]
        t = 15
        for s in SIDES:
            if need[s] == "tab":
                a, b = int(cw * 0.30), int(cw * 0.70)
                if s == "top":
                    rect = (x0 + a, y0 - t, x0 + b, y0)
                elif s == "bottom":
                    rect = (x0 + a, y0 + ch, x0 + b, y0 + ch + t)
                else:
                    a, b = int(ch * 0.30), int(ch * 0.70)
                    rect = (x0 - t, y0 + a, x0, y0 + b) if s == "left" \
                        else (x0 + cw, y0 + a, x0 + cw + t, y0 + b)
                cv2.rectangle(canvas, rect[:2], rect[2:], (255, 255, 255), -1)
            elif need[s] == "notch":
                a, b = int(cw * 0.30), int(cw * 0.70)
                if s == "top":
                    rect = (x0 + a, y0, x0 + b, y0 + t)
                elif s == "bottom":
                    rect = (x0 + a, y0 + ch - t, x0 + b, y0 + ch)
                else:
                    a, b = int(ch * 0.30), int(ch * 0.70)
                    rect = (x0, y0 + a, x0 + t, y0 + b) if s == "left" \
                        else (x0 + cw - t, y0 + a, x0 + cw, y0 + b)
                cv2.rectangle(canvas, rect[:2], rect[2:], (6, 9, 17), -1)
        wpts.append((x0 + cw / 2.0, y0 + ch / 2.0))

    if noise > 0:
        canvas = np.clip(canvas.astype(np.float32) +
                         np.random.RandomState(seed + 1).normal(0, noise, canvas.shape),
                         0, 255).astype(np.uint8)

    cal = {"board": (bx, by, cw * 5, ch * 5),
           "grid": (bx + cw * 5 + 80, by, float(CELL_W), float(CELL_H)),
           "lw": float(cw), "lh": float(ch)}

    reqs, _ = read_board_required(canvas, cal)
    feats = [white_features(canvas, x, y, cw, ch) for (x, y) in wpts]
    res = assign_whites(whites, feats, reqs)
    if res is None:
        print("自检失败：匹配没有结果")
        return False, 9.9
    assign, cost, margin = res
    ok = True
    if verbose:
        print("自检：合成 5 个白格 + 5 个白块（seed=%d noise=%.0f 间距=%d）"
              % (seed, noise, spacing))
    for (r, c) in whites:
        want = whites.index((r, c))
        got = assign.get((r, c))
        flag = "✔" if got == want else "✘"
        if got != want:
            ok = False
        if verbose:
            print("  r%dc%d 需要 %-28s → 匹配到白块%s   %s"
                  % (r, c, {s: truth[(r, c)][s] for s in SIDES}, got, flag))
    if verbose:
        print("总代价 %.3f，领先幅度 %.2f，结果 %s"
              % (cost, margin, "全部正确" if ok else "有错配"))

    # 关键指标：按形状代价排序后，**真正属于这一格的那块排第几**（排名 = 要试几次）
    ranks = []
    for k, (r, c) in enumerate(whites):
        order = rank_pieces(feats, reqs.get((r, c), {}))
        ranks.append(order.index(k) + 1)
    avg_rank = sum(ranks) / float(len(ranks))
    if verbose:
        print("按形状排序时，正确白块的排名：%s（平均 %.2f）" % (ranks, avg_rank))
    return ok, cost, margin, avg_rank


def selftest_all():
    """跑几组难度递增的场景，看形状匹配的稳定性。"""
    scenes = [
        ("干净、宽敞", 7, 0.0, 40),
        ("加噪声 8", 7, 8.0, 40),
        ("加噪声 15", 7, 15.0, 40),
        ("白块挨得近（22 px，真实情况）", 7, 0.0, 22),
        ("又近又有噪声（22 px + 12）", 7, 12.0, 22),
        ("换一组布局 seed=3", 3, 8.0, 20),
        ("再换一组 seed=11", 11, 8.0, 20),
        ("最难：20 px + 噪声 18", 5, 18.0, 20),
    ]
    pass_n = 0
    for name, sd, nz, sp in scenes:
        ok, _cost, _m, _r = selftest(seed=sd, noise=nz, spacing=sp, verbose=False)
        print("  %-28s %s" % (name, "✔ 配对正确" if ok else "✘ 有错配"))
        if not ok:
            selftest(seed=sd, noise=nz, spacing=sp, verbose=True)   # 失败时打印细节
        pass_n += 1 if ok else 0
    print("\n合计 %d / %d 组通过" % (pass_n, len(scenes)))
    return pass_n == len(scenes)


def eval_random(n=150):
    """随机生成大量局面，统计"按形状排序后，正确的那块平均排第几"。

    这个数字直接等于**白块平均要试几次**：
        随机顺序 ≈ 3.0 次（5 个候选取平均）
        按形状排序 < 3.0 次 就是净赚
    """
    import random
    rnd = random.Random(2)
    ranks, acc = [], 0
    for _ in range(n):
        sd = rnd.randrange(10000)
        nz = rnd.uniform(3, 16)
        sp = rnd.choice([20, 22, 26, 32, 40])
        ok, _cost, _m, avg_rank = selftest(seed=sd, noise=nz, spacing=sp, verbose=False)
        ranks.append(avg_rank)
        acc += 1 if ok else 0

    ranks.sort()
    mean = sum(ranks) / len(ranks)
    print("随机 %d 组（噪声 3~16，块间距 20~40）" % n)
    print("  · 一次配对全对的比例：%.1f%%" % (acc * 100.0 / n))
    print("  · 正确白块的平均排名：%.2f 次（随机基准 3.00 次）" % mean)
    print("  · 排名分布：第1名 %.0f%%｜前2名 %.0f%%｜前3名 %.0f%%"
          % (100.0 * sum(1 for r in ranks if r == 1) / n,
             100.0 * sum(1 for r in ranks if r <= 2) / n,
             100.0 * sum(1 for r in ranks if r <= 3) / n))
    print("\n结论：%.2f 次/格 → 5 个白格平均试 %.1f 次（原来随机约 15 次）"
          % (mean, mean * 5))
    return True


def main():
    if "--selftest" in sys.argv:
        if "--eval" in sys.argv:
            ok = eval_random(int(sys.argv[sys.argv.index("--eval") + 1])
                             if sys.argv.index("--eval") + 1 < len(sys.argv)
                             and sys.argv[sys.argv.index("--eval") + 1].isdigit() else 150)
        elif "--all" in sys.argv:
            ok = selftest_all()
        else:
            ok, _cost, _m, _r = selftest()
        sys.exit(0 if ok else 1)
    if "--board" in sys.argv:
        board_report(sys.argv[sys.argv.index("--board") + 1])
        return

    dpi_aware()
    from ultralytics import YOLO
    print("加载模型 …")
    model = YOLO(MODEL)
    model.predict(np.zeros((320, 320, 3), np.uint8), imgsz=IMGSZ, verbose=False)
    print("模型就绪")

    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0        # 显式的 T_HOLD/T_SETTLE 已经卡好节奏，这里再加就是白等
    trigger = threading.Event()
    start_hotkey_listener(trigger)
    print("按 F8 开始（先拼照片块，再读盘面直接拼白块）")

    while True:
        trigger.wait()
        trigger.clear()
        try:
            img = grab()
            if find_warning_banner(img):
                print("[!] 页面有红条，请 Ctrl+R 刷新后重来")
                continue
            if not click_start_once(pyautogui):
                print("没有可点的开始按钮，直接等棋盘")
            print("等待棋盘生成 …")
            img = wait_for_ready(model, trigger)
            if img is None:
                continue
            play(pyautogui, model, img, trigger)
        except Exception as e:
            print("[错误]", repr(e))
        print("\n再按 F8 可以重来一次\n")



if __name__ == "__main__":
    main()

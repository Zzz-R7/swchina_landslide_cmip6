# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:39:05 2026

@author: ZHENGrx
"""

# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ==================== 配置 ====================
TRANSITION_DIR = r"D:/AAAAllllll/htpro/data_2008-2024/CMIP6/analysis/Transition"
OUTPUT_DIR = TRANSITION_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCENARIOS = ["SSP126", "SSP245", "SSP585"]
SCENARIO_LABELS = {"SSP126": "SSP1-2.6", "SSP245": "SSP2-4.5", "SSP585": "SSP5-8.5"}

# ==================== CSV ====================
all_matrices = {}
for ssp in SCENARIOS:
    csv_path = f"{TRANSITION_DIR}/transition_matrix_{ssp}.csv"
    if not os.path.exists(csv_path):
        print(f"⚠️ 缺失: {csv_path}（请先跑数据计算部分）")
        continue
    df = pd.read_csv(csv_path, index_col=0, encoding='utf-8-sig')
    # 确保行列顺序与原脚本一致（极低/低/中/高/极高）
    df = df.reindex(index=df.index, columns=df.columns)
    all_matrices[ssp] = df.values.astype(float)   # 5x5 百分比矩阵
    print(f"✅ 读取 {ssp}: shape={df.shape}")

if not all_matrices:
    raise SystemExit("没有可用的矩阵 CSV，请先运行数据计算脚本。")

# ==================== 字体与样式（Times New Roman） ====================
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
mpl.rcParams['mathtext.fontset'] = 'stix'   
mpl.rcParams['axes.unicode_minus'] = False


FS_LABEL = 22     # 节点旁等级标签
FS_STAGE = 24     # 顶部 "2021–2040 / 2081–2100"
FS_SUB   = 28    # 每个子图的 SSP 标题
FS_SUP   = 26     # 总标题

LEVEL_EN = ["Very Low", "Low", "Moderate", "High", "Very High"]
LEVEL_COLORS = {
    "Very Low":  "#4575B4",
    "Low":       "#91BFDB",
    "Moderate":  "#FEE090",
    "High":      "#FC8D59",
    "Very High": "#D73027",
}


def _bezier(p0, p1, p2, p3, n=80):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 \
           + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3


def draw_sankey(ax, mat_pct, title):
    M = mat_pct / mat_pct.sum()


    order = [4, 3, 2, 1, 0]          # 原索引 4=Very High，0=Very Low
    M = M[np.ix_(order, order)]       # 同步重排行/列

    row_sum = M.sum(axis=1)
    col_sum = M.sum(axis=0)

    gap = 0.04
    scale = 1.0 - gap * 4

    xL0, xL1 = 0.03, 0.07
    xR0, xR1 = 0.93, 0.97

    left_nodes, right_nodes = [], []
    y = 1.0
    for v in row_sum:
        h = v * scale
        left_nodes.append((y - h, y)); y -= (h + gap)
    y = 1.0
    for v in col_sum:
        h = v * scale
        right_nodes.append((y - h, y)); y -= (h + gap)

    right_cursor = [right_nodes[j][1] for j in range(5)]
    for i in range(5):
        cursor_L = left_nodes[i][1]
        for j in range(5):
            v = M[i, j] * scale
            if v <= 0:
                continue
            yL0, yL1 = cursor_L, cursor_L - v; cursor_L = yL1
            yR0, yR1 = right_cursor[j], right_cursor[j] - v; right_cursor[j] = yR1
            xm = (xL1 + xR0) / 2
            top = _bezier(np.array([xL1, yL0]), np.array([xm, yL0]),
                          np.array([xm, yR0]), np.array([xR0, yR0]))
            bot = _bezier(np.array([xR0, yR1]), np.array([xm, yR1]),
                          np.array([xm, yL1]), np.array([xL1, yL1]))
            poly = np.vstack([top, bot])
            # 颜色用重排后对应等级
            ax.fill(poly[:, 0], poly[:, 1],
                    color=LEVEL_COLORS[LEVEL_EN[order[i]]],
                    alpha=0.42, edgecolor='none', zorder=1)

    for i, (yb, yt) in enumerate(left_nodes):
        lvl = LEVEL_EN[order[i]]
        ax.add_patch(plt.Rectangle((xL0, yb), xL1 - xL0, yt - yb,
                                   color=LEVEL_COLORS[lvl], zorder=3))
        ax.text(xL0 - 0.02, (yt + yb) / 2,
                f"{lvl}\n({row_sum[i] * 100:.1f}%)",
                ha='right', va='center',
                fontsize=FS_LABEL, fontfamily='serif')

    for j, (yb, yt) in enumerate(right_nodes):
        lvl = LEVEL_EN[order[j]]
        ax.add_patch(plt.Rectangle((xR0, yb), xR1 - xR0, yt - yb,
                                   color=LEVEL_COLORS[lvl], zorder=3))
        ax.text(xR1 + 0.02, (yt + yb) / 2,
                f"{lvl}\n({col_sum[j] * 100:.1f}%)",
                ha='left', va='center',
                fontsize=FS_LABEL, fontfamily='serif')


    ax.text((xL0 + xL1) / 2, 1.01, "T1", ha='center', va='bottom',
            fontsize=FS_STAGE, fontweight='bold', fontfamily='serif')
    ax.text((xR0 + xR1) / 2, 1.01, "T4", ha='center', va='bottom',
            fontsize=FS_STAGE, fontweight='bold', fontfamily='serif')


    ax.set_title(title, fontsize=FS_SUB, fontweight='bold',
                 fontfamily='serif', pad=4)
    ax.set_xlim(-0.38, 1.38)
    ax.set_ylim(-0.02, 1.07)    
    ax.axis('off')


# ==================== 绘图 ====================
fig, axes = plt.subplots(1, 3, figsize=(17, 8.5))
for ax, ssp in zip(axes, SCENARIOS):
    if ssp not in all_matrices:
        ax.set_visible(False)
        continue
    draw_sankey(ax, all_matrices[ssp], SCENARIO_LABELS[ssp])

fig.suptitle("Sankey Diagram of Landslide Susceptibility Class Transitions",
             fontsize=FS_SUP, fontweight='bold', fontfamily='serif', y=1.03)
plt.tight_layout()
out_path = f"{OUTPUT_DIR}/transition_sankey.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n✅ 桑基图已保存: {out_path}")
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ==================== 配置 ====================
ENSEMBLE_DIR = r"D:/AAAAllllll/htpro/data_2008-2024/CMIP6/analysis/Ensemble_Susc"
OUTPUT_DIR = r"D:/AAAAllllll/htpro/data_2008-2024/CMIP6/analysis/Transition"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCENARIOS = ["SSP126", "SSP245", "SSP585"]
SCENARIO_LABELS = {"SSP126": "SSP1-2.6", "SSP245": "SSP2-4.5", "SSP585": "SSP5-8.5"}
BASELINE = "2021-2040"
FUTURE = "2081-2100"

BINS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.01]
LEVEL_EN = ["Very Low", "Low", "Moderate", "High", "Very High"]
LEVEL_NAMES_CN = ["极低", "低", "中", "高", "极高"]
BLOCK_SIZE = 2048

# ==================== 分级函数 ====================
def classify(data):
    out = np.full(data.shape, -1, dtype=np.int8)
    valid = ~np.isnan(data) & (data > -9000)
    for i in range(5):
        lo, hi = BINS[i], BINS[i + 1]
        out[valid & (data >= lo) & (data < hi)] = i
    return out

# ==================== 计算转移矩阵 ====================
all_matrices = {}
for ssp in SCENARIOS:
    base_path = os.path.join(ENSEMBLE_DIR, f"Mean_{ssp}_{BASELINE}.tif")
    fut_path = os.path.join(ENSEMBLE_DIR, f"Mean_{ssp}_{FUTURE}.tif")
    if not (os.path.exists(base_path) and os.path.exists(fut_path)):
        print(f" 缺失 {ssp} 的tif文件，跳过")
        continue

    print(f"\n=== {ssp}: {BASELINE} → {FUTURE} ===")
    with rasterio.open(base_path) as b_src, rasterio.open(fut_path) as f_src:
        H, W = b_src.height, b_src.width
        matrix = np.zeros((5, 5), dtype=np.int64)

        for r0 in range(0, H, BLOCK_SIZE):
            for c0 in range(0, W, BLOCK_SIZE):
                h = min(BLOCK_SIZE, H - r0)
                w = min(BLOCK_SIZE, W - c0)
                win = Window(c0, r0, w, h)
                b = b_src.read(1, window=win).astype(np.float32)
                f = f_src.read(1, window=win).astype(np.float32)
                cb = classify(b)
                cf = classify(f)
                valid = (cb >= 0) & (cf >= 0)
                for i in range(5):
                    for j in range(5):
                        matrix[i, j] += int(((cb == i) & (cf == j) & valid).sum())

    total = matrix.sum()
    mat_pct = matrix / total * 100
    all_matrices[ssp] = mat_pct

    df_mat = pd.DataFrame(
        mat_pct,
        index=[f"Base_{n}" for n in LEVEL_NAMES_CN],
        columns=[f"Future_{n}" for n in LEVEL_NAMES_CN]
    )
    csv_out = os.path.join(OUTPUT_DIR, f"transition_matrix_{ssp}.csv")
    df_mat.to_csv(csv_out, encoding='utf-8-sig')
    print(df_mat.round(2).to_string())

if not all_matrices:
    raise SystemExit(" 没有成功计算任何转移矩阵，请检查tif路径！")

# ==================== 字体与绘图样式（Times New Roman） ====================
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
mpl.rcParams['axes.unicode_minus'] = False

# ==================== 英文对数热力图 ====================
fig, axes = plt.subplots(1, 3, figsize=(21, 8))

im = None
for idx, (ax, ssp) in enumerate(zip(axes, SCENARIOS)):
    if ssp not in all_matrices:
        ax.set_visible(False)
        continue
    M = all_matrices[ssp]

    positive_values = M[M > 0]
    vmin_val = positive_values.min()
    vmax_val = positive_values.max()

    M_masked = np.ma.masked_less_equal(M, 0)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="#eeeeee")

    im = ax.imshow(M_masked, cmap=cmap, norm=LogNorm(vmin=vmin_val, vmax=vmax_val), aspect='equal')

    # 白色粗网格
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=3)
    ax.tick_params(which='minor', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 单元格标注逻辑
    for i in range(5):
        for j in range(5):
            val = M[i, j]
            if val == 0:
                text_str, color = "0", '#444444'
            elif val < 0.01:
                text_str, color = "<0.01", '#444444'
            else:
                text_str = f"{val:.2f}"
                color = 'white' if val > 1.5 else '#333333'
            ax.text(j, i, text_str, ha='center', va='center',
                    color=color, fontsize=22, fontfamily='serif', fontweight='bold')

    ax.set_xticks(range(5))
    ax.set_xticklabels(LEVEL_EN, rotation=35, ha='right', fontsize=20)
    ax.set_xlabel("T4 (2081–2100)", fontsize=21, fontweight='bold', labelpad=12)

    ax.set_yticks(range(5))
    ax.set_yticklabels(LEVEL_EN, fontsize=20)

    if idx == 0:
        ax.set_ylabel("T1 (2021–2040)", fontsize=22, fontweight='bold', labelpad=12)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', which='both', length=0)

    ax.set_title(SCENARIO_LABELS[ssp], fontsize=26, fontweight='bold', pad=20)

# 共用colorbar
cbar_ax = fig.add_axes([0.9, 0.15, 0.015, 0.75])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Transition Area (%) (Log Color Scale)', fontsize=20, fontweight='bold', labelpad=20)
cbar.ax.tick_params(labelsize=20)
for l in cbar.ax.yaxis.get_ticklabels():
    l.set_fontweight('bold')

plt.subplots_adjust(left=0.1, right=0.88, bottom=0.17, top=0.9, wspace=0.04)

out_img = os.path.join(OUTPUT_DIR, "transition_heatmap_log_en.png")
plt.savefig(out_img, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n 热力图已保存至: {out_img}")
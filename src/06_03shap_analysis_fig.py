
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ==================== 路径 ====================
SHAP_CACHE_PATH = r"D:/AAAAllllll/htpro/data_xinan/modeldata/shap_values_cache.npy"
CSV_PATH        = r"D:/AAAAllllll/htpro/data_xinan/modeldata/training_features.csv"
OUT_DIR         = r"D:/AAAAllllll/htpro/data_xinan/modeldata/shap_result"
os.makedirs(OUT_DIR, exist_ok=True)

factor_names = [
    "Rx1day", "R20mm", "dem", "slope", "aspect",
    "plan", "profile", "stream", "dc", "road", "litho"
]
display_map = {
    "Rx1day": "Rx1day", "R20mm": "R20mm", "dem": "Elevation",
    "slope": "Slope", "aspect": "Aspect", "plan": "Plan curvature",
    "profile": "Profile curvature", "stream": "Distance to rivers",
    "dc": "Distance to faults", "road": "Distance to roads", "litho": "Lithology",
}
CATEGORICAL = {"litho"}

AUTO_THRESHOLD_RX1DAY = True
MANUAL_THRESHOLD_DICT = {"Rx1day": (60, "60 mm")}

# ========== 分特征指定图例位置配置 ==========
legend_pos_config = {
    # 右上角
    "dc":       dict(loc="upper right", bbox_to_anchor=(0.97, 0.97)),
    "road":     dict(loc="upper right", bbox_to_anchor=(0.97, 0.97)),
    "stream":   dict(loc="upper right", bbox_to_anchor=(0.97, 0.97)),
    "dem":      dict(loc="upper right", bbox_to_anchor=(0.97, 0.97)),
    "plan":     dict(loc="upper right", bbox_to_anchor=(0.97, 0.97)),
    "slope":    dict(loc="upper right", bbox_to_anchor=(0.97, 0.97)),
    # 左下角
    "profile":  dict(loc="lower left", bbox_to_anchor=(0.03, 0.03)),
    # 右下角
    "R20mm":    dict(loc="lower right", bbox_to_anchor=(0.97, 0.03)),
    "Rx1day":   dict(loc="lower right", bbox_to_anchor=(0.97, 0.03)),
    # 左上角不变
    "aspect":   dict(loc="upper left", bbox_to_anchor=(0.03, 0.97)),
}

# ==================== 读取数据 ====================
shap_values_array = np.load(SHAP_CACHE_PATH)
if isinstance(shap_values_array, list):
    shap_values_array = shap_values_array[1] if len(shap_values_array) > 1 else shap_values_array[0]
if shap_values_array.ndim == 3:
    shap_values_array = shap_values_array[:, :, 1] if shap_values_array.shape[2] == 2 else shap_values_array.mean(axis=2)

X_df = pd.read_csv(CSV_PATH, index_col=0)[factor_names]

# ==================== 字体与样式 ====================
FS = 23
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = FS
plt.rcParams['axes.labelsize'] = FS
plt.rcParams['xtick.labelsize'] = FS
plt.rcParams['ytick.labelsize'] = FS
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300


def force_font_and_size(fig):
    for ax in fig.axes:
        ax.xaxis.label.set_fontfamily('Times New Roman')
        ax.yaxis.label.set_fontfamily('Times New Roman')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily('Times New Roman')
            label.set_fontsize(FS)
    for text in fig.findobj(match=plt.Text):
        text.set_fontfamily('Times New Roman')

def find_curve_inflection_point(x_arr, y_arr):
    if len(x_arr) < 5:
        return None
    dy = np.diff(y_arr)
    abs_dy = np.abs(dy)
    idx_inflect = np.argmax(abs_dy)
    thr_x = x_arr[idx_inflect + 1]
    return thr_x


# ==================== 连续变量绘图 ====================
def plot_continuous(feat_key, feat_display, shap_vals, feat_vals, out_path,
                    auto_calc_threshold=False,
                    threshold=None, threshold_label=None):
    feat_vals = np.asarray(feat_vals, dtype=float)
    shap_vals = np.asarray(shap_vals, dtype=float)

    # 过滤异常值
    mask = np.isfinite(feat_vals) & np.isfinite(shap_vals)
    if feat_display == "Aspect":
        mask = mask & (feat_vals >= 0) & (feat_vals <= 360)
    feat_vals, shap_vals = feat_vals[mask], shap_vals[mask]

    fig = plt.figure(figsize=(10, 8))
    # 关键：增加左侧留白，解决Y轴标签被截断；right/top/bottom微调
    fig.subplots_adjust(left=0.13, right=0.94, top=0.94, bottom=0.11)
    gs = gridspec.GridSpec(5, 5, hspace=0.22, wspace=0.22)
    ax_main = fig.add_subplot(gs[1:, :4])
    ax_top = fig.add_subplot(gs[0, :4], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, 4], sharey=ax_main)

    # hexbin开启rasterized，底图光栅化，文字线条保持矢量
    hb = ax_main.hexbin(feat_vals, shap_vals, gridsize=50, cmap='viridis',
                        mincnt=1, linewidths=0.1, zorder=2, rasterized=True)
    cbar = fig.colorbar(hb, ax=ax_main, shrink=0.82)
    cbar.set_label("Sample count", fontfamily="Times New Roman")
    for t in cbar.ax.get_yticklabels():
        t.set_fontfamily("Times New Roman")

    ax_main.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1)

    # 分箱均值与bootstrap CI
    bins = 25
    q = np.unique(np.quantile(feat_vals, np.linspace(0, 1, bins + 1)))
    bin_idx = np.digitize(feat_vals, q[1:-1])
    xm, ym, ylo, yhi = [], [], [], []
    rng = np.random.default_rng(42)
    for b in np.unique(bin_idx):
        sel = bin_idx == b
        if sel.sum() < 10:
            continue
        xv, yv = feat_vals[sel], shap_vals[sel]
        xm.append(np.mean(xv))
        ym.append(np.mean(yv))
        boot = np.array([np.mean(yv[rng.integers(0, len(yv), len(yv))]) for _ in range(300)])
        ylo.append(np.percentile(boot, 2.5))
        yhi.append(np.percentile(boot, 97.5))
    xm, ym, ylo, yhi = map(np.array, [xm, ym, ylo, yhi])

    # 自动计算拐点阈值
    calc_thr = None
    if auto_calc_threshold and len(xm) > 5:
        calc_thr = find_curve_inflection_point(xm, ym)
        if calc_thr is not None:
            threshold = calc_thr
            threshold_label = f"{calc_thr:.2f}"

    ax_main.plot(xm, ym, color='#D32F2F', linewidth=2.5, zorder=4, label='Binned mean')
    ax_main.fill_between(xm, ylo, yhi, color='#D32F2F', alpha=0.25, zorder=3, label='95% CI')

    # 绘制阈值竖线 + 防止文字越界
    if threshold is not None:
        ax_main.axvline(threshold, color='black', linestyle=':', linewidth=2, zorder=5)
        xlim_left, xlim_right = ax_main.get_xlim()
        offset = (xlim_right - xlim_left) * 0.015
        text_x = threshold + offset
        if text_x > xlim_right:
            text_x = threshold - offset
        ax_main.text(text_x, ax_main.get_ylim()[1] * 0.94,
                     f'{threshold_label}',
                     fontsize=FS, ha='left', va='top', zorder=6)

    # ========== 顶部直方图，打开坐标轴 ==========
    ax_top.hist(feat_vals, bins=60, color='#4B8BBE', alpha=0.7, edgecolor='white', linewidth=0.5)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.tick_params(axis="x", labelbottom=False)

    # ========== 右侧水平直方图 ==========
    ax_right.hist(shap_vals, bins=60, orientation='horizontal', color='#4B8BBE', alpha=0.7, edgecolor='white', linewidth=0.5)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.tick_params(axis="y", labelleft=False)

    ax_main.set_xlabel(feat_display)
    ax_main.set_ylabel(f"SHAP value for {feat_display}")
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

    # ========== 根据特征key读取图例位置参数 ==========
    leg_kw = legend_pos_config[feat_key]
    ax_main.legend(frameon=False, fontsize=FS, **leg_kw)

    force_font_and_size(fig)

    fig.savefig(out_path, dpi=300)
    pdf_path = out_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)

# ==================== 分类变量绘图 ====================
def plot_categorical(feat_display, shap_vals, feat_vals, out_path):
    df = pd.DataFrame({"cat": feat_vals, "shap": np.asarray(shap_vals, dtype=float)}).dropna()
    order = sorted(df["cat"].unique())
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(left=0.13, right=0.94, top=0.94, bottom=0.11)
    data = [df.loc[df["cat"] == c, "shap"].values for c in order]

    bp = ax.boxplot(data, positions=range(len(order)), widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=2))
    for patch in bp['boxes']:
        patch.set_facecolor('#8fb9d9')
        patch.set_alpha(0.8)
    rng = np.random.default_rng(42)
    for i, d in enumerate(data):
        xj = i + rng.uniform(-0.15, 0.15, len(d))
        ax.scatter(xj, d, s=5, alpha=0.4, color='#4B8BBE', edgecolors='none', rasterized=True)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"Class {c}" for c in order])
    ax.set_xlabel(feat_display)
    ax.set_ylabel(f"SHAP value for {feat_display}")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    force_font_and_size(fig)
    fig.savefig(out_path, dpi=300)
    pdf_path = out_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)
    

# ==================== 批量绘图入口 ====================
if __name__ == "__main__":
    print("开始绘制SHAP依赖图...")

    for feat in factor_names:
        idx = factor_names.index(feat)
        s_vals = shap_values_array[:, idx]
        f_vals = X_df[feat].values
        out_p = os.path.join(OUT_DIR, f"shap_dep_{display_map[feat].replace(' ', '_')}.png")

        if feat in CATEGORICAL:
            plot_categorical(display_map[feat], s_vals, f_vals, out_p)
            print(f"  {display_map[feat]} 完成")
        else:
            auto_thr_flag = False
            thr, thr_lab = None, None
            if feat == "Rx1day" and AUTO_THRESHOLD_RX1DAY:
                auto_thr_flag = True
            else:
                thr, thr_lab = MANUAL_THRESHOLD_DICT.get(feat, (None, None))

            plot_continuous(
                feat_key=feat,
                feat_display=display_map[feat],
                shap_vals=s_vals,
                feat_vals=f_vals,
                out_path=out_p,
                auto_calc_threshold=auto_thr_flag,
                threshold=thr, threshold_label=thr_lab
            )
            print(f"  {display_map[feat]} 完成")

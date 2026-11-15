# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 10:36:01 2026

@author: ZHENGrx
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D

OUT_DIR = r'D:/AAAAllllll/htpro/data_xinan/modeldata/shap_result'
CACHE_X_PATH    = os.path.join(OUT_DIR, 'val_X_cache.pkl')
CACHE_SHAP_PATH = os.path.join(OUT_DIR, 'val_shap_cache.pkl')
CACHE_MEAN_PATH = os.path.join(OUT_DIR, 'shap_folds_mean_cache.npy')

factor_names = ['Rx1day','R20mm','dem','slope','aspect','plan','profile','stream','dc','road','litho']
display_map = {'Rx1day':'Rx1day','R20mm':'R20mm','dem':'Elevation','slope':'Slope','aspect':'Aspect',
               'plan':'Plan curvature','profile':'Profile curvature','stream':'Distance to rivers',
               'dc':'Distance to faults','road':'Distance to roads','litho':'Lithology'}

X_val_all = pd.read_pickle(CACHE_X_PATH)
shap_val_all = pd.read_pickle(CACHE_SHAP_PATH)
fold_shap_means = np.load(CACHE_MEAN_PATH)

# ---------- 对齐校验 ----------
assert isinstance(shap_val_all, pd.DataFrame), 'SHAP 缓存应为带 index/columns 的 DataFrame'
miss_s = set(factor_names) - set(shap_val_all.columns)
assert not miss_s, 'SHAP 缺列: {0}'.format(miss_s)
miss_x = set(factor_names) - set(X_val_all.columns)
assert not miss_x, 'X 缺列: {0}'.format(miss_x)
assert fold_shap_means.shape[1] == len(factor_names), \
    'fold_shap_means.shape[1]={0} 与 len(factor_names)={1} 不符'.format(fold_shap_means.shape[1], len(factor_names))
assert len(X_val_all) == len(shap_val_all), \
    '行数不等 X={0} SHAP={1}, 两者必须来自同一批样本'.format(len(X_val_all), len(shap_val_all))
shap_val_all = shap_val_all[factor_names]
X_val_all = X_val_all[factor_names].reindex(shap_val_all.index)
assert X_val_all.index.equals(shap_val_all.index), 'reindex 后 index 仍不一致'
assert X_val_all.notna().all().all(), 'reindex 后出现 NaN: index 无法一一匹配, 需回训练端核对'
print('对齐检查通过, n_samples =', len(shap_val_all))

FS = 26
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = FS

mean_importance = fold_shap_means.mean(axis=0)
std_importance  = fold_shap_means.std(axis=0)
order = np.argsort(mean_importance)[::-1]
names_sorted = [display_map[factor_names[i]] for i in order]

shap_sorted = shap_val_all[[factor_names[i] for i in order]].values
X_sorted    = X_val_all[[factor_names[i] for i in order]].values

# ---------- 稳健横轴 (1%-99% 定界, 越界点画箭头提示) ----------
q_lo, q_hi = np.nanpercentile(shap_sorted, [1, 99])
lo, hi = q_lo - (q_hi - q_lo) * 0.08, q_hi + (q_hi - q_lo) * 0.08
ax_lo, ax_hi = lo, hi
# 替换原来 fig, ax = plt.subplots(...) 这一行
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(20, 10))  # 画布加宽
gs = gridspec.GridSpec(1, 2, width_ratios=[0.97, 0.03])  # 主图:色条 = 97:3
ax = fig.add_subplot(gs[0])
cax = fig.add_subplot(gs[1])  # 单独给色条的轴

n_features = len(names_sorted)
for i in range(n_features):
    y_pos  = n_features - 1 - i
    s_vals = shap_sorted[:, i]
    f_vals = X_sorted[:, i]


    ok = ~np.isnan(f_vals)
    norm_vals = np.full(len(f_vals), 0.5, dtype=float)
    if ok.sum() > 1:
        p5, p95 = np.nanpercentile(f_vals[ok], [5, 95])
        if p95 > p5:
            norm_vals[ok] = np.clip((f_vals[ok] - p5) / (p95 - p5), 0.0, 1.0)

    rng = np.random.default_rng(42 + i)
    y_jitter = rng.normal(0, 0.08, size=len(s_vals))
    ax.scatter(s_vals, y_pos + y_jitter, c=norm_vals, cmap='coolwarm', vmin=0, vmax=1,
               alpha=0.6, s=20, edgecolors='none', zorder=2)


    ax.errorbar(mean_importance[order[i]], y_pos, xerr=std_importance[order[i]],
                fmt='^', color='black', capsize=8, capthick=3, elinewidth=3,
                markersize=14, zorder=5, markeredgecolor='white', markeredgewidth=1.5)

ax.set_yticks(range(n_features))
ax.set_yticklabels(names_sorted[::-1], fontsize=FS)
ax.axvline(0, color='gray', linestyle='--', linewidth=1.5, zorder=1)
ax.set_xlabel('SHAP value', fontsize=FS + 2)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xlim(-0.38, 0.38)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', labelsize=FS - 2)
ax.tick_params(axis='y', length=0)


legend_elements = [
    Line2D([0], [0], marker='^', linestyle='none', label='Mean +/- SD (Spatial CV)',
           markerfacecolor='black', markeredgecolor='white', markersize=16)
]
ax.legend(handles=legend_elements, loc='lower right', frameon=False, fontsize=FS - 2)

sm = plt.cm.ScalarMappable(cmap='coolwarm',
                           norm=matplotlib.colors.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label('Feature value', fontsize=FS)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(['Low', 'Mid', 'High'])
cbar.ax.tick_params(labelsize=FS - 2)

plt.tight_layout()
out = os.path.join(OUT_DIR, 'PanelAB_Merged_Importance_Beeswarm_v6_pct_colorbar.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
print('OUTPUT_FILE:', out)
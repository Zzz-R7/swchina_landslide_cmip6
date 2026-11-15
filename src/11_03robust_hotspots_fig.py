import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 20

# 数据
data = np.array([
    [31.36, 30.10, 25.47, 13.06],   # Tier 1 Broad（最上面一行）
    [35.22, 31.76, 22.43, 10.59],   # Tier 2 High conf
    [35.80, 52.74,  9.74,  1.71]    # Tier 3 Core（最下面一行）
])

tier_labels = ["Tier 1 Broad", "Tier 2 High conf", "Tier 3 Core"]
prov_colors = {
    'Sichuan': '#6baed6',
    'Yunnan': '#2b5c8a',
    'Guizhou': '#74c476',
    'Chongqing': '#9e9ac8'
}
color_list = [prov_colors["Sichuan"], prov_colors["Yunnan"], prov_colors["Guizhou"], prov_colors["Chongqing"]]

# 画布尺寸 + 子图位置
fig = plt.figure(figsize=(18, 6.0))
ax = fig.add_axes([0.06, 0.14, 0.92, 0.66])

y_pos = np.arange(len(tier_labels))
bar_h = 0.55
left = np.zeros(len(tier_labels))

for i_col in range(data.shape[1]):
    vals = data[:, i_col]
    bars = ax.barh(y_pos, vals, left=left, height=bar_h, color=color_list[i_col], ec="white", linewidth=1.2)
    
    for idx, b in enumerate(bars):
        w = b.get_width()
        # 只对 >5.0 的色块标注，直接忽略 2.7%
        if w >= 5.0: 
            ax.text(b.get_x() + w/2, b.get_y() + bar_h/2, f"{w:.1f}%",
                    ha="center", va="center", c="white", fontsize=21, weight="bold")
            
    left += vals

# 反转Y
ax.invert_yaxis()

ax.set_yticks(y_pos)
ax.set_yticklabels(tier_labels, fontsize=22, weight="bold")
ax.set_xlim(0, 100)
ax.set_xticks([])
ax.set_xlabel("Percentage of Total Area within Each Tier (%)", fontsize=22, weight="bold", labelpad=14)

# 边框
ax.spines[["top", "right", "bottom"]].set_visible(False)
ax.spines["left"].set_linewidth(1.6)

# 标题
fig.text(0.5, 0.93, "(c) Proportion of Hotspot Areas by Province",
         ha="center", va="center", fontsize=24, weight="bold")

# 图例
handles = [Line2D([0], [0], color=c, lw=18) for c in prov_colors.values()]
leg_labels = list(prov_colors.keys())
fig.legend(handles, leg_labels, loc="lower center",
           bbox_to_anchor=(0.5, 0.77), 
           ncol=4, frameon=False, fontsize=22, 
           handlelength=2.0, columnspacing=2.0, handletextpad=0.6)

# 保存
plt.savefig(r"E:\CMIP6_analysis\Hotspots_Tiered\Fig_Stacked_long.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.show()
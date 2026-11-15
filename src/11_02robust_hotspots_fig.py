import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 16

CSV_PATH = r"E:/CMIP6_analysis/Hotspots_Tiered/Provincial_Stats.csv"
OUT_DIR = r"E:/CMIP6_analysis/Hotspots_Tiered"
df = pd.read_csv(CSV_PATH)

provinces = df.sort_values('Total_Hotspot_km2', ascending=False)['Province'].tolist()
tiers = ['Tier1_Broad_km2', 'Tier2_HighConf_km2', 'Tier3_Core_km2']
tier_labels = ['Tier 1 Broad', 'Tier 2 High conf', 'Tier 3 Core']

tier_colors = ['#fee391', '#f28e2b', '#c44e02']


fig, ax = plt.subplots(figsize=(7, 8)) 

size_scale = 6000  

min_size = 500     

text_fontsize = 20 

max_val = df[tiers].values.max()

for i, prov in enumerate(provinces):       # i 控制行 (Y轴)
    for j, tier in enumerate(tiers):       # j 控制列 (X轴)
        val = df.loc[df['Province'] == prov, tier].values[0]
        
        
        s_size = (np.sqrt(val) / np.sqrt(max_val)) * size_scale + min_size 
        
        color = tier_colors[j]
        
        
        ax.scatter(j, i, s=s_size, color=color, alpha=1, edgecolor='#888888', linewidth=1.2, zorder=3)
        
        # 标数值
        if val > 0:
            
            text_color = 'white' if j >= 1 else '#333333' 
            ax.text(j, i, f"{val:,.0f}", ha='center', va='center', 
                    fontsize=text_fontsize, color=text_color, fontweight='bold', zorder=4)


ax.set_xticks(np.arange(len(tiers)))
ax.set_xticklabels(tier_labels, fontweight='bold', fontsize=20)
ax.set_yticks(np.arange(len(provinces)))
ax.set_yticklabels(provinces, fontweight='bold', fontsize=20)


ax.invert_yaxis()


ax.set_xticks(np.arange(len(tiers)+1)-0.5, minor=True)
ax.set_yticks(np.arange(len(provinces)+1)-0.5, minor=True)
ax.grid(which="minor", color="#e0e0e0", linestyle='--', linewidth=1, zorder=0)
ax.tick_params(which="minor", bottom=False, left=False)
ax.tick_params(which="major", length=0, pad=8) # pad 控制坐标轴标签与图表边缘的距离

# 去除外边框
for spine in ax.spines.values():
    spine.set_visible(False)

# 设置坐标轴边界，压缩留白
cell_width = 0.6
ax.set_xlim(-cell_width/2, (len(tiers)-1)+cell_width/2)
ax.set_ylim(len(provinces) - 0.5, -0.5)

plt.title("(b) Matrix of Absolute Hotspot Area (km²)", fontweight='bold', pad=20, fontsize=20)


plt.tight_layout()

plt.savefig(f"{OUT_DIR}/Fig_Bubble_Matrix_BigFont.png", dpi=400, bbox_inches='tight', facecolor='white')
print(" 气泡矩阵图已保存")
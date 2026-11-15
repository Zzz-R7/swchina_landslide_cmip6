import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv('landslide_precip_factors.csv')
print("数据概览：")
print(df.describe())  # 查看均值、标准差、分位数

# 检查是否有空值
print(f"\n缺失值统计：\n{df.isnull().sum()}")

# 如果有空值（比如某个点落在了海洋上），直接剔除
df_clean = df.dropna(subset=['Rx1day_mm', 'R20mm_天数', '极端降水贡献率'])
print(f"\n剔除空值后剩余 {len(df_clean)} 条有效记录")

# 设定阈值
thresholds = {
    'Rx1day_高阈值': 50,      # 日降水≥50mm（暴雨）
    'Rx1day_极高阈值': 100,   # 日降水≥100mm（大暴雨）
    'R20mm_天数_高': 10,      # 一年中≥20mm的天数超过10天
    'R20mm_天数_极高': 20,    # 超过20天
    '贡献率_高': 0.4,         # 极端降水贡献率≥40%
    '贡献率_极高': 0.5        # ≥50%
}

print("========== 极端因子捕获率分析 ==========")
total_points = len(df_clean)

for name, val in thresholds.items():
    if 'Rx1day' in name:
        count = (df_clean['Rx1day_mm'] >= val).sum()
    elif 'R20mm' in name:
        count = (df_clean['R20mm_天数'] >= val).sum()
    elif '贡献率' in name:
        count = (df_clean['极端降水贡献率'] >= val).sum()
    else:
        continue
    rate = count / total_points * 100
    print(f"{name} (阈值={val}): 捕获 {count} 个点，占比 {rate:.1f}%")

print("\n结论：捕获率越高的因子，在该数据集中对滑坡的指示性越强。")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ========== 解决中文显示核心代码 ==========
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示方框问题

# 你的原有代码
corr_data = df_clean[['Rx1day_mm', 'R20mm_天数', '极端降水贡献率']]
corr_matrix = corr_data.corr(method='spearman')

plt.figure(figsize=(6,4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('三个降水因子间的相关性矩阵')
plt.tight_layout()  # 防止标题被截断
plt.show()

# 打印斯皮尔曼相关系数矩阵
print("===== 三个降水因子 Spearman 相关系数矩阵 =====")
print(corr_matrix.round(2))



# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen
import os
import random


# ====================== 全局统一随机种子 ======================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['SimHei'] #无黑体就注释
# ============================================================

# ==================== 请核对修改路径 ====================
model_path = r"D:/AAAAllllll/htpro/data_xinan/modeldata/best/final_model_RF.pkl"
raster_dir = r"D:/AAAAllllll/htpro/data_xinan/rasters_90m/"
shp_path = r"D:/AAAAllllll/htpro/data_xinan/modeldata/modeling_dataset.shp"
cached_X_path = r"D:/AAAAllllll/htpro/data_xinan/modeldata/training_features.csv"
# ========================================================

factor_names = [
    "Rx1day", "R20mm", "dem", "slope", "aspect",
    "plan", "profile", "stream", "dc", "road", "litho"
]

# 1.加载训练完成的RF模型（pkl内部已存储random_state=42）
print("加载最终RF模型...")
model = joblib.load(model_path)
if hasattr(model, "random_state"):
    model.random_state = RANDOM_SEED

# 2.读取/缓存训练特征矩阵
if os.path.exists(cached_X_path):
    print(f"从缓存读取特征矩阵：{cached_X_path}")
    X_df = pd.read_csv(cached_X_path, index_col=0)
    print("缓存文件现有列：", list(X_df.columns))
    X_df = X_df[factor_names]
    print(f"特征矩阵 shape = {X_df.shape}")
else:
    print("未检测到缓存，批量从栅格提取特征...")
    gdf = gpd.read_file(shp_path)
    points = [(pt.x, pt.y) for pt in gdf.geometry]
    X_dict = {}
    for name in factor_names:
        tif_path = os.path.join(raster_dir, f"{name}.tif")
        if not os.path.exists(tif_path):
            raise FileNotFoundError(f"栅格缺失：{tif_path}")
        with rasterio.open(tif_path) as src:
            gen_result = list(sample_gen(src, points))
            X_dict[name] = [v[0] for v in gen_result]

    X_df = pd.DataFrame(X_dict, columns=factor_names)
    if X_df.isnull().any().any():
        print("检测缺失值，中位数填充")
        X_df = X_df.fillna(X_df.median())
    X_df.to_csv(cached_X_path)
    print(f"特征矩阵已缓存至 {cached_X_path}, shape={X_df.shape}")

X = X_df.values

# ========= 使用shap.maskers.Independent，max_samples等于全部样本，消除子采样警告 =========
print("初始化SHAP解释器，使用全部训练样本作为masker背景集...")
masker = shap.maskers.Independent(X_df, max_samples=X_df.shape[0])
explainer = shap.Explainer(model, masker=masker, feature_names=factor_names)
shap_values = explainer(X_df)

# 二分类：提取正类SHAP二维数组
shap_values_array = shap_values.values[:, :, 1]
print(f"SHAP值数组shape：{shap_values_array.shape}")

# 4.输出SHAP全局重要性表格
shap_abs_mean = np.abs(shap_values_array).mean(axis=0)
importance_df = pd.DataFrame({
    '因子': factor_names,
    'SHAP_平均绝对值重要性': shap_abs_mean
}).sort_values("SHAP_平均绝对值重要性", ascending=False)

print("\n======== SHAP全局特征重要性 ========")
print(importance_df.round(4))

# ====================== 5.蜂群beeswarm图 ======================
print("\n绘制并保存蜂群图 beeswarm")
fig, ax = plt.subplots(figsize=(12, 8))
plt.sca(ax)  # 将 ax 设为当前 axes
shap.summary_plot(shap_values_array, X_df, feature_names=factor_names, show=False)
plt.tight_layout()
plt.savefig(r"D:/AAAAllllll/htpro/data_xinan/modeldata/shap_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close(fig) 

# ====================== 6.条形重要性图 ======================
print("绘制并保存条形重要图 bar")
fig, ax = plt.subplots(figsize=(10, 6))
plt.sca(ax)
shap.summary_plot(shap_values_array, X_df, feature_names=factor_names, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(r"D:/AAAAllllll/htpro/data_xinan/modeldata/shap_bar.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ====================== 7.依赖图循环 ======================
print("\n生成关键因子SHAP依赖图")
target_features = ["Rx1day", "R20mm", "road"]
for feat in target_features:
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.dependence_plot(
            feat,
            shap_values_array,
            X_df,
            ax=ax,
            show=False,
            interaction_index=None
        )
        ax.set_title(f"SHAP Dependence Plot — {feat}", fontsize=12)
        plt.tight_layout()
        out_path = rf"D:/AAAAllllll/htpro/data_xinan/modeldata/shap_dep_{feat}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"   {feat} 依赖图已保存")
    except Exception as err:
        print(f"   {feat}绘图失败：{str(err)}")

print("\n SHAP全部分析完成！")
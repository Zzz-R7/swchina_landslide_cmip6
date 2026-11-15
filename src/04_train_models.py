
"""
RF、XGB、Stacking 超参数优化（5折随机搜索）
输出每个模型的最优参数，并保存为 .pkl 文件
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score, f1_score, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# 1. 样本点文件
shp_path = r"D:/AAAAllllll/htpro/data_xinan/modeldata/modeling_dataset.shp"  
# 2. 存放所有因子TIF的文件夹
raster_dir = r"D:/AAAAllllll/htpro/data_xinan/sample/alltif"  
# 3. 输出目录
output_dir = r"D:/AAAAllllll/htpro/data_xinan/modeldata/best"
# 因子列表
factor_names = [
    "Rx1day",      # 极端降水强度（历史多年平均）
    "R20mm",     # 强降水频次（历史多年平均）
    "dem",            # 高程
    "slope",          # 坡度
    "aspect",         # 坡向
    "plan",      # 平面曲率
    "profile",   # 剖面曲率
    "stream",     # 距河流距离
    "dc",     # 距断层距离
    "road",      # 距道路距离
    "litho"           # 岩性（已编码为数值）
]
# 文件路径
tif_files = [f"{raster_dir}/{name}.tif" for name in factor_names]

# ==================== 1. 加载样本并提取栅格值 ====================
print("正在读取样本点...")
gdf = gpd.read_file(shp_path)
print(f"总样本数: {len(gdf)}, 滑坡: {sum(gdf['label']==1)}, 非滑坡: {sum(gdf['label']==0)}")

print("正在提取栅格值（11个因子）...")
X_list = []
for idx, row in gdf.iterrows():
    point = row.geometry
    values = []
    for tif in tif_files:
        with rasterio.open(tif) as src:
            row_idx, col_idx = src.index(point.x, point.y)
            if row_idx < 0 or row_idx >= src.height or col_idx < 0 or col_idx >= src.width:
                val = np.nan
            else:
                val = src.read(1)[row_idx, col_idx]
            values.append(val)
    X_list.append(values)

X = np.array(X_list)
y = gdf['label'].values

# 处理缺失值（用中位数填充）
X_df = pd.DataFrame(X, columns=factor_names)
if X_df.isnull().any().any():
    print("发现缺失值，用各因子中位数填充...")
    X_df = X_df.fillna(X_df.median())
X = X_df.values
print(f"特征矩阵形状: {X.shape}")  # (780, 11)

# 定义评分指标（我们同时关注AUC和F1，但搜索时以AUC为主）
scoring = {'AUC': make_scorer(roc_auc_score, needs_proba=True),
           'F1': make_scorer(f1_score),
           'Accuracy': make_scorer(accuracy_score)}
# 由于RandomizedSearchCV可以接受多指标，我们采用默认的'AUC'作为主要优化目标

# 固定随机种子保证可重复
RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ==================== 2. 随机森林（RF）超参数搜索 ====================
print("\n========== 开始随机森林超参数优化 ==========")
rf_base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
rf_param_dist = {
    'n_estimators': [100, 200, 300, 500, 800],
    'max_depth': [None, 10, 15, 20, 30],
    'min_samples_split': [2, 5, 8, 12],
    'min_samples_leaf': [1, 2, 4, 6],
    'class_weight': ['balanced', 'balanced_subsample', None]
}
rf_random = RandomizedSearchCV(
    rf_base, rf_param_dist, n_iter=30, cv=CV, scoring='roc_auc',
    n_jobs=-1, random_state=RANDOM_STATE, verbose=1
)
rf_random.fit(X, y)
print(f"RF最佳参数: {rf_random.best_params_}")
print(f"RF最佳AUC (CV均值): {rf_random.best_score_:.4f}")
joblib.dump(rf_random.best_params_, f"{output_dir}/rf_best_params.pkl")

# ==================== 3. XGBoost 超参数搜索 ====================
print("\n========== 开始XGBoost超参数优化 ==========")
xgb_base = XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss')
xgb_param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9, 12],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.5],
    'reg_alpha': [0, 0.1, 1, 10],
    'reg_lambda': [0.1, 1, 10]
}
xgb_random = RandomizedSearchCV(
    xgb_base, xgb_param_dist, n_iter=40, cv=CV, scoring='roc_auc',
    n_jobs=-1, random_state=RANDOM_STATE, verbose=1
)
xgb_random.fit(X, y)
print(f"XGB最佳参数: {xgb_random.best_params_}")
print(f"XGB最佳AUC (CV均值): {xgb_random.best_score_:.4f}")
joblib.dump(xgb_random.best_params_, f"{output_dir}/xgb_best_params.pkl")

# ==================== 4. Stacking 集成模型超参数搜索 ====================
print("\n========== 开始Stacking集成超参数优化 ==========")

base_rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5,
                                 min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)
base_xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                         random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss')

stack_base = StackingClassifier(
    estimators=[('rf', base_rf), ('xgb', base_xgb)],
    final_estimator=LogisticRegression(random_state=RANDOM_STATE),
    cv=5,   # 基模型预测时使用的交叉验证折数
    stack_method='predict_proba'
)

stack_param_dist = {
    'final_estimator__C': [0.01, 0.1, 1, 10, 100],
    'rf__n_estimators': [100, 200, 300],
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.05, 0.1, 0.15]
}


stack_random = RandomizedSearchCV(
    stack_base, stack_param_dist, n_iter=20, cv=CV, scoring='roc_auc',
    n_jobs=-1, random_state=RANDOM_STATE, verbose=1
)
stack_random.fit(X, y)
print(f"Stacking最佳参数: {stack_random.best_params_}")
print(f"Stacking最佳AUC (CV均值): {stack_random.best_score_:.4f}")
joblib.dump(stack_random.best_params_, f"{output_dir}/stacking_best_params.pkl")

print("\n三个模型的最优参数已保存到指定目录。")
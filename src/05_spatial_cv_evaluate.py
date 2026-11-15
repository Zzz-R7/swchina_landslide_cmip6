
"""
使用最优超参数重新训练 RF、XGB、Stacking
执行 2km 空间阻隔交叉验证（5折）
输出三个模型的化性能（均值 ± 标准差）
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

shp_path = r"D:/AAAAllllll/htpro/data_xinan/modeldata/modeling_dataset.shp"  
raster_dir = r"D:/AAAAllllll/htpro/data_xinan/rasters_90m/"  
output_dir = r"D:/AAAAllllll/htpro/data_xinan/modeldata/best"
factor_names = [
    "Rx1day",      
    "R20mm",     
    "dem",            
    "slope",          
    "aspect",         
    "plan",      
    "profile",   
    "stream",     
    "dc",       
    "road",
    "litho"           
]
tif_files = [f"{raster_dir}/{name}.tif" for name in factor_names]

# ==================== 1. 加载样本与栅格值 ====================
print("正在读取样本点...")
gdf = gpd.read_file(shp_path)
print(f"总样本数: {len(gdf)}, 滑坡: {sum(gdf['label']==1)}, 非滑坡: {sum(gdf['label']==0)}")

print("正在提取栅格值...")
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
X_df = pd.DataFrame(X, columns=factor_names)
# 先构造DataFrame，再用自身的中位数填充NaN
X_df = X_df.fillna(X_df.median())
X = X_df.values
print(f"特征矩阵形状: {X.shape}")

# ==================== 2. 生成 2km 空间分组标签（基于地理坐标） ====================
print("正在生成2km空间分组...")
# 确保是投影坐标系（Albers，单位米）
gdf_proj = gdf.to_crs("+proj=aea +lat_1=25 +lat_2=47 +lon_0=105")
xs = gdf_proj.geometry.x.values
ys = gdf_proj.geometry.y.values

min_x, max_x = xs.min(), xs.max()
min_y, max_y = ys.min(), ys.max()
grid_size = 2000  # 2km

grid_x = np.floor((xs - min_x) / grid_size).astype(int)
grid_y = np.floor((ys - min_y) / grid_size).astype(int)
groups = np.array([f"{gx}_{gy}" for gx, gy in zip(grid_x, grid_y)])

unique_groups = np.unique(groups)
print(f"共生成 {len(unique_groups)} 个渔网格，平均每组样本数: {len(gdf)/len(unique_groups):.1f}")

# ==================== 3. 加载第1步搜索出的最优参数 ====================
print("\n加载最优参数...")
rf_params = joblib.load(f"{output_dir}/rf_best_params.pkl")
xgb_params = joblib.load(f"{output_dir}/xgb_best_params.pkl")
stacking_params = joblib.load(f"{output_dir}/stacking_best_params.pkl")
print("参数加载完成。")

# ==================== 4. 定义要比较的三个模型 ====================

final_C = stacking_params.get('final_estimator__C', 1.0)

base_rf_for_stack = RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1)
base_xgb_for_stack = XGBClassifier(**xgb_params, random_state=42, n_jobs=-1, eval_metric='logloss')

model_rf = RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1)
model_xgb = XGBClassifier(**xgb_params, random_state=42, n_jobs=-1, eval_metric='logloss')
model_stacking = StackingClassifier(
    estimators=[('rf', base_rf_for_stack), ('xgb', base_xgb_for_stack)],
    final_estimator=LogisticRegression(C=final_C, random_state=42),
    cv=5, 
    stack_method='predict_proba'
)

models = {
    'RandomForest': model_rf,
    'XGBoost': model_xgb,
    'Stacking': model_stacking
}

# ==================== 5. 空间交叉验证循环 ====================
gkf = GroupKFold(n_splits=5)
results = {}

for name, model in models.items():
    print(f"\n========== 正在执行 {name} 的空间交叉验证 ==========")
    auc_list, f1_list, acc_list = [], [], []
    fold_idx = 0
    
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 训练
        model.fit(X_train, y_train)
        
        # 预测概率和类别
        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        
        # 计算指标
        auc = roc_auc_score(y_val, y_proba)
        f1 = f1_score(y_val, y_pred)
        acc = accuracy_score(y_val, y_pred)
        
        auc_list.append(auc)
        f1_list.append(f1)
        acc_list.append(acc)
        
        print(f"  Fold {fold_idx+1}: AUC={auc:.4f}, F1={f1:.4f}, ACC={acc:.4f}")
        fold_idx += 1
    
    # 保存结果
    results[name] = {
        'AUC': (np.mean(auc_list), np.std(auc_list)),
        'F1': (np.mean(f1_list), np.std(f1_list)),
        'ACC': (np.mean(acc_list), np.std(acc_list))
    }
    print(f"  ---> {name} 均值 ± 标准差: AUC = {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")

# ==================== 6. 输出综合对比表格 ====================

print(f"{'模型':<15} {'AUC均值':<10} {'AUC标准差':<10} {'F1均值':<10} {'ACC均值':<10}")
print("-" * 60)
for name, metrics in results.items():
    auc_mean, auc_std = metrics['AUC']
    f1_mean, _ = metrics['F1']
    acc_mean, _ = metrics['ACC']
    print(f"{name:<15} {auc_mean:.4f}     {auc_std:.4f}     {f1_mean:.4f}     {acc_mean:.4f}")

# ==================== 7. 保存最终模型  ====================
print("\n正在基于全部数据重新训练最终模型..")
final_rf = RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1)
final_xgb = XGBClassifier(**xgb_params, random_state=42, n_jobs=-1, eval_metric='logloss')

final_rf.fit(X, y)
final_xgb.fit(X, y)

joblib.dump(final_rf, f"{output_dir}/final_model_RF.pkl")
joblib.dump(final_xgb, f"{output_dir}/final_model_XGB.pkl")
print("最终模型（RF 和 XGB）已保存")
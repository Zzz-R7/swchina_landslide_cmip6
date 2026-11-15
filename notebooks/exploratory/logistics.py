# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 23:14:39 2026

@author: ZHENGrx
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# ======================== 1. 读取数据 ========================
# 滑坡点数据
df_landslide = pd.read_csv(r'D:/AAAAllllll/htpro/data_xinan/sample/landslide_precip_factors.csv')
# 非滑坡点数据
df_non = pd.read_csv(r'D:/AAAAllllll/htpro/data_xinan/sample/non_landslide_precip_factors.csv')

# 只需三个降水因子
factor_cols = ['Rx1day_mm', 'R20mm_天数', '极端降水贡献率']

# 校验列完整性（修复引号缺失bug）
for col in factor_cols:
    if col not in df_non.columns:
        raise ValueError(f"非滑坡数据缺少列: {col}")

# 打标签：滑坡=1，非滑坡=0
df_landslide['Label'] = 1
df_non['Label'] = 0

# 合并数据集（修正变量名df_landslide）
df_all = pd.concat([df_landslide[factor_cols + ['Label']], 
                    df_non[factor_cols + ['Label']]], axis=0, ignore_index=True)

print(f"合并后总样本数: {len(df_all)}")
print(f"滑坡样本: {df_all['Label'].sum()}, 非滑坡样本: {len(df_all) - df_all['Label'].sum()}")

# ======================== 2. 划分训练/测试集 ========================
X = df_all[factor_cols]
y = df_all['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ======================== 3. 标准化 ========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ======================== 4. 逻辑回归建模 ========================
model = LogisticRegression(max_iter=1000, penalty='l2', C=1.0, random_state=42)
model.fit(X_train_scaled, y_train)

# 预测结果
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n========== 模型性能指标 ==========")
print(f"测试集准确率: {model.score(X_test_scaled, y_test):.3f}")
print(f"AUC值: {roc_auc_score(y_test, y_prob):.3f}")
print("\n分类评估报告:\n", classification_report(y_test, y_pred, target_names=['非滑坡', '滑坡']))

# ======================== 5. 标准化系数 & 优势比 ========================
coef = model.coef_[0]
odds_ratio = np.exp(coef)

coef_df = pd.DataFrame({
    '因子': factor_cols,
    '标准化系数': coef,
    '优势比(OR)': odds_ratio,
    '影响方向': ['正向促进滑坡' if c > 0 else '负向抑制滑坡' for c in coef]
})
print("\n========== 各降水因子影响权重 ==========")
print(coef_df)
print("说明：标准化系数绝对值越大，因子对滑坡发生的影响越强；OR>1代表该指标升高会提高滑坡发生概率")
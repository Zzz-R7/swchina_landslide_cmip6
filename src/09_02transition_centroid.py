# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 09:39:45 2026

@author: ZHENGrx
"""

# -*- coding: utf-8 -*-
"""
计算高易发区面积与重心
"""
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

# ==================== 配置区 ====================
# 易发性栅格根目录
SUSC_BASE = r"E:/CMIP6"
OUTPUT_DIR = r"E:/CMIP6_analysis/Area"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = ["ACCESS-CM2", "NorESM2-LM", "EC-Earth3-Veg", "INM-CM4-8", "KACE-1-0-G"]
SCENARIOS = ["SSP126", "SSP245", "SSP585"]
PERIODS = ["2021-2040", "2041-2060", "2061-2080", "2081-2100"]

# 阈值
THRESHOLDS = {"high_and_above": 0.5, "very_high": 0.7}


def get_susc_path(model, ssp, period):
   
    return f"{SUSC_BASE}/{model}/{ssp}/susceptibility/susceptibility_{period}.tif"


def compute_area_centroid(susc_path, threshold):
    """分块计算高易发区面积(km²)和重心坐标"""
    with rasterio.open(susc_path) as src:
        transform = src.transform
        pixel_area_m2 = abs(transform[0] * transform[4])
        nodata = src.nodata
        total_pixels = 0
        sum_x = 0.0
        sum_y = 0.0

        block_size = 2048
        for row_start in range(0, src.height, block_size):
            for col_start in range(0, src.width, block_size):
                win_h = min(block_size, src.height - row_start)
                win_w = min(block_size, src.width - col_start)
                window = Window(col_start, row_start, win_w, win_h)
                data = src.read(1, window=window)

                mask = data >= threshold
                if nodata is not None:
                    mask &= (data != nodata)
                mask &= ~np.isnan(data)

                rows, cols = np.where(mask)
                if len(rows) == 0:
                    continue

                global_rows = rows + row_start
                global_cols = cols + col_start
                xs = transform[2] + (global_cols + 0.5) * transform[0]
                ys = transform[5] + (global_rows + 0.5) * transform[4]

                total_pixels += len(rows)
                sum_x += xs.sum()
                sum_y += ys.sum()

        if total_pixels == 0:
            return 0.0, np.nan, np.nan

        area_km2 = total_pixels * pixel_area_m2 / 1e6
        centroid_x = sum_x / total_pixels
        centroid_y = sum_y / total_pixels
        return area_km2, centroid_x, centroid_y


# ==================== 主循环 ====================
results = []

for model in MODELS:
    for ssp in SCENARIOS:
        for period in PERIODS:
            susc_path = get_susc_path(model, ssp, period)
            if not os.path.exists(susc_path):
                print(f" 不存在: {susc_path}")
                continue

            print(f"处理: {model} | {ssp} | {period}")
            for label, thr in THRESHOLDS.items():
                area, cx, cy = compute_area_centroid(susc_path, thr)
                results.append({
                    "Model": model,
                    "Scenario": ssp,
                    "Period": period,
                    "Threshold": label,
                    "Threshold_value": thr,
                    "Area_km2": area,
                    "Centroid_X": cx,
                    "Centroid_Y": cy,
                })

df = pd.DataFrame(results)
df.to_csv(f"{OUTPUT_DIR}/area_centroid.csv", index=False, encoding='utf-8-sig')
print(f"\n 面积与重心结果已保存: {OUTPUT_DIR}/area_centroid.csv")
print(df.head(10))
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.features import geometry_mask
import geopandas as gpd

# ==================== Configuration ====================
BASE_DIR = r"E:\CMIP6"
DELTA_DIR = r"E:/CMIP6_analysis/delta_sus"
SHP_PATH = r"D:\AAAAllllll\htpro\data_xinan\四省边界.shp"
OUTPUT_DIR = r"E:/CMIP6_analysis/Hotspots_Tiered"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = ["ACCESS-CM2", "NorESM2-LM", "EC-Earth3-Veg", "INM-CM4-8", "KACE-1-0-G"]
SSP_FUTURE = "SSP585"          
PERIOD_FUT = "2081-2100"
PERIOD_BASE = "2021-2040"      

BLOCK_SIZE = 2048

PROV_MAP = {
    '四川省': 'Sichuan', '四川': 'Sichuan',
    '云南省': 'Yunnan', '云南': 'Yunnan',
    '贵州省': 'Guizhou', '贵州': 'Guizhou',
    '重庆市': 'Chongqing', '重庆': 'Chongqing'
}

# ==================== 1. 计算 TIF ====================
print("1. Calculating tiered hotspot grids (3-4-5 scheme)...")

prob_paths = [os.path.join(BASE_DIR, m, SSP_FUTURE, "susceptibility", f"susceptibility_{PERIOD_FUT}.tif") for m in MODELS]
delta_paths = [f"{DELTA_DIR}/{m}_{SSP_FUTURE}_{PERIOD_FUT}_vs_{PERIOD_BASE}.tif" for m in MODELS]

with rasterio.open(prob_paths[0]) as src:
    H, W = src.height, src.width
    profile = src.profile.copy()
    profile.update(dtype='uint8', count=1, nodata=255, compress='LZW')
    transform = src.transform
    crs = src.crs

prob_srcs = [rasterio.open(p) for p in prob_paths]
delta_srcs = [rasterio.open(p) for p in delta_paths]
out_path = f"{OUTPUT_DIR}/hotspot_tiers.tif"
out = rasterio.open(out_path, 'w', **profile)

try:
    for r0 in range(0, H, BLOCK_SIZE):
        for c0 in range(0, W, BLOCK_SIZE):
            h, w = min(BLOCK_SIZE, H - r0), min(BLOCK_SIZE, W - c0)
            win = Window(c0, r0, w, h)

            probs = np.stack([s.read(1, window=win).astype(np.float32) for s in prob_srcs], axis=0)
            deltas = np.stack([s.read(1, window=win).astype(np.float32) for s in delta_srcs], axis=0)

            valid = np.all((probs >= 0) & (probs <= 1), axis=0) & \
                    np.all(np.isfinite(deltas) & (deltas > -9000) & (deltas < 100), axis=0)

            # 计算满足条件的模式数量
            agree_05 = np.sum(probs >= 0.5, axis=0)
            agree_07 = np.sum(probs >= 0.7, axis=0)
            delta_mean = np.mean(deltas, axis=0)

            # 3-4-5 递进式分类
            mask_t3 = (agree_07 == 5) & (delta_mean > 0.05) & valid
            mask_t2 = (agree_07 >= 4) & (delta_mean > 0.02) & valid & ~mask_t3
            mask_t1 = (agree_05 >= 3) & (delta_mean > 0.01) & valid & ~mask_t2 & ~mask_t3

            out_arr = np.full((h, w), 255, dtype=np.uint8)
            out_arr[valid] = 0
            out_arr[mask_t1] = 1
            out_arr[mask_t2] = 2
            out_arr[mask_t3] = 3
            out.write(out_arr, 1, window=win)
finally:
    out.close()
    for s in prob_srcs + delta_srcs: s.close()

pixel_area = abs(transform.a * transform.e) / 1e6
print(f"Done! TIF saved at: {out_path}")

# ==================== 2. 统计并覆盖 CSV ====================
print("2. Calculating provincial statistics...")
with rasterio.open(out_path) as src:
    tier_arr = src.read(1)

gdf_prov = gpd.read_file(SHP_PATH).to_crs(crs)
# 尝试寻找名称字段
prov_field = [c for c in gdf_prov.columns if c.upper() in ['NAME', 'PROVINCE', 'PR_NAME', 'NAME_1']]
if not prov_field:
    prov_field = [c for c in gdf_prov.columns if gdf_prov[c].dtype == 'object']
prov_field = prov_field[0]

records = []
for _, row in gdf_prov.iterrows():
    raw_name = str(row[prov_field])
    eng_name = PROV_MAP.get(raw_name, raw_name) 
    
    mask_prov = geometry_mask([row.geometry], out_shape=(H, W), transform=transform, invert=True)
    
    area_t1 = np.sum(mask_prov & (tier_arr == 1)) * pixel_area
    area_t2 = np.sum(mask_prov & (tier_arr == 2)) * pixel_area
    area_t3 = np.sum(mask_prov & (tier_arr == 3)) * pixel_area
    total_hot = area_t1 + area_t2 + area_t3
    
    records.append({
        "Province": eng_name,
        "Tier1_Broad_km2": area_t1,
        "Tier2_HighConf_km2": area_t2,
        "Tier3_Core_km2": area_t3,
        "Total_Hotspot_km2": total_hot
    })

df_prov = pd.DataFrame(records)
df_prov = df_prov.sort_values("Total_Hotspot_km2", ascending=False)
csv_out = f"{OUTPUT_DIR}/Provincial_Stats.csv"
df_prov.to_csv(csv_out, index=False, encoding='utf-8-sig')
print(f"CSV Saved and overwritten at: {csv_out}")
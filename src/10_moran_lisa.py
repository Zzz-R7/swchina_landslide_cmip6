# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from affine import Affine
import libpysal
from libpysal.weights import lat2W, w_subset


import sys, types, importlib.util

for _m in list(sys.modules.keys()):
    if _m == 'esda' or _m.startswith('esda.'):
        del sys.modules[_m]

_spec = importlib.util.find_spec('esda')
_fake_esda = types.ModuleType('esda')
_fake_esda.__path__ = _spec.submodule_search_locations
_fake_esda.__spec__ = _spec
sys.modules['esda'] = _fake_esda

from esda.moran import Moran, Moran_Local
# ============================================================

# ==================== 配置 ====================
INPUT_DIR = r"E:/CMIP6_analysis/Ensemble_Susc"
OUTPUT_DIR = r"E:/CMIP6_analysis/MoranI"
LISA_TIF_DIR = f"{OUTPUT_DIR}/lisa_tif"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LISA_TIF_DIR, exist_ok=True)

SCENARIOS = ["SSP126", "SSP245", "SSP585"]
PERIODS = ["2021-2040", "2041-2060", "2061-2080", "2081-2100"]

SCALE_FACTOR = 50    
PERMUTATIONS = 199   


# ==================== 读数据 ====================
def read_downsampled(path, scale):

    with rasterio.open(path) as src:
        H, W = src.height, src.width
        new_H = max(1, H // scale)
        new_W = max(1, W // scale)
        data = src.read(
            1, out_shape=(new_H, new_W),
            resampling=Resampling.average
        ).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        # 有效范围 [0, 1]
        data[(data < 0) | (data > 1)] = np.nan

        x_scale = W / new_W
        y_scale = H / new_H
        new_transform = src.transform * Affine.scale(x_scale, y_scale)
        return data, new_transform, src.crs


# ==================== 主循环 ====================
global_results = []
lisa_area_results = []

t_start = time.time()

for ssp in SCENARIOS:
    print(f"\n{'='*60}\n情景: {ssp}\n{'='*60}", flush=True)

    ref_path = f"{INPUT_DIR}/Mean_{ssp}_{PERIODS[0]}.tif"
    if not os.path.exists(ref_path):
        print(f"缺失: {ref_path}")
        continue

    w_built = None

    for period in PERIODS:
        t0 = time.time()
        path = f"{INPUT_DIR}/Mean_{ssp}_{period}.tif"
        if not os.path.exists(path):
            print(f"  缺失: {path}")
            continue

        data, new_transform, crs = read_downsampled(path, SCALE_FACTOR)
        H, W = data.shape
        flat = data.flatten()
        valid_mask = ~np.isnan(flat)
        valid_idx = np.where(valid_mask)[0]
        values = flat[valid_idx]

        if len(values) < 100:
            print(f"  {period}: 有效像元太少 ({len(values)})")
            continue

        # 权重构建（仅第一次）
        if w_built is None:
            print(f"  构建权重矩阵 ({H}×{W}) ...", flush=True)
            w_full = lat2W(H, W)
            w_built = w_subset(w_full, valid_idx)
            w_built.transform = 'r'
            print(f"    邻居关系: {w_built.n} 像元, "
                  f"平均邻居数 {w_built.mean_neighbors:.2f}", flush=True)

        # 全局 Moran's I
        mi = Moran(values, w_built, permutations=PERMUTATIONS)
        z = (mi.I - mi.EI) / mi.seI_norm if mi.seI_norm > 0 else 0.0

        print(f"  {period}: Moran's I = {mi.I:.4f}, "
              f"p = {mi.p_sim:.4f}, z = {z:.2f}, "
              f"耗时 {time.time()-t0:.1f}s", flush=True)

        global_results.append({
            "SSP": ssp, "Period": period,
            "MoranI": mi.I, "E[I]": mi.EI,
            "z_score": z, "p_value": mi.p_sim,
        })

        # 局部 LISA
        lisa = Moran_Local(values, w_built, permutations=PERMUTATIONS, seed=42)

        sig = lisa.p_sim < 0.05
        q = lisa.q
        lisa_class = np.zeros(len(values), dtype=np.int8)   # 0 = NS
        lisa_class[sig & (q == 1)] = 1   # HH
        lisa_class[sig & (q == 3)] = 2   # LL
        lisa_class[sig & (q == 2)] = 3   # LH
        lisa_class[sig & (q == 4)] = 4   # HL

        # ============ 写 LISA 分类栅格（uint8） ============
        lisa_2d = np.full(H * W, 255, dtype=np.uint8)
        lisa_2d[valid_idx] = lisa_class.astype(np.uint8)
        lisa_2d = lisa_2d.reshape(H, W)

        lisa_tif_path = f"{LISA_TIF_DIR}/LISA_{ssp}_{period}.tif"
        with rasterio.open(
            lisa_tif_path, 'w',
            driver='GTiff',
            height=H, width=W,
            count=1, dtype='uint8',
            crs=crs,
            transform=new_transform,
            nodata=255,
            compress='LZW',
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            dst.write(lisa_2d, 1)
            dst.update_tags(
                CATEGORY_0='NS (Not Significant)',
                CATEGORY_1='HH (High-High)',
                CATEGORY_2='LL (Low-Low)',
                CATEGORY_3='LH (Low-High)',
                CATEGORY_4='HL (High-Low)',
                CATEGORY_255='NoData',
                MORAN_I=f"{mi.I:.4f}",
                P_VALUE=f"{mi.p_sim:.4f}",
            )
        print(f" 已保存: {lisa_tif_path}")

        # 统计各类型占比
        total = len(values)
        n_hh = int((lisa_class == 1).sum())
        n_ll = int((lisa_class == 2).sum())
        n_lh = int((lisa_class == 3).sum())
        n_hl = int((lisa_class == 4).sum())
        n_sig = n_hh + n_ll + n_lh + n_hl

        lisa_area_results.append({
            "SSP": ssp, "Period": period,
            "HH_pct": n_hh / total * 100,
            "LL_pct": n_ll / total * 100,
            "LH_pct": n_lh / total * 100,
            "HL_pct": n_hl / total * 100,
            "Sig_pct": n_sig / total * 100,
            "NS_pct": (total - n_sig) / total * 100,
        })


# ==================== 保存表 ====================
df_global = pd.DataFrame(global_results)
df_global.to_csv(f"{OUTPUT_DIR}/moran_global.csv",
                 index=False, encoding='utf-8-sig')

df_lisa = pd.DataFrame(lisa_area_results)
df_lisa.to_csv(f"{OUTPUT_DIR}/lisa_stats.csv",
               index=False, encoding='utf-8-sig')

print(f"\n{'='*60}")
print(" 全局 Moran's I 汇总:")
print(df_global.to_string(index=False))
print(f"\n LISA 聚类面积占比:")
print(df_lisa.to_string(index=False))
print(f"\n 完成，用时 {(time.time()-t_start)/60:.1f} 分钟")
print(f"输出目录: {LISA_TIF_DIR}")
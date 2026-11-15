# -*- coding: utf-8 -*-
"""
不确定性量化
 逐像元计算 GCM 与 SSP 贡献率
 区域ANOVA验证

输出:
  E:/CMIP6_analysis/uncertainty/GCM_pct_{period}.tif  ×4
  E:/CMIP6_analysis/uncertainty/SSP_pct_{period}.tif  ×4
  E:/CMIP6_analysis/uncertainty/ANOVA_results.csv
"""
import os
import time
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from scipy.stats import f as f_dist

# ==================== 配置 ====================
SUSC_BASE = r"E:/CMIP6"
OUTPUT_DIR = r"E:/CMIP6_analysis/uncertainty"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = ["ACCESS-CM2", "NorESM2-LM", "EC-Earth3-Veg", "INM-CM4-8", "KACE-1-0-G"]
SCENARIOS = ["SSP126", "SSP245", "SSP585"]
PERIODS = ["2041-2060", "2061-2080", "2081-2100"]

BLOCK_SIZE = 2048   # 内存吃紧可改 1024

# 子区域（经纬度，WGS84），代码自动转 Albers
SUBREGIONS_LONLAT = {
    "川西高原": (97.0, 28.0, 103.0, 34.0),
    "四川盆地": (103.0, 28.0, 107.0, 32.0),
    "重庆":    (105.3, 28.2, 110.2, 32.2),
    "云贵高原": (102.0, 23.0, 108.0, 28.0),
    "滇南":    (99.0, 21.0, 104.0, 24.0),
}

TEST_ONLY = False   

def get_path(model, ssp, period):
    return f"{SUSC_BASE}/{model}/{ssp}/susceptibility/susceptibility_{period}.tif"


def lonlat_to_window(src, lon_min, lat_min, lon_max, lat_max):
    """经纬度边界 → 栅格像元 Window"""
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    x0, y0 = transformer.transform(lon_min, lat_min)
    x1, y1 = transformer.transform(lon_max, lat_max)
    left, right = min(x0, x1), max(x0, x1)
    bottom, top = min(y0, y1), max(y0, y1)

    try:
        r0, c0 = src.index(left, top)
        r1, c1 = src.index(right, bottom)
    except Exception as e:
        return None, (left, bottom, right, top)

    r0, r1 = sorted([max(0, min(r0, src.height)), max(0, min(r1, src.height))])
    c0, c1 = sorted([max(0, min(c0, src.width)), max(0, min(c1, src.width))])
    if r1 <= r0 or c1 <= c0:
        return None, (left, bottom, right, top)
    return Window(c0, r0, c1 - c0, r1 - r0), (left, bottom, right, top)


# ==================== 探测栅格信息 ====================
probe_path = get_path(MODELS[0], SCENARIOS[0], PERIODS[0])
print(f"探测栅格: {probe_path}")
with rasterio.open(probe_path) as src:
    print(f"  CRS    : {src.crs}")
    print(f"  Shape  : {src.height} × {src.width}  ({src.height*src.width/1e6:.1f}M 像元)")
    print(f"  dtype  : {src.dtypes[0]}")
    print(f"  nodata : {src.nodata}")
    print(f"  bounds : {src.bounds}")

print("\n子区域窗口预览（经纬度 → Albers 米 → 像元窗口）:")
with rasterio.open(probe_path) as src:
    for name, (a, b, c, d) in SUBREGIONS_LONLAT.items():
        win, box = lonlat_to_window(src, a, b, c, d)
        if win is None:
            print(f"  {name}: 范围无效或超出栅格 (Albers框: {box})")
        else:
            n_pix = int(win.height) * int(win.width)
            print(f"  {name:8s}: 窗口 (行{int(win.row_off)}:{int(win.row_off+win.height)}, "
                  f"列{int(win.col_off)}:{int(win.col_off+win.width)})  "
                  f"≈ {n_pix/1e6:.1f}M 像元")

if TEST_ONLY:
    print("\n TEST_ONLY=True，仅做探测。确认坐标无误后改成 False。")
    raise SystemExit



def compute_contribution(ds, period):
    print(f"\n {period}", flush=True)
    t0 = time.time()

    ref = ds[(MODELS[0], SCENARIOS[0])]
    H, W = ref.height, ref.width
    nodata = ref.nodata

    profile = {
        'driver': 'GTiff', 'height': H, 'width': W,
        'count': 1, 'dtype': 'float32',
        'crs': ref.crs, 'transform': ref.transform,
        'compress': 'LZW', 'tiled': True,
        'blockxsize': 512, 'blockysize': 512,
        'nodata': -9999.0,
    }
    out_gcm = rasterio.open(f"{OUTPUT_DIR}/GCM_pct_{period}.tif", 'w', **profile)
    out_ssp = rasterio.open(f"{OUTPUT_DIR}/SSP_pct_{period}.tif", 'w', **profile)

    n_rows = (H + BLOCK_SIZE - 1) // BLOCK_SIZE
    n_cols = (W + BLOCK_SIZE - 1) // BLOCK_SIZE
    n_blocks = n_rows * n_cols
    idx = 0

    for r0 in range(0, H, BLOCK_SIZE):
        for c0 in range(0, W, BLOCK_SIZE):
            idx += 1
            h = min(BLOCK_SIZE, H - r0)
            w = min(BLOCK_SIZE, W - c0)
            win = Window(c0, r0, w, h)

            stack = np.empty((5, 3, h, w), dtype=np.float32)
            valid = np.ones((h, w), dtype=bool)
            for i, m in enumerate(MODELS):
                for j, s in enumerate(SCENARIOS):
                    arr = ds[(m, s)].read(1, window=win).astype(np.float32)
                    if nodata is not None:
                        bad = (arr == nodata) | np.isnan(arr)
                    else:
                        bad = np.isnan(arr)
                    valid &= ~bad
                    stack[i, j] = np.where(bad, np.nan, arr)

            gcm_blk = np.full((h, w), -9999.0, dtype=np.float32)
            ssp_blk = np.full((h, w), -9999.0, dtype=np.float32)

            if valid.any():
                s = stack[:, :, valid]                # (5, 3, n)
                gcm_var = np.var(s, axis=0, ddof=1)   # 固定 SSP，沿 GCM 方差
                gcm_c = np.mean(gcm_var, axis=0)
                ssp_var = np.var(s, axis=1, ddof=1)   # 固定 GCM，沿 SSP 方差
                ssp_c = np.mean(ssp_var, axis=0)
                total = gcm_c + ssp_c
                with np.errstate(divide='ignore', invalid='ignore'):
                    gp = np.where(total > 0, gcm_c / total * 100, -9999.0)
                    sp = np.where(total > 0, ssp_c / total * 100, -9999.0)
                gcm_blk[valid] = gp.astype(np.float32)
                ssp_blk[valid] = sp.astype(np.float32)

            out_gcm.write(gcm_blk, 1, window=win)
            out_ssp.write(ssp_blk, 1, window=win)

            if idx % 5 == 0 or idx == n_blocks:
                elapsed = time.time() - t0
                eta = elapsed / idx * (n_blocks - idx)
                print(f"  块 {idx}/{n_blocks} ({idx/n_blocks*100:.0f}%)  "
                      f"已用 {elapsed/60:.1f}min  预计剩余 {eta/60:.1f}min",
                      flush=True)

    out_gcm.close(); out_ssp.close()
    print(f"  {period} 完成，用时 {(time.time()-t0)/60:.1f}min", flush=True)


# ==================== 区域ANOVA ====================
def subregion_mean(ds, win, nodata):
    """分块读取子区域，返回 (5,3) 均值矩阵"""
    H, W = int(win.height), int(win.width)
    sums = np.zeros((5, 3), dtype=np.float64)
    counts = np.zeros((5, 3), dtype=np.int64)

    for r0 in range(0, H, BLOCK_SIZE):
        for c0 in range(0, W, BLOCK_SIZE):
            bh = min(BLOCK_SIZE, H - r0)
            bw = min(BLOCK_SIZE, W - c0)
            sub_win = Window(int(win.col_off) + c0,
                             int(win.row_off) + r0, bw, bh)

            block = np.empty((5, 3, bh, bw), dtype=np.float32)
            valid = np.ones((bh, bw), dtype=bool)
            for i, m in enumerate(MODELS):
                for j, s in enumerate(SCENARIOS):
                    arr = ds[(m, s)].read(1, window=sub_win).astype(np.float32)
                    if nodata is not None:
                        bad = (arr == nodata) | np.isnan(arr)
                    else:
                        bad = np.isnan(arr)
                    valid &= ~bad
                    block[i, j] = np.where(bad, np.nan, arr)

            if valid.any():
                v = block[:, :, valid]              # (5, 3, n)
                sums += np.nansum(v, axis=2)
                counts += valid.sum()

    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(counts > 0, sums / counts, np.nan)


# ========== 替换 run_anova 函数 ==========
def two_way_anova_no_rep(matrix):
    """
    无重复双因素方差分析（5 GCM × 3 SSP）
    matrix: (a, b) 二维数组，a=GCM数，b=SSP数
    返回: dict(GCM_F, GCM_p, SSP_F, SSP_p, df_GCM, df_SSP, df_err)
    
    数学上等价于 statsmodels 的:
        ols('value ~ C(GCM) + C(SSP)', data=df).fit()  +  anova_lm(typ=2)
    """
    a, b = matrix.shape
    grand_mean = np.nanmean(matrix)

    # 主效应 A（GCM）
    mean_A = np.nanmean(matrix, axis=1)          # (a,)
    SS_A = b * np.nansum((mean_A - grand_mean) ** 2)

    # 主效应 B（SSP）
    mean_B = np.nanmean(matrix, axis=0)          # (b,)
    SS_B = a * np.nansum((mean_B - grand_mean) ** 2)

    # 总平方和
    SS_T = np.nansum((matrix - grand_mean) ** 2)

    # 残差
    SS_E = SS_T - SS_A - SS_B
    if SS_E < 0:
        SS_E = 0.0

    # 自由度
    df_A = a - 1
    df_B = b - 1
    df_E = (a - 1) * (b - 1)

    # 均方
    MS_A = SS_A / df_A if df_A > 0 else np.nan
    MS_B = SS_B / df_B if df_B > 0 else np.nan
    MS_E = SS_E / df_E if df_E > 0 else np.nan

    # F 值
    if MS_E > 0:
        F_A = MS_A / MS_E
        F_B = MS_B / MS_E
        p_A = float(f_dist.sf(F_A, df_A, df_E))
        p_B = float(f_dist.sf(F_B, df_B, df_E))
    else:
        F_A = F_B = np.nan
        p_A = p_B = np.nan

    return {
        "GCM_F": float(F_A), "GCM_p": p_A,
        "SSP_F": float(F_B), "SSP_p": p_B,
        "df_GCM": df_A, "df_SSP": df_B, "df_err": df_E,
    }


def run_anova(ds, sub_name, lonlat_box, period, nodata):
    ref = ds[(MODELS[0], SCENARIOS[0])]
    win, _ = lonlat_to_window(ref, *lonlat_box)
    if win is None:
        print(f"   {sub_name}: 范围无效，跳过", flush=True)
        return None

    means = subregion_mean(ds, win, nodata)   # (5, 3)

    # 有效值不足就跳过
    valid_count = np.isfinite(means).sum()
    if valid_count < 8:
        print(f"   {sub_name}: 有效数据不足 ({valid_count})", flush=True)
        return None

    # 用子区域均值填补极少数缺失值
    if valid_count < means.size:
        fill = np.nanmean(means)
        means = np.where(np.isfinite(means), means, fill)

    res = two_way_anova_no_rep(means)

    print(f"  {sub_name:8s}: GCM F={res['GCM_F']:7.2f} p={res['GCM_p']:.3g} | "
          f"SSP F={res['SSP_F']:7.2f} p={res['SSP_p']:.3g}", flush=True)

    return {
        "Period": period, "Subregion": sub_name,
        "GCM_F": res["GCM_F"], "GCM_p": res["GCM_p"],
        "SSP_F": res["SSP_F"], "SSP_p": res["SSP_p"],
        "df_GCM": res["df_GCM"], "df_SSP": res["df_SSP"], "df_err": res["df_err"],
        "Mean_Susc": float(np.nanmean(means)),
    }


# ==================== 主循环 ====================
anova_results = []

for period in PERIODS:
    print(f"\n{'='*60}\n时段: {period}\n{'='*60}", flush=True)

    ds, ok = {}, True
    for m in MODELS:
        for s in SCENARIOS:
            p = get_path(m, s, period)
            if not os.path.exists(p):
                print(f" 缺失: {p}")
                ok = False
                break
            ds[(m, s)] = rasterio.open(p)
        if not ok:
            break
    if not ok:
        for d in ds.values(): d.close()
        continue

    compute_contribution(ds, period)

    print(f"\n[任务1.3] 区域ANOVA", flush=True)
    nodata = ds[(MODELS[0], SCENARIOS[0])].nodata
    for sub_name, box in SUBREGIONS_LONLAT.items():
        res = run_anova(ds, sub_name, box, period, nodata)
        if res:
            anova_results.append(res)

    for d in ds.values(): d.close()

if anova_results:
    df_anova = pd.DataFrame(anova_results)
    df_anova.to_csv(f"{OUTPUT_DIR}/ANOVA_results.csv",
                    index=False, encoding='utf-8-sig')
    print(f"\n ANOVA结果已保存: {OUTPUT_DIR}/ANOVA_results.csv")
    print(df_anova.to_string(index=False))

print(f"\n 输出目录: {OUTPUT_DIR}")
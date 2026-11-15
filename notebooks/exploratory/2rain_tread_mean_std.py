# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置区 ====================
BASE_DIR = r"E:/CMIP6"
OUTPUT_DIR = r"E:/CMIP6_analysis"  
os.makedirs(OUTPUT_DIR, exist_ok=True)


MODELS = ["ACCESS-CM2", "NorESM2-LM", "EC-Earth3-Veg", "INM-CM4-8", "KACE-1-0-G"]

SCENARIOS = ["SSP126", "SSP245", "SSP585"]

PERIODS = ["2021-2040", "2041-2060", "2061-2080", "2081-2100"]

TARGET_RES = 1000  

VARIABLES = ["Rx1day", "R20mm"]

# 基准时段
BASELINE_PERIOD = "2021-2040"


# ==================== 工具函数 ====================
def get_file_path(model, ssp, period, var):
    """根据目录结构生成文件路径：E:/CMIP6/{GCM}/{SSP}/{var}_{GCM}_{SSP}_{时段}_avg.tif"""
    return f"{BASE_DIR}/{model}/{ssp}/{var}_{model}_{ssp}_{period}_avg.tif"


def downsample_to_1km(src_path, dst_path):
    """将90m栅格降采样到1km，LZW压缩保存"""
    if os.path.exists(dst_path):
        return dst_path  # 已存在则跳过
    
    da = rxr.open_rasterio(src_path, masked=True)
    
    # 降采样（平均法，适合连续变量）
    da_1km = da.rio.reproject(
        da.rio.crs,
        resolution=TARGET_RES,
        resampling=5  # 5 = average
    )
    
    # 保存（LZW压缩 + Tiled）
    da_1km.rio.to_raster(
        dst_path,
        compress='LZW',
        tiled=True,
        blockxsize=512,
        blockysize=512,
        dtype='float32'
    )
    
    da.close()
    da_1km.close()
    return dst_path


# ==================== 第1步：降采样 90m → 1km ====================
print("=" * 60)
print("第1步：降采样 90m → 1km")
print("=" * 60)

downsampled_dir = f"{OUTPUT_DIR}/1km_rasters"
os.makedirs(downsampled_dir, exist_ok=True)

for model in MODELS:
    for ssp in SCENARIOS:
        for period in PERIODS:
            for var in VARIABLES:
                src = get_file_path(model, ssp, period, var)
                if not os.path.exists(src):
                    print(f"⚠️ 源文件不存在: {src}")
                    continue
                
                dst = f"{downsampled_dir}/{var}_{model}_{ssp}_{period}_1km.tif"
                downsample_to_1km(src, dst)
                print(f"  ✅ {var} | {model} | {ssp} | {period}")

print("降采样完成！\n")


# ==================== 第2步：提取区域平均值 ====================
print("=" * 60)
print("第2步：提取区域平均值")
print("=" * 60)

results_rx1day = []
results_r20mm = []

for model in MODELS:
    for ssp in SCENARIOS:
        for period in PERIODS:
            for var in VARIABLES:
                src_path = f"{downsampled_dir}/{var}_{model}_{ssp}_{period}_1km.tif"
                if not os.path.exists(src_path):
                    continue
                
                with rxr.open_rasterio(src_path, masked=True) as da:
                    # 区域平均值（排除NoData）
                    mean_val = float(da.mean().values)
                
                if var == "Rx1day":
                    results_rx1day.append({
                        "Model": model, "Scenario": ssp, "Period": period,
                        "Value": mean_val
                    })
                else:
                    results_r20mm.append({
                        "Model": model, "Scenario": ssp, "Period": period,
                        "Value": mean_val
                    })

df_rx1 = pd.DataFrame(results_rx1day)
df_r20 = pd.DataFrame(results_r20mm)

# 保存CSV
df_rx1.to_csv(f"{OUTPUT_DIR}/Rx1day_trend.csv", index=False)
df_r20.to_csv(f"{OUTPUT_DIR}/R20mm_trend.csv", index=False)

print(f"✅ Rx1day 数据: {len(df_rx1)} 条")
print(f"✅ R20mm 数据: {len(df_r20)} 条")
print(f"✅ CSV已保存至 {OUTPUT_DIR}\n")


# ==================== 第3步：绘制趋势图 ====================
print("=" * 60)
print("第3步：绘制趋势图")
print("=" * 60)

scenario_labels = {"SSP126": "SSP1-2.6", "SSP245": "SSP2-4.5", "SSP585": "SSP5-8.5"}
colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

for j, ssp in enumerate(SCENARIOS):
    # ========== 第一行：R20mm ==========
    ax = axes[0, j]
    for i, model in enumerate(MODELS):
        sub = df_r20[(df_r20["Model"] == model) & (df_r20["Scenario"] == ssp)]
        sub = sub.set_index("Period").reindex(PERIODS)
        if sub["Value"].notna().any():
            ax.plot(PERIODS, sub["Value"], marker='s',
                    label=model, color=colors[i], alpha=0.7, linewidth=1.5)

    mean_vals, std_vals = [], []
    for p in PERIODS:
        vals = df_r20[(df_r20["Scenario"] == ssp) & (df_r20["Period"] == p)]["Value"].values
        mean_vals.append(np.mean(vals))
        std_vals.append(np.std(vals))

    ax.plot(PERIODS, mean_vals, 'k-', linewidth=3, label='Ensemble Mean')
    ax.fill_between(PERIODS,
                    np.array(mean_vals) - np.array(std_vals),
                    np.array(mean_vals) + np.array(std_vals),
                    color='gray', alpha=0.2, label='±1 Std')
    ax.set_title(f"{subplot_labels[j]} R20mm - {scenario_labels[ssp]}", fontsize=15)
    ax.set_ylabel("R20mm (days)", fontsize=13)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(alpha=0.3)
    if j == 0:
        ax.legend(fontsize=11, loc='upper left')

    # ========== 第二行：Rx1day ==========
    ax = axes[1, j]
    for i, model in enumerate(MODELS):
        sub = df_rx1[(df_rx1["Model"] == model) & (df_rx1["Scenario"] == ssp)]
        sub = sub.set_index("Period").reindex(PERIODS)
        if sub["Value"].notna().any():
            ax.plot(PERIODS, sub["Value"], marker='o',
                    label=model, color=colors[i], alpha=0.7, linewidth=1.5)

    mean_vals, std_vals = [], []
    for p in PERIODS:
        vals = df_rx1[(df_rx1["Scenario"] == ssp) & (df_rx1["Period"] == p)]["Value"].values
        mean_vals.append(np.mean(vals))
        std_vals.append(np.std(vals))

    ax.plot(PERIODS, mean_vals, 'k-', linewidth=3, label='Ensemble Mean')
    ax.fill_between(PERIODS,
                    np.array(mean_vals) - np.array(std_vals),
                    np.array(mean_vals) + np.array(std_vals),
                    color='gray', alpha=0.2)
    ax.set_title(f"{subplot_labels[j + 3]} Rx1day - {scenario_labels[ssp]}", fontsize=15)
    ax.set_ylabel("Rx1day (mm)", fontsize=13)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/precipitation_trends.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ 趋势图已保存: {OUTPUT_DIR}/precipitation_trends.png\n")


# ==================== 第4步：趋势检验（Theil-Sen + 线性回归） ====================
print("=" * 60)
print("第4步：趋势检验")
print("=" * 60)

trend_results = []

for var_name, df in [("Rx1day", df_rx1), ("R20mm", df_r20)]:
    print(f"\n--- {var_name} ---")
    for ssp in SCENARIOS:
        for model in MODELS:
            sub = df[(df["Model"] == model) & (df["Scenario"] == ssp)]
            sub = sub.set_index("Period").reindex(PERIODS)
            vals = sub["Value"].dropna().values
            
            if len(vals) < 3:
                continue
            
            # 线性回归
            x = np.arange(len(vals))
            slope, intercept, r, p_value, std_err = stats.linregress(x, vals)
            
            # Theil-Sen 斜率
            sen_slope = stats.theilslopes(vals, x)[0]
            
            trend = "↑上升" if slope > 0 else "↓下降"
            sig = "显著" if p_value < 0.05 else "不显著"
            
            trend_results.append({
                "Variable": var_name, "Scenario": ssp, "Model": model,
                "Linear_Slope": slope, "TheilSen_Slope": sen_slope,
                "p_value": p_value, "Trend": trend, "Significance": sig
            })
            
            print(f"  {ssp} | {model}: 斜率={slope:.3f}/时段, "
                  f"p={p_value:.4f}, {trend}({sig})")

df_trend = pd.DataFrame(trend_results)
df_trend.to_csv(f"{OUTPUT_DIR}/trend_test_results.csv", index=False)
print(f"\n✅ 趋势检验结果已保存: {OUTPUT_DIR}/trend_test_results.csv\n")


# ==================== 第5步：计算ΔP（相对易发性变化） ====================
print("=" * 60)
print("第5步：计算ΔP（相对易发性变化）")
print("=" * 60)

delta_dir = f"{OUTPUT_DIR}/Delta_P"
os.makedirs(delta_dir, exist_ok=True)

def subtract_rasters_rio(path1, path2, output_path):
    """使用rioxarray分块相减，低内存，LZW压缩"""
    da1 = rxr.open_rasterio(path1, masked=True)
    da2 = rxr.open_rasterio(path2, masked=True)
    
    # 确保空间对齐
    da2 = da2.rio.reproject_match(da1)
    
    delta = da1 - da2
    delta.rio.to_raster(
        output_path,
        compress='LZW',
        tiled=True,
        blockxsize=512,
        blockysize=512,
        dtype='float32'
    )
    
    da1.close()
    da2.close()
    return output_path

# ========== 对比1：远期 vs 近期（2081-2100 vs 2021-2040） ==========
print("\n--- 对比1：2081-2100 vs 2021-2040 ---")
for model in MODELS:
    for ssp in SCENARIOS:
        src_future = f"{downsampled_dir}/Rx1day_{model}_{ssp}_2081-2100_1km.tif"
        src_baseline = f"{downsampled_dir}/Rx1day_{model}_{ssp}_{BASELINE_PERIOD}_1km.tif"
        
        if os.path.exists(src_future) and os.path.exists(src_baseline):
            dst = f"{delta_dir}/{model}_{ssp}_2081-2100_vs_2021-2040.tif"
            subtract_rasters_rio(src_future, src_baseline, dst)
            print(f"  ✅ {model} {ssp}")

# ========== 对比2：SSP5-8.5 vs SSP1-2.6（同一时段2081-2100） ==========
print("\n--- 对比2：SSP5-8.5 vs SSP1-2.6（2081-2100） ---")
for model in MODELS:
    src_585 = f"{downsampled_dir}/Rx1day_{model}_SSP585_2081-2100_1km.tif"
    src_126 = f"{downsampled_dir}/Rx1day_{model}_SSP126_2081-2100_1km.tif"
    
    if os.path.exists(src_585) and os.path.exists(src_126):
        dst = f"{delta_dir}/{model}_2081-2100_SSP585_vs_SSP126.tif"
        subtract_rasters_rio(src_585, src_126, dst)
        print(f"  ✅ {model}")

print(f"\n✅ 所有ΔP栅格已保存至: {delta_dir}\n")


# ==================== 第6步：多模式集合平均与标准差 ====================
print("=" * 60)
print("第6步：多模式集合平均与标准差")
print("=" * 60)

ensemble_dir = f"{OUTPUT_DIR}/Ensemble"
os.makedirs(ensemble_dir, exist_ok=True)

def ensemble_stats(file_paths, output_mean_path, output_std_path):
    """计算多个栅格的集合均值与标准差"""
    das = [rxr.open_rasterio(p, masked=True) for p in file_paths if os.path.exists(p)]
    if len(das) == 0:
        return
    
    # 对齐到第一个栅格
    ref = das[0]
    aligned = [ref] + [d.rio.reproject_match(ref) for d in das[1:]]
    
    # 堆叠计算
    stacked = xr.concat(aligned, dim='model')
    mean_da = stacked.mean(dim='model')
    std_da = stacked.std(dim='model')
    
    mean_da.rio.to_raster(output_mean_path, compress='LZW', tiled=True, dtype='float32')
    std_da.rio.to_raster(output_std_path, compress='LZW', tiled=True, dtype='float32')
    
    for d in das:
        d.close()

# 集合平均：以2081-2100 SSP585的ΔP为例
print("\n--- 集合平均：2081-2100 vs 2021-2040 (ΔP) ---")
for ssp in SCENARIOS:
    paths = []
    for model in MODELS:
        p = f"{delta_dir}/{model}_{ssp}_2081-2100_vs_2021-2040.tif"
        if os.path.exists(p):
            paths.append(p)
    
    if len(paths) > 0:
        mean_path = f"{ensemble_dir}/EnsembleMean_{ssp}_2081-2100_vs_2021-2040.tif"
        std_path = f"{ensemble_dir}/EnsembleStd_{ssp}_2081-2100_vs_2021-2040.tif"
        ensemble_stats(paths, mean_path, std_path)
        print(f"  ✅ {ssp} 集合平均完成")

# 集合平均：SSP585 vs SSP126
print("\n--- 集合平均：SSP585 vs SSP126 (2081-2100) ---")
paths = []
for model in MODELS:
    p = f"{delta_dir}/{model}_2081-2100_SSP585_vs_SSP126.tif"
    if os.path.exists(p):
        paths.append(p)

if len(paths) > 0:
    mean_path = f"{ensemble_dir}/EnsembleMean_2081-2100_SSP585_vs_SSP126.tif"
    std_path = f"{ensemble_dir}/EnsembleStd_2081-2100_SSP585_vs_SSP126.tif"
    ensemble_stats(paths, mean_path, std_path)
    print(f"  ✅ SSP585 vs SSP126 集合平均完成")


print("\n" + "=" * 60)
print("=" * 60)
print(f"\n输出目录: {OUTPUT_DIR}")
print(f"  ├── 1km_rasters/       : 1km降采样栅格")
print(f"  ├── Delta_P/           : ΔP栅格（相对易发性变化）")
print(f"  ├── Ensemble/          : 集合平均与标准差")
print(f"  ├── Rx1day_trend.csv   : Rx1day区域平均值")
print(f"  ├── R20mm_trend.csv    : R20mm区域平均值")
print(f"  ├── trend_test_results.csv : 趋势检验结果")
print(f"  └── precipitation_trends.png : 趋势图")
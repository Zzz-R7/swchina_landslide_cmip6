import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio import features
from rasterio.warp import reproject, Resampling
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 路径配置 ====================
output_dir = r"D:/AAAAllllll/htpro/data_xinan/sample/HPBSF"
iv_excel_path = os.path.join(output_dir, "IV_result_10.xlsx")
landslide_points = r"D:/AAAAllllll/htpro/data_xinan/sample/降雨滑坡点_带日期_Albers.shp"
buffer_radius = 500

def find_classified_rasters(output_dir):
    classified = {}
    for name in ["降雨", "高程", "坡度", "坡向", "平面曲率", "剖面曲率"]:
        path = os.path.join(output_dir, f"{name}_class_5cl.tif")
        if os.path.exists(path):
            classified[name] = path
    for name, fname in [("距道路", "距道路_class_dist.tif"),
                        ("距断层", "距断层_class_dist.tif"),
                        ("距河流", "距河流_class_dist_fixed.tif")]:
        path = os.path.join(output_dir, fname)
        if os.path.exists(path):
            classified[name] = path
    litho = os.path.join(output_dir, "lithology_90m.tif")
    if os.path.exists(litho):
        classified["岩性"] = litho
    return classified

classified_rasters = find_classified_rasters(output_dir)
print(f"找到 {len(classified_rasters)} 个分类栅格:")
for k, v in classified_rasters.items():
    print(f"  {k} -> {os.path.basename(v)}")

iv_dict = {}
with pd.ExcelFile(iv_excel_path) as xls:
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        iv_map = {}
        for _, row in df.iterrows():
            cls = row['类别']
            if cls != 0:
                iv_map[cls] = row['IV值']
        iv_dict[sheet] = iv_map
print("IV 映射读取完成")

base_name = "降雨"
base_path = classified_rasters[base_name]
with rasterio.open(base_path) as src:
    base_profile = src.profile.copy()
    base_transform = src.transform
    base_crs = src.crs
    base_height, base_width = src.height, src.width
print(f"基准栅格: {base_name}，尺寸 {base_height} x {base_width}")

iv_paths = []
for name, path in classified_rasters.items():
    if name not in iv_dict:
        print(f"跳过 {name}（无 IV 映射）")
        continue
    print(f"生成 IV 栅格: {name}")
    with rasterio.open(path) as src:
        data = src.read(1)
        iv_data = np.zeros((base_height, base_width), dtype=np.float32)
        reproject(
            source=data,
            destination=iv_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=base_transform,
            dst_crs=base_crs,
            resampling=Resampling.nearest
        )
    iv_mapped = np.zeros_like(iv_data, dtype=np.float32)
    for cls, val in iv_dict[name].items():
        iv_mapped[iv_data == cls] = val
    out_path = os.path.join(output_dir, f"{name}_IV_aligned.tif")
    prof = base_profile.copy()
    prof.update(dtype=rasterio.float32, count=1, nodata=0)
    with rasterio.open(out_path, 'w', **prof) as dst:
        dst.write(iv_mapped, 1)
    iv_paths.append(out_path)
    print(f"  {os.path.basename(out_path)}")

print("\n累加总 IV...")
total_iv = np.zeros((base_height, base_width), dtype=np.float32)
for p in iv_paths:
    with rasterio.open(p) as src:
        total_iv += src.read(1)

total_iv_path = os.path.join(output_dir, "total_IV_sum_aligned.tif")
with rasterio.open(iv_paths[0]) as src_ref:
    prof = src_ref.profile.copy()
    prof.update(dtype=rasterio.float32, nodata=0)
    with rasterio.open(total_iv_path, 'w', **prof) as dst:
        dst.write(total_iv, 1)
print(f"总 IV 图保存: {total_iv_path}")

print("\n总 IV 自然断点分 5 级...")
from sklearn.cluster import KMeans
valid = total_iv[total_iv != 0]
if len(valid) > 200000:
    idx = np.random.choice(len(valid), 200000, replace=False)
    sample = valid[idx]
else:
    sample = valid
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(sample.reshape(-1,1))
centers = np.sort(kmeans.cluster_centers_.flatten())
bins = [(centers[i]+centers[i+1])/2 for i in range(4)]
sus = np.digitize(total_iv, bins=bins, right=False)
sus[total_iv == 0] = 0

sus_path = os.path.join(output_dir, "total_susceptibility_5class_aligned.tif")
with rasterio.open(total_iv_path) as src:
    prof = src.profile.copy()
    prof.update(dtype=rasterio.int16, nodata=0)
    with rasterio.open(sus_path, 'w', **prof) as dst:
        dst.write(sus.astype(np.int16), 1)
print(f"易发性分级图保存，断点: {bins}")

candidate = (sus == 1) | (sus == 2)
cand_path = os.path.join(output_dir, "candidate_low_susceptibility_aligned.tif")
with rasterio.open(sus_path) as src:
    prof = src.profile.copy()
    prof.update(dtype=rasterio.int16, nodata=0)
    with rasterio.open(cand_path, 'w', **prof) as dst:
        dst.write(candidate.astype(np.int16), 1)
cand_count = np.sum(candidate)
total_valid = np.sum(total_iv != 0)
print(f"候选区像元数: {cand_count} / {total_valid} ({100*cand_count/total_valid:.2f}%)")

print("\n生成 500m 缓冲区...")
points = gpd.read_file(landslide_points, encoding='gbk')
if points.crs.is_geographic:
    points = points.to_crs("EPSG:32649")
with rasterio.open(cand_path) as src:
    raster_crs = src.crs
if points.crs != raster_crs:
    points = points.to_crs(raster_crs)
buffer_gdf = points.copy()
buffer_gdf.geometry = buffer_gdf.buffer(buffer_radius)
buffer_shp = os.path.join(output_dir, "landslide_buffer_500m.shp")
buffer_gdf.to_file(buffer_shp)

# 缓冲区转栅格
with rasterio.open(cand_path) as src:
    transform = src.transform
    shape = (src.height, src.width)
buffer_raster = features.rasterize(
    ((geom, 1) for geom in buffer_gdf.geometry),
    out_shape=shape,
    transform=transform,
    fill=0,
    dtype=np.int16
    # crs 参数已移除
)
buf_raster_path = os.path.join(output_dir, "buffer_mask.tif")
with rasterio.open(cand_path) as src:
    prof = src.profile.copy()
    with rasterio.open(buf_raster_path, 'w', **prof) as dst:
        dst.write(buffer_raster, 1)

# 排除缓冲区
final = candidate & (buffer_raster == 0)
final_path = os.path.join(output_dir, "final_candidate_low_without_buffer_aligned.tif")
with rasterio.open(cand_path) as src:
    prof = src.profile.copy()
    with rasterio.open(final_path, 'w', **prof) as dst:
        dst.write(final.astype(np.int16), 1)
final_count = np.sum(final)
print(f"最终候选区像元数: {final_count}")

# 随机采样
ys, xs = np.where(final == 1)
if len(ys) == 0:
    raise ValueError("候选区为空！")
n = len(points)
if len(ys) < n:
    print(f"候选区像元数 ({len(ys)}) 少于采样数 ({n})，全部采用。")
    idx = np.arange(len(ys))
else:
    idx = np.random.choice(len(ys), n, replace=False)
sample_ys, sample_xs = ys[idx], xs[idx]
with rasterio.open(final_path) as src:
    transform = src.transform
    coords = [rasterio.transform.xy(transform, y, x) for y, x in zip(sample_ys, sample_xs)]
sample_gdf = gpd.GeoDataFrame(
    {'id': range(len(coords)), 'geometry': gpd.points_from_xy([c[0] for c in coords], [c[1] for c in coords])},
    crs=points.crs
)
sample_gdf = sample_gdf.to_crs(points.crs)
sample_path = os.path.join(output_dir, "non_landslide_samples_HPBSF.shp")
sample_gdf.to_file(sample_path)
print(f"负样本点已保存，共 {len(sample_gdf)} 个")

print("\n" + "="*60)
print("HPBSF 采样完成")
print("="*60)
print(f"输出目录: {output_dir}")
print(f"关键文件:")
print(f"  - total_IV_sum_aligned.tif")
print(f"  - total_susceptibility_5class_aligned.tif")
print(f"  - candidate_low_susceptibility_aligned.tif")
print(f"  - final_candidate_low_without_buffer_aligned.tif")
print(f"  - non_landslide_samples_HPBSF.shp  ")
print(f"\n统计:")
print(f"  有效像元数: {total_valid}")
print(f"  初始候选区占比: {100*cand_count/total_valid:.2f}%")
print(f"  最终候选区像元数: {final_count}")
print(f"  采样点数: {len(sample_gdf)}")
print("="*60)
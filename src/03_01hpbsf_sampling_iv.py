import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio import features
import mapclassify
import os
import gc

# ===================== 1. 路径配置 =====================
raster_list = {
    "降雨": r"D:/AAAAllllll/htpro/data_xinan/sample/alltif/rain.tif",
    "高程": r"D:/AAAAllllll/htpro/data_xinan/sample/alltif/dem.tif",
    "坡度": r"D:/AAAAllllll/htpro/data_xinan/sample/alltif/slope.tif",
    "坡向": r"D:/AAAAllllll/htpro/data_xinan/sample/alltif/aspect.tif",
    "平面曲率": r"D:/AAAAllllll/htpro/data_xinan/sample/alltif/plan.tif",
    "剖面曲率": r"D:/AAAAllllll/htpro/data_xinan/sample/alltif/profile.tif",
}

distance_raster_list = {
    "距道路": r"D:/AAAAllllll/htpro/data_xinan/sample/alltif/road.tif",
    "距断层": r"D:/AAAAllllll/htpro/data_xinan/sample/alltif/dc.tif",
    "距河流": r"D:/AAAAllllll/htpro/data_xinan/sample/alltif/stream.tif",
}

litho_vector = r"D:/AAAAllllll/htpro/data_xinan/yan/xinan_lithology_classified.shp"
landslide_points = r"D:/AAAAllllll/htpro/data_xinan/sample/降雨滑坡点_带日期_Albers.shp"
output_dir = r"D:/AAAAllllll/htpro/data_xinan/sample/HPBSF"

# ===================== 2. 参数 =====================
NUM_CLASSES = 5
SAMPLE_SIZE = 200000
code_field = "CODE"

os.makedirs(output_dir, exist_ok=True)
classified_rasters = {}

# 用于保存参考栅格信息的变量（从第一个因子获取）
ref_profile = None
ref_transform = None
ref_crs = None
ref_height = None
ref_width = None

# ===================== 3. 处理连续因子 =====================
for name, path in raster_list.items():
    print(f"正在处理连续因子: {name}")
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
        profile = src.profile
        nodata = src.nodata if src.nodata is not None else -9999

        # 如果是第一个因子，缓存参考信息
        if ref_profile is None:
            ref_profile = profile.copy()
            ref_transform = src.transform
            ref_crs = src.crs
            ref_height = src.height
            ref_width = src.width

        valid_mask = (data != nodata) & np.isfinite(data)
        valid_data = data[valid_mask]

        if len(valid_data) == 0:
            print(f"  ⚠️ {name} 无有效数据，跳过")
            continue

        if len(valid_data) > SAMPLE_SIZE:
            sample_idx = np.random.choice(len(valid_data), size=SAMPLE_SIZE, replace=False)
            sample_data = valid_data[sample_idx]
        else:
            sample_data = valid_data

        if len(np.unique(sample_data)) < NUM_CLASSES:
            unique_vals = np.sort(np.unique(sample_data))
            if len(unique_vals) > 1:
                bins = unique_vals[:-1]
            else:
                bins = [unique_vals[0]]
        else:
            classifier = mapclassify.NaturalBreaks(sample_data, k=NUM_CLASSES)
            bins = classifier.bins

        classified = np.digitize(data, bins=bins, right=False)
        classified[~valid_mask] = 0

        out_path = os.path.join(output_dir, f"{name}_class_{NUM_CLASSES}cl.tif")
        profile.update(dtype=rasterio.int16, count=1, nodata=0)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(classified, 1)
        classified_rasters[name] = out_path
        print(f"  已保存: {os.path.basename(out_path)}，断点: {bins}")
        gc.collect()

# ===================== 4. 引用距离因子 =====================
for name, path in distance_raster_list.items():
    classified_rasters[name] = path
    print(f"已引用距离因子: {name}")

# ===================== 5. 岩性矢量转栅格（使用缓存的参考信息） =====================
print("正在转岩性矢量...")
gdf = gpd.read_file(litho_vector, encoding='gbk')
if code_field not in gdf.columns:
    raise ValueError(f"矢量文件中找不到字段 '{code_field}'")

# 使用之前缓存的参考信息
transform = ref_transform
crs = ref_crs
height, width = ref_height, ref_width

shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[code_field]))
litho_raster = features.rasterize(
    shapes=shapes,
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype=np.int16
)
litho_path = os.path.join(output_dir, "lithology_90m.tif")
# 从缓存的 profile 生成新 profile
profile_out = ref_profile.copy()
profile_out.update(dtype=rasterio.int16, count=1, nodata=0)
with rasterio.open(litho_path, 'w', **profile_out) as dst:
    dst.write(litho_raster, 1)
classified_rasters["岩性"] = litho_path
print("✅ 岩性栅格保存完成")

# ===================== 6. 提取滑坡点分类值 =====================
print("提取滑坡点分类值...")
points = gpd.read_file(landslide_points)
points = points.to_crs(crs)
factor_data = {}

for name, path in classified_rasters.items():
    print(f"  提取: {name}")
    with rasterio.open(path) as src:
        coords = [(x, y) for x, y in zip(points.geometry.x, points.geometry.y)]
        values = [v[0] for v in src.sample(coords)]
        factor_data[name] = values

df_points = pd.DataFrame(factor_data)
print(f"✅ 已提取 {len(df_points)} 个滑坡点的因子值")

# ===================== 7. 计算 IV 值 =====================
print("计算 IV 值...")
total_pixels = height * width
total_landslides = len(points)
iv_results = {}

for factor in classified_rasters.keys():
    with rasterio.open(classified_rasters[factor]) as src:
        arr = src.read(1)
        unique_vals, counts = np.unique(arr[arr > 0], return_counts=True)
        total_counts = pd.Series(counts, index=unique_vals)

    counts_landslide = df_points[factor].value_counts().sort_index()
    all_classes = sorted(set(counts_landslide.index) | set(total_counts.index))
    iv_list = []

    for cls in all_classes:
        Ni = counts_landslide.get(cls, 0)
        Si = total_counts.get(cls, 0)
        if Ni == 0 or Si == 0:
            IV = 0.0
        else:
            IV = np.log((Ni / total_landslides) / (Si / total_pixels))
        iv_list.append({'类别': int(cls), '滑坡点数(N_i)': int(Ni), '像元数(S_i)': int(Si), 'IV值': round(IV, 6)})

    iv_results[factor] = iv_list

# ===================== 8. 输出 Excel =====================
excel_path = os.path.join(output_dir, "IV_计算结果.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    for factor, rows in iv_results.items():
        sheet_name = factor[:31]
        pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name, index=False)

print(f"\n 全部完成！")
print(f"分类栅格和IV结果均保存在: {output_dir}")
print(f"IV详细结果表: {excel_path}")



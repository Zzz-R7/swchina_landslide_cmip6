import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import Point
from statsmodels.stats.outliers_influence import variance_inflation_factor
# ------------------ 1. 读取滑坡点 ------------------
landslide_gdf = gpd.read_file(r'D:\AAAAllllll\htpro\data_xinan\xinan_landslidepoint_diy.shp')
print(f"滑坡点数量: {len(landslide_gdf)}")
# ------------------ 2. 生成随机非滑坡点------------------

study_area = gpd.read_file(r'D:\AAAAllllll\htpro\data_xinan\四省边界.shp')
minx, miny, maxx, maxy = study_area.total_bounds
crs = study_area.crs
np.random.seed(42)  # 固定随机种子
random_points = []
for _ in range(len(landslide_gdf)):
    x = np.random.uniform(minx, maxx)
    y = np.random.uniform(miny, maxy)
    random_points.append(Point(x, y))
non_landslide_gdf = gpd.GeoDataFrame(geometry=random_points, crs=crs)
# 合并
all_points = gpd.GeoDataFrame(pd.concat([landslide_gdf, non_landslide_gdf], ignore_index=True), crs=crs)
print(f"总样本点数: {len(all_points)}")
# ------------------ 3. 提取所有连续因子栅格值 ------------------
factor_files = {
    'elevation': r'D:\AAAAllllll\htpro\data_xinan\xinan_dem_cut_diy.tif',
    'slope': r'D:\AAAAllllll\htpro\data_xinan\xinan_slope.tif',
    'aspect': r'D:\AAAAllllll\htpro\data_xinan\xinan_aspect.tif',
    'river_dist': r'D:\AAAAllllll\htpro\data_xinan\水文\xinan_stream3dist.tif',
    'fault_dist': r'D:\AAAAllllll\htpro\data_xinan\yan\xinan_dc_albers_cuttif.tif',
    'road_dist': r'D:\AAAAllllll\htpro\data_xinan\road\xinan_roadist90_2cut.tif',
    'precipitation': r'D:\AAAAllllll\htpro\data_xinan\rain\xinan_rain17avg.tif',
    'profile_curv': r'D:\AAAAllllll\htpro\data_xinan\terrain_factors\xinan_profile_curvature.tif',
    'plane_curv': r'D:\AAAAllllll\htpro\data_xinan\terrain_factors\plan_curvature.tif',
    'ndvi': r'D:\AAAAllllll\htpro\data_xinan\NDVI\xinan_ndviavg_albers.tif'
}
def extract_raster_values(gdf, raster_path, col_name):
    values = []
    with rasterio.open(raster_path) as src:
        for geom in gdf.geometry:
            # 提取点所在像元值
            for val in src.sample([(geom.x, geom.y)]):
                values.append(val[0])
    gdf[col_name] = values
    return gdf
for name, path in factor_files.items():
    print(f"提取 {name} ...")
    all_points = extract_raster_values(all_points, path, name)
# ------------------ 4. 提取岩性 ------------------
# 读取岩性面矢量
litho_gdf = gpd.read_file(r'D:\AAAAllllll\htpro\data_xinan\yan\xinan_lithology_classified.shp', encoding='GBK')
# 统一CRS
litho_gdf = litho_gdf.to_crs(all_points.crs)
# 空间连接：将每个点所在面的 ROCK_CODE 赋给点
all_points = gpd.sjoin(all_points, litho_gdf[['ROCK_CODE', 'geometry']], how='left', predicate='within')
print("未匹配到岩性的点数:", all_points['ROCK_CODE'].isna().sum())
# 剔除未匹配的点
all_points = all_points.dropna(subset=['ROCK_CODE'])
# 剔除水体（CODE=0）
all_points = all_points[all_points['ROCK_CODE'] != 0]
print(f"剔除水体及未匹配点后样本数: {len(all_points)}")
# 确保几何列仍是有效
all_points = all_points.set_geometry('geometry')
# ------------------ 5. 数据清洗（去除NaN） ------------------
# 检查连续因子是否有NaN（可能栅格范围不全）
numeric_factors = list(factor_files.keys())
print("提取前各因子NaN数量:")
print(all_points[numeric_factors].isnull().sum())
all_points = all_points.dropna(subset=numeric_factors)
print(f"剔除NaN后样本数: {len(all_points)}")
# ------------------ 6. 构建VIF矩阵 ------------------
# 连续因子
X_numeric = all_points[numeric_factors].copy()
# 岩性One-Hot编码（去掉第一个类别以避免虚拟变量陷阱）
litho_dummies = pd.get_dummies(all_points['ROCK_CODE'], prefix='litho', drop_first=True)
# 合并
X_all = pd.concat([X_numeric, litho_dummies], axis=1)
X_all = X_all.astype(float)
print("用于VIF检验的因子总数（含哑变量）:", X_all.shape[1])
# ------------------ 7. 计算VIF并迭代剔除 ------------------
def calculate_vif(df):
    vif_df = pd.DataFrame()
    vif_df['Factor'] = df.columns
    vif_df['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    vif_df['TOL'] = 1 / vif_df['VIF']
    return vif_df
# 迭代剔除VIF > 10的因子
while True:
    vif_df = calculate_vif(X_all)
    max_vif = vif_df['VIF'].max()
    max_factor = vif_df.loc[vif_df['VIF'].idxmax(), 'Factor']
    
    if max_vif < 10:
        print("所有因子VIF < 10，多重共线性检验通过！")
        break
    else:
        print(f"因子 '{max_factor}' 的VIF = {max_vif:.2f} > 10，剔除该因子")
        X_all = X_all.drop(columns=[max_factor])
print("\n最终保留的因子列表:")
print(list(X_all.columns))
print("\n最终VIF结果:")
print(calculate_vif(X_all).to_string())
# 保存最终因子列表（以备后续建模使用）
final_factors = list(X_all.columns)
with open('final_factors.txt', 'w') as f:
    for item in final_factors:
        f.write(item + '\n')
print("\n因子列表已保存到 final_factors.txt")
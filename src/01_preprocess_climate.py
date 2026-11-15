#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMIP6 逐日降水 → 时段平均极端指数 → 90m Albers 重采样
"""

import os
import sys
import time
import warnings
import numpy as np
import xarray as xr
import rasterio
from rasterio.windows import Window
from rasterio.crs import CRS
from scipy.interpolate import RegularGridInterpolator
from pyproj import Transformer
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========================= 配置 =========================
# 原始数据
NC_FILE = r"D:/AAAAllllll/htpro/data_2008-2024/CMIP6/HiCPC_pr_INM-CM4-8_ssp126.nc"
# 模板栅格
TEMPLATE_TIF = r"D:/AAAAllllll/htpro/data_xinan/rasters_90m/dem.tif"
# 输出目录
OUTPUT_DIR = r"E:/CMIP6/INM-CM4-8/SSP126"  # 可修改为移动硬盘路径
# 模型与情景标识
MODEL = "INM-CM4-8" 
SCENARIO = "SSP126"

# 四个时段
PERIODS = [
    ('2021-01-01', '2040-12-31'),
    ('2041-01-01', '2060-12-31'),
    ('2061-01-01', '2080-12-31'),
    ('2081-01-01', '2100-12-31')
]

TEST_MODE = False  # 关闭测试模式

PERIODS = [
    ('2021-01-01', '2040-12-31'),
    ('2041-01-01', '2060-12-31'),
    ('2061-01-01', '2080-12-31'),
    ('2081-01-01', '2100-12-31')
]
# 重采样分块大小，根据内存调整
BLOCK_SIZE = 512
# 插值方法：'linear' 或 'nearest'
INTERP_METHOD = 'linear'

# ========================= 工具函数 =========================
def resample_to_target(lowres_da, template_tif, output_tif, block_size=512, method='linear'):
    """
    将低分辨率（经纬度网格）的 DataArray 重采样到目标投影网格（Albers 90m）
    
    Parameters:
        lowres_da: xr.DataArray，维度为 (lat, lon)，值为浮点数
        template_tif: 模板 GeoTIFF 路径
        output_tif: 输出 GeoTIFF 路径
        block_size: 分块写入的像素尺寸
        method: 插值方法 'linear' 或 'nearest'
    """
    # ---- 读取模板 ----
    with rasterio.open(template_tif) as src:
        dst_crs = src.crs
        dst_transform = src.transform
        dst_width = src.width
        dst_height = src.height
        dst_nodata = src.nodata if src.nodata is not None else np.nan

    # ---- 准备低分辨率网格 ----
    # 确保维度顺序为 (lat, lon)
    da = lowres_da.transpose('lat', 'lon')
    lat_vals = da.lat.values.astype(np.float64)
    lon_vals = da.lon.values.astype(np.float64)
    values = da.values.astype(np.float32)

    # 处理纬度（统一转为升序）
    if lat_vals[0] > lat_vals[-1]:
        lat_vals = lat_vals[::-1]
        values = values[::-1, :]
    # 处理经度（统一转为升序）
    if lon_vals[0] > lon_vals[-1]:
        lon_vals = lon_vals[::-1]
        values = values[:, ::-1]

    # ---- 构建插值器 ----
    interp = RegularGridInterpolator(
        (lat_vals, lon_vals),
        values,
        method=method,
        bounds_error=False,
        fill_value=np.nan
    )

    # ---- 建立坐标转换器（目标投影 → WGS84 经纬度） ----
    transformer = Transformer.from_crs(dst_crs, 'EPSG:4326', always_xy=True)

    # ---- 准备输出 GeoTIFF 配置 ----
    profile = {
        'driver': 'GTiff',
        'height': dst_height,
        'width': dst_width,
        'count': 1,
        'dtype': 'float32',
        'crs': dst_crs,
        'transform': dst_transform,
        'nodata': np.nan,
        'compress': 'DEFLATE',   # 压缩以减少磁盘占用
        'predictor': 2,
        'zlevel': 6,
        'tiled': True,
        'blockxsize': block_size,
        'blockysize': block_size,
    }

    # ---- 分块重采样并写入 ----
    logger.info(f"开始重采样，目标栅格大小：{dst_height}×{dst_width}")
    with rasterio.open(output_tif, 'w', **profile) as dst:
        total_blocks = ((dst_height + block_size - 1) // block_size) * \
                       ((dst_width + block_size - 1) // block_size)
        block_idx = 0

        for row_start in range(0, dst_height, block_size):
            row_end = min(row_start + block_size, dst_height)
            block_h = row_end - row_start
            for col_start in range(0, dst_width, block_size):
                col_end = min(col_start + block_size, dst_width)
                block_w = col_end - col_start

                # 生成该块内所有像素中心的投影坐标 (x, y)
                cols = np.arange(col_start, col_end, dtype=np.float64)
                rows = np.arange(row_start, row_end, dtype=np.float64)
                cols_grid, rows_grid = np.meshgrid(cols, rows)
                # 像素中心偏移 0.5
                xs, ys = dst_transform * (cols_grid + 0.5, rows_grid + 0.5)

                # 将投影坐标转换为经纬度 (lon, lat)
                lons_flat = xs.ravel()
                lats_flat = ys.ravel()
                lons, lats = transformer.transform(lons_flat, lats_flat)

                # 构建插值输入点 (lat, lon)
                points = np.stack((lats, lons), axis=-1)
                # 执行插值
                block_data = interp(points).reshape(block_h, block_w).astype(np.float32)

                # 写入当前块
                window = Window(col_start, row_start, block_w, block_h)
                dst.write(block_data, 1, window=window)

                block_idx += 1
                if block_idx % 100 == 0:
                    logger.info(f"已处理 {block_idx}/{total_blocks} 个分块")

    logger.info(f"重采样完成，输出文件：{output_tif}")
    return output_tif


def verify_output(tif_path, expected_crs=None):
    
    if not os.path.exists(tif_path):
        logger.error(f"文件不存在：{tif_path}")
        return False
    with rasterio.open(tif_path) as src:
        crs = src.crs
        if expected_crs is not None and crs != expected_crs:
            logger.warning(f"CRS 不匹配：期望 {expected_crs}，实际 {crs}")
        # 计算有效数据比例（非 NaN）
        data = src.read(1)
        valid = ~np.isnan(data)
        valid_ratio = valid.sum() / data.size if data.size > 0 else 0
        logger.info(f"文件 {os.path.basename(tif_path)}："
                    f"尺寸 {src.width}×{src.height}，"
                    f"有效像素占比 {valid_ratio:.2%}，"
                    f"CRS {crs}")
        return True


# ========================= 主处理流程 =========================
def main():
    start_time = time.time()
    logger.info("========== 开始处理 CMIP6 极端降水指数 ==========")

    # ---- 1. 创建输出目录 ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- 2. 打开原始 NetCDF 并预处理 ----
    logger.info(f"读取原始数据：{NC_FILE}")
    ds = xr.open_dataset(NC_FILE, chunks={'time': 365})  # 按年分块，避免一次读入全部
    # 确保经纬度升序，便于切片
    ds = ds.sortby('lat').sortby('lon')
    # 裁剪到西南四省范围（略留余量，实际根据研究区调整）
    ds = ds.sel(lat=slice(21, 35), lon=slice(97, 111))
    # 单位换算：kg/(m²·s) → mm/day
    pr_mm = ds['pr']
    logger.info(f"原始数据裁剪后纬度范围：{ds.lat.values.min():.2f} ~ {ds.lat.values.max():.2f}，"
                f"经度范围：{ds.lon.values.min():.2f} ~ {ds.lon.values.max():.2f}")

    # ---- 3. 获取目标 CRS（用于验证） ----
    with rasterio.open(TEMPLATE_TIF) as src:
        target_crs = src.crs

    # ---- 4. 循环处理每个时段 ----
    for idx, (start_date, end_date) in enumerate(PERIODS, 1):
        logger.info(f"处理时段 {idx}/{len(PERIODS)}：{start_date} 至 {end_date}")

        # 提取该时段的所有数据（Dask 延迟，仅定义计算图）
        pr_period = pr_mm.sel(time=slice(start_date, end_date))
        # 获取该时段包含的年份列表
        years = np.unique(pr_period.time.dt.year.values)
        logger.info(f"包含年份：{years.tolist()}")

        # ---- 4.1 逐年计算 Rx1day 和 R20mm，并累加 ----
        sum_rx1 = None
        sum_r20 = None
        n_years = len(years)

        for y in years:
            # 选择该年数据，并加载到内存（每年约 365×~20000 个网格，~30 MB）
            pr_year = pr_period.sel(time=slice(f'{y}-01-01', f'{y}-12-31')).load()
            # Rx1day：年最大日降水量（mm/day）
            rx1 = pr_year.max(dim='time', skipna=True)
            # R20mm：日降水量 ≥20mm 的天数
            r20 = (pr_year >= 20.0).sum(dim='time', skipna=True)

            if sum_rx1 is None:
                sum_rx1 = rx1
                sum_r20 = r20
            else:
                sum_rx1 = sum_rx1 + rx1
                sum_r20 = sum_r20 + r20

            # 释放该年数据，降低内存占用
            del pr_year

        # 时段平均
        avg_rx1 = sum_rx1 / n_years
        avg_r20 = sum_r20 / n_years
        logger.info(f"时段平均计算完成，低分辨率栅格形状：{avg_rx1.shape}")

        # ---- 4.2 重采样到目标网格 ----
        # 构建输出文件名
        period_str = f"{start_date[:4]}-{end_date[:4]}"
        out_rx1 = os.path.join(OUTPUT_DIR, f"Rx1day_{MODEL}_{SCENARIO}_{period_str}_avg.tif")
        out_r20 = os.path.join(OUTPUT_DIR, f"R20mm_{MODEL}_{SCENARIO}_{period_str}_avg.tif")

        # 重采样（Rx1day）
        logger.info("重采样 Rx1day...")
        resample_to_target(avg_rx1, TEMPLATE_TIF, out_rx1, BLOCK_SIZE, INTERP_METHOD)

        # 重采样（R20mm）
        logger.info("重采样 R20mm...")
        resample_to_target(avg_r20, TEMPLATE_TIF, out_r20, BLOCK_SIZE, INTERP_METHOD)

        # ---- 4.3 简单验证 ----
        verify_output(out_rx1, target_crs)
        verify_output(out_r20, target_crs)

        # 释放时段数据
        del pr_period, sum_rx1, sum_r20, avg_rx1, avg_r20

    # ---- 5. 全部完成 ----
    elapsed = time.time() - start_time
    logger.info(f"========== 全部处理完成，总耗时 {elapsed/60:.2f} 分钟 ==========")
    logger.info(f"输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
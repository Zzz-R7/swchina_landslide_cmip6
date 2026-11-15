#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
滑坡易发性预测 —— ACCESS-CM2 SSP245 四个未来时段
输入：四个时段的 Rx1day + R20mm + 9 个静态因子
输出：4 张滑坡易发性图（概率 0~1）
"""

import os
import time
import numpy as np
import rasterio
from rasterio.windows import Window
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# ================== 【配置区】 ====================
# ============================================================

# 静态因子目录
STATIC_DIR = r"D:/AAAAllllll/htpro/data_xinan/rasters_90m"

# 训练好的模型
MODEL_PATH = r"D:/AAAAllllll/htpro/data_xinan/modeldata/best/final_model_RF.pkl"

# 未来降水因子所在目录
FUTURE_DIR = r"E:/CMIP6/EC-Earth3-Veg/SSP585"

# 输出目录（4 张易发性图）
OUTPUT_DIR = r"E:/CMIP6/EC-Earth3-Veg/SSP585/susceptibility"
 
# 四个时段（文件名前缀）
PERIODS = ["2021-2040", "2041-2060", "2061-2080", "2081-2100"]

MODEL = "EC-Earth3-Veg" 
SCENARIO = "SSP585"
# 特征列表
FACTOR_ORDER = [
    "Rx1day", "R20mm",
    "dem", "slope", "aspect", "plan", "profile",
    "stream", "dc", "road", "litho"
]

# 静态因子文件名映射
STATIC_FILES = {
    "dem":     "dem.tif",
    "slope":   "slope.tif",
    "aspect":  "aspect.tif",
    "plan":    "plan.tif",
    "profile": "profile.tif",
    "stream":  "stream.tif",
    "dc":      "dc.tif",
    "road":    "road.tif",
    "litho":   "litho.tif"
}

# 分块大小（像素）
BLOCK_SIZE = 512


def predict_one_period(period, model, static_srcs, template_profile):
    """预测单个时段的滑坡易发性图"""
    logger.info("=" * 70)
    logger.info(f"[时段] {period}")

    # 未来降水文件路径
    rx1_path = os.path.join(FUTURE_DIR, f"Rx1day_{MODEL}_{SCENARIO}_{period}_avg.tif")
    r20_path = os.path.join(FUTURE_DIR, f"R20mm_{MODEL}_{SCENARIO}_{period}_avg.tif")
    if not (os.path.exists(rx1_path) and os.path.exists(r20_path)):
        logger.error(f"缺少输入文件：{rx1_path} 或 {r20_path}")
        return

    out_path = os.path.join(OUTPUT_DIR, f"susceptibility_{period}.tif")

    # 打开该时段两个降水文件
    src_rx1 = rasterio.open(rx1_path)
    src_r20 = rasterio.open(r20_path)
    height, width = src_rx1.height, src_rx1.width

    profile = template_profile.copy()
    profile.update({
        'height': height, 'width': width,
        'count': 1, 'dtype': 'float32',
        'compress': 'DEFLATE', 'predictor': 2, 'zlevel': 6,
        'nodata': np.nan,
        'blockxsize': BLOCK_SIZE, 'blockysize': BLOCK_SIZE,
    })

    total_blocks = ((height + BLOCK_SIZE - 1) // BLOCK_SIZE) * \
                   ((width + BLOCK_SIZE - 1) // BLOCK_SIZE)
    block_count = 0

    with rasterio.open(out_path, 'w', **profile) as dst:
        for row_start in range(0, height, BLOCK_SIZE):
            row_end = min(row_start + BLOCK_SIZE, height)
            block_h = row_end - row_start
            for col_start in range(0, width, BLOCK_SIZE):
                col_end = min(col_start + BLOCK_SIZE, width)
                block_w = col_end - col_start
                window = Window(col_start, row_start, block_w, block_h)

                # 按 FACTOR_ORDER 的顺序依次读取 11 个特征
                bands = []
                for name in FACTOR_ORDER:
                    if name == "Rx1day":
                        data = src_rx1.read(1, window=window).astype(np.float32)
                    elif name == "R20mm":
                        data = src_r20.read(1, window=window).astype(np.float32)
                    else:
                        data = static_srcs[name].read(1, window=window).astype(np.float32)
                    bands.append(data)

                # 堆叠成 (h, w, 11)
                stacked = np.stack(bands, axis=-1)

                # 有效像元掩膜（所有特征均非 NaN）
                valid_mask = ~np.isnan(stacked).any(axis=-1)

                out_block = np.full((block_h, block_w), np.nan, dtype=np.float32)
                if valid_mask.any():
                    valid_pixels = stacked[valid_mask]  # (n, 11)
                    proba = model.predict_proba(valid_pixels)[:, 1]
                    out_block[valid_mask] = proba.astype(np.float32)

                dst.write(out_block, 1, window=window)
                block_count += 1

            if (row_start // BLOCK_SIZE) % 5 == 0:
                logger.info(f"  进度 {block_count}/{total_blocks} 块")

    src_rx1.close()
    src_r20.close()

    # 验证
    with rasterio.open(out_path) as src:
        data = src.read(1)
        valid = ~np.isnan(data)
        ratio = valid.sum() / data.size if data.size > 0 else 0
        if valid.any():
            logger.info(f"  ✔ 输出：{os.path.basename(out_path)} | "
                        f"有效 {ratio:.2%} | "
                        f"min={data[valid].min():.3f} "
                        f"max={data[valid].max():.3f} "
                        f"mean={data[valid].mean():.3f}")


def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("滑坡易发性批量预测 —— ACCESS-CM2 SSP245 四个未来时段")
    logger.info("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- 加载模型 ----
    logger.info(f"加载模型：{MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # ---- 打开静态因子（全程复用） ----
    logger.info("打开 9 个静态因子...")
    static_srcs = {}
    for name, fname in STATIC_FILES.items():
        path = os.path.join(STATIC_DIR, fname)
        static_srcs[name] = rasterio.open(path)

    # 用第一个静态因子（dem.tif）的 profile 作为模板
    template_profile = static_srcs["dem"].profile.copy()

    # ---- 批量预测四个时段 ----
    for period in PERIODS:
        predict_one_period(period, model, static_srcs, template_profile)

    # ---- 关闭静态因子 ----
    for src in static_srcs.values():
        src.close()

    logger.info("=" * 70)
    logger.info(f"全部完成，总耗时 {(time.time()-t0)/60:.2f} 分钟")
    logger.info(f"输出目录：{OUTPUT_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
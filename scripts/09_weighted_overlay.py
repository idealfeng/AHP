"""加权叠加分析脚本：将五个因子栅格按照指定权重进行叠加，生成连续得分图和分级图，并统计各级别的面积。"""
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

BASE_DIR = Path(r"D:\Paper\毕设")

FACTOR_DIR = BASE_DIR / "data" / "factors"
RESULT_DIR = BASE_DIR / "results" / "suitability"
TABLE_DIR = BASE_DIR / "results" / "tables"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

factor_paths = {
    "road": FACTOR_DIR / "road_score.tif",
    "poi": FACTOR_DIR / "poi_score.tif",
    "substation": FACTOR_DIR / "substation_score.tif",
    "landuse": FACTOR_DIR / "landuse_score.tif",
    "slope": FACTOR_DIR / "slope_score.tif",
}

# baseline 权重
weights = {
    "road": 0.25,
    "poi": 0.30,
    "substation": 0.20,
    "landuse": 0.15,
    "slope": 0.10,
}

assert abs(sum(weights.values()) - 1.0) < 1e-9, "权重和必须为 1"

arrays = {}
meta = None
valid_mask = None

for name, path in factor_paths.items():
    if not path.exists():
        raise FileNotFoundError(f"找不到因子文件：{path}")

    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata

        if meta is None:
            meta = src.meta.copy()
        else:
            # 检查对齐
            if (
                src.width != meta["width"]
                or src.height != meta["height"]
                or src.transform != meta["transform"]
                or src.crs != meta["crs"]
            ):
                raise ValueError(f"{name} 与其他因子栅格不对齐。")

    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata

    arrays[name] = arr

    if valid_mask is None:
        valid_mask = valid
    else:
        valid_mask &= valid

print("有效像元数：", int(valid_mask.sum()))

# 加权叠加
suitability = np.full(arr.shape, -9999.0, dtype="float32")

weighted_sum = np.zeros(arr.shape, dtype="float32")
for name, weight in weights.items():
    weighted_sum += arrays[name] * weight

suitability[valid_mask] = weighted_sum[valid_mask]

# 保存连续得分图
score_meta = meta.copy()
score_meta.update({"dtype": "float32", "nodata": -9999.0, "compress": "lzw"})

score_path = RESULT_DIR / "suitability_score.tif"
with rasterio.open(score_path, "w", **score_meta) as dst:
    dst.write(suitability, 1)

print("已输出：", score_path)

# 分位数分级：1-5
valid_values = suitability[valid_mask]

q20, q40, q60, q80 = np.quantile(valid_values, [0.2, 0.4, 0.6, 0.8])

classes = np.zeros(arr.shape, dtype="uint8")
classes[(suitability > -9999) & (suitability <= q20)] = 1
classes[(suitability > q20) & (suitability <= q40)] = 2
classes[(suitability > q40) & (suitability <= q60)] = 3
classes[(suitability > q60) & (suitability <= q80)] = 4
classes[(suitability > q80)] = 5

class_meta = meta.copy()
class_meta.update({"dtype": "uint8", "nodata": 0, "compress": "lzw"})

class_path = RESULT_DIR / "suitability_class.tif"
with rasterio.open(class_path, "w", **class_meta) as dst:
    dst.write(classes, 1)

print("已输出：", class_path)

# 面积统计
pixel_area_m2 = abs(meta["transform"].a * meta["transform"].e)
pixel_area_km2 = pixel_area_m2 / 1_000_000

rows = []
label_map = {
    1: "不适宜区",
    2: "较低适宜区",
    3: "中等适宜区",
    4: "较高适宜区",
    5: "高适宜区",
}

total_valid = int(valid_mask.sum())

for cls in [1, 2, 3, 4, 5]:
    count = int((classes == cls).sum())
    area_km2 = count * pixel_area_km2
    ratio = count / total_valid if total_valid > 0 else 0

    rows.append(
        {
            "class": cls,
            "label": label_map[cls],
            "pixel_count": count,
            "area_km2": area_km2,
            "ratio": ratio,
        }
    )

stats_df = pd.DataFrame(rows)
stats_path = TABLE_DIR / "suitability_area_stats.csv"
stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")

print("已输出：", stats_path)
print("\n分级阈值：")
print(f"Q20={q20:.4f}, Q40={q40:.4f}, Q60={q60:.4f}, Q80={q80:.4f}")

print("\n面积统计：")
print(stats_df)

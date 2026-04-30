"""基于 AHP 权重的加权叠加适宜性分析脚本"""
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

BASE_DIR = Path(r"D:\Paper\毕设")

FACTOR_DIR = BASE_DIR / "data" / "factors"
WEIGHTS_CSV = BASE_DIR / "results" / "weights" / "manual_ahp_weights.csv"

SCENARIO_NAME = "manual_ahp"
RESULT_DIR = BASE_DIR / "results" / "suitability" / SCENARIO_NAME
TABLE_DIR = BASE_DIR / "results" / "tables" / SCENARIO_NAME

RESULT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

factor_paths = {
    "road": FACTOR_DIR / "road_score.tif",
    "poi": FACTOR_DIR / "poi_score.tif",
    "substation": FACTOR_DIR / "substation_score.tif",
    "landuse": FACTOR_DIR / "landuse_score.tif",
    "slope": FACTOR_DIR / "slope_score.tif",
}

SCORE_OUT = RESULT_DIR / "suitability_score.tif"
CLASS_OUT = RESULT_DIR / "suitability_class.tif"
STATS_OUT = TABLE_DIR / "suitability_area_stats.csv"
WEIGHTS_USED_OUT = TABLE_DIR / "weights_used.csv"


def read_weights():
    if not WEIGHTS_CSV.exists():
        raise FileNotFoundError(f"找不到 AHP 权重文件：{WEIGHTS_CSV}")

    df = pd.read_csv(WEIGHTS_CSV)

    if "factor" not in df.columns or "weight" not in df.columns:
        raise ValueError("权重文件必须包含 factor 和 weight 两列。")

    weights = dict(zip(df["factor"], df["weight"]))

    missing = set(factor_paths.keys()) - set(weights.keys())
    if missing:
        raise ValueError(f"权重文件缺少因子：{missing}")

    # 只保留当前需要的因子，并重新归一化，避免浮点误差
    weights = {k: float(weights[k]) for k in factor_paths.keys()}
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    return weights


def main():
    weights = read_weights()

    print("\n=== 使用 AHP 权重 ===")
    for k, v in weights.items():
        print(f"{k}: {v:.6f}")
    print("权重和：", sum(weights.values()))

    pd.DataFrame([{"factor": k, "weight": v} for k, v in weights.items()]).to_csv(
        WEIGHTS_USED_OUT, index=False, encoding="utf-8-sig"
    )

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

    suitability = np.full(arr.shape, -9999.0, dtype="float32")
    weighted_sum = np.zeros(arr.shape, dtype="float32")

    for name, weight in weights.items():
        weighted_sum += arrays[name] * weight

    suitability[valid_mask] = weighted_sum[valid_mask]

    score_meta = meta.copy()
    score_meta.update({"dtype": "float32", "nodata": -9999.0, "compress": "lzw"})

    with rasterio.open(SCORE_OUT, "w", **score_meta) as dst:
        dst.write(suitability, 1)

    print("已输出综合适宜性得分：", SCORE_OUT)

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

    with rasterio.open(CLASS_OUT, "w", **class_meta) as dst:
        dst.write(classes, 1)

    print("已输出适宜性分级：", CLASS_OUT)

    pixel_area_m2 = abs(meta["transform"].a * meta["transform"].e)
    pixel_area_km2 = pixel_area_m2 / 1_000_000

    label_map = {
        1: "不适宜区",
        2: "较低适宜区",
        3: "中等适宜区",
        4: "较高适宜区",
        5: "高适宜区",
    }

    total_valid = int(valid_mask.sum())

    rows = []

    for cls in [1, 2, 3, 4, 5]:
        count = int((classes == cls).sum())
        area_km2 = count * pixel_area_km2
        ratio = count / total_valid if total_valid > 0 else 0

        rows.append(
            {
                "scenario": SCENARIO_NAME,
                "class": cls,
                "label": label_map[cls],
                "pixel_count": count,
                "area_km2": area_km2,
                "ratio": ratio,
            }
        )

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(STATS_OUT, index=False, encoding="utf-8-sig")

    print("已输出面积统计：", STATS_OUT)

    print("\n分级阈值：")
    print(f"Q20={q20:.4f}, Q40={q40:.4f}, Q60={q60:.4f}, Q80={q80:.4f}")

    print("\n综合适宜性统计：")
    print(f"min={valid_values.min():.4f}")
    print(f"max={valid_values.max():.4f}")
    print(f"mean={valid_values.mean():.4f}")
    print(f"std={valid_values.std():.4f}")

    print("\n面积统计：")
    print(stats_df)


if __name__ == "__main__":
    main()

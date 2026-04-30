"""对 LLM-AHP 方案进行单因子去除敏感性分析，评估每个因子对适宜性结果的影响"""
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

BASE_DIR = Path(r"D:\Paper\毕设")

SCENARIO_NAME = "llm_ahp"

FACTOR_DIR = BASE_DIR / "data" / "factors"
WEIGHTS_CSV = BASE_DIR / "results" / "weights" / "llm_ahp" / "llm_ahp_weights.csv"

BASE_SCORE = (
    BASE_DIR / "results" / "suitability" / SCENARIO_NAME / "suitability_score.tif"
)
BASE_CLASS = (
    BASE_DIR / "results" / "suitability" / SCENARIO_NAME / "suitability_class.tif"
)

OUT_DIR = BASE_DIR / "results" / "sensitivity" / SCENARIO_NAME
TABLE_DIR = BASE_DIR / "results" / "tables" / SCENARIO_NAME
MAP_DIR = BASE_DIR / "results" / "maps" / SCENARIO_NAME

OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
MAP_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = TABLE_DIR / "sensitivity_analysis.csv"
BAR_MAP = MAP_DIR / "sensitivity_change_ratio_bar.png"
JACCARD_MAP = MAP_DIR / "sensitivity_high_jaccard_bar.png"

factor_paths = {
    "road": FACTOR_DIR / "road_score.tif",
    "poi": FACTOR_DIR / "poi_score.tif",
    "substation": FACTOR_DIR / "substation_score.tif",
    "landuse": FACTOR_DIR / "landuse_score.tif",
    "slope": FACTOR_DIR / "slope_score.tif",
}

FACTOR_CN = {
    "road": "道路距离",
    "poi": "POI密度",
    "substation": "变电站距离",
    "landuse": "土地利用",
    "slope": "坡度",
}

plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def read_weights():
    df = pd.read_csv(WEIGHTS_CSV)

    weights = dict(zip(df["factor"], df["weight"]))

    weights = {k: float(weights[k]) for k in factor_paths.keys()}

    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    return weights


def read_factor_arrays():
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

    return arrays, meta, valid_mask


def read_base_result():
    with rasterio.open(BASE_SCORE) as src:
        base_score = src.read(1).astype("float32")
        score_nodata = src.nodata

    with rasterio.open(BASE_CLASS) as src:
        base_class = src.read(1).astype("uint8")
        class_nodata = src.nodata

    valid_score = np.isfinite(base_score)
    if score_nodata is not None:
        valid_score &= base_score != score_nodata

    valid_class = base_class > 0
    if class_nodata is not None:
        valid_class &= base_class != class_nodata

    valid = valid_score & valid_class

    return base_score, base_class, valid


def get_base_thresholds(base_score, valid):
    values = base_score[valid]
    q20, q40, q60, q80 = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
    return q20, q40, q60, q80


def classify_with_thresholds(score, valid, thresholds):
    q20, q40, q60, q80 = thresholds

    classes = np.zeros(score.shape, dtype="uint8")

    classes[valid & (score <= q20)] = 1
    classes[valid & (score > q20) & (score <= q40)] = 2
    classes[valid & (score > q40) & (score <= q60)] = 3
    classes[valid & (score > q60) & (score <= q80)] = 4
    classes[valid & (score > q80)] = 5

    return classes


def save_score_raster(score, meta, path):
    out_meta = meta.copy()
    out_meta.update({"dtype": "float32", "nodata": -9999.0, "compress": "lzw"})

    with rasterio.open(path, "w", **out_meta) as dst:
        dst.write(score.astype("float32"), 1)


def save_class_raster(classes, meta, path):
    out_meta = meta.copy()
    out_meta.update({"dtype": "uint8", "nodata": 0, "compress": "lzw"})

    with rasterio.open(path, "w", **out_meta) as dst:
        dst.write(classes.astype("uint8"), 1)


def run_remove_one_factor(
    remove_factor,
    arrays,
    weights,
    meta,
    valid_mask,
    base_score,
    base_class,
    base_valid,
    thresholds,
):
    print(f"\n=== 去除因子：{remove_factor} / {FACTOR_CN[remove_factor]} ===")

    remaining = [f for f in factor_paths.keys() if f != remove_factor]

    remaining_weight_sum = sum(weights[f] for f in remaining)

    new_weights = {f: weights[f] / remaining_weight_sum for f in remaining}

    print("重新归一化权重：")
    for f, w in new_weights.items():
        print(f"  {f}: {w:.6f}")

    score = np.full(base_score.shape, -9999.0, dtype="float32")
    weighted_sum = np.zeros(base_score.shape, dtype="float32")

    for f, w in new_weights.items():
        weighted_sum += arrays[f] * w

    valid = valid_mask & base_valid
    score[valid] = weighted_sum[valid]

    classes = classify_with_thresholds(score, valid, thresholds)

    remove_dir = OUT_DIR / f"remove_{remove_factor}"
    remove_dir.mkdir(parents=True, exist_ok=True)

    score_out = remove_dir / "suitability_score.tif"
    class_out = remove_dir / "suitability_class.tif"

    save_score_raster(score, meta, score_out)
    save_class_raster(classes, meta, class_out)

    # 对比原始 LLM-AHP
    compare_valid = valid & (base_class > 0) & (classes > 0)

    changed = compare_valid & (classes != base_class)
    same = compare_valid & (classes == base_class)

    class_diff = np.zeros(base_class.shape, dtype="int16")
    class_diff[compare_valid] = classes[compare_valid].astype("int16") - base_class[
        compare_valid
    ].astype("int16")

    score_diff = np.full(base_score.shape, -9999.0, dtype="float32")
    score_diff[compare_valid] = score[compare_valid] - base_score[compare_valid]

    base_high = compare_valid & (base_class == 5)
    remove_high = compare_valid & (classes == 5)

    overlap = base_high & remove_high
    union = base_high | remove_high

    pixel_area_m2 = abs(meta["transform"].a * meta["transform"].e)
    pixel_area_km2 = pixel_area_m2 / 1_000_000

    total_pixels = int(compare_valid.sum())
    changed_pixels = int(changed.sum())
    same_pixels = int(same.sum())

    change_ratio = changed_pixels / total_pixels if total_pixels > 0 else np.nan

    high_jaccard = overlap.sum() / union.sum() if union.sum() > 0 else np.nan

    summary = {
        "removed_factor": remove_factor,
        "removed_factor_cn": FACTOR_CN[remove_factor],
        "original_weight": weights[remove_factor],
        "total_pixels": total_pixels,
        "same_pixels": same_pixels,
        "changed_pixels": changed_pixels,
        "changed_area_km2": changed_pixels * pixel_area_km2,
        "change_ratio": change_ratio,
        "score_diff_min": float(score_diff[compare_valid].min()),
        "score_diff_max": float(score_diff[compare_valid].max()),
        "score_diff_mean": float(score_diff[compare_valid].mean()),
        "score_diff_abs_mean": float(np.abs(score_diff[compare_valid]).mean()),
        "base_high_area_km2": float(base_high.sum() * pixel_area_km2),
        "remove_high_area_km2": float(remove_high.sum() * pixel_area_km2),
        "high_overlap_area_km2": float(overlap.sum() * pixel_area_km2),
        "high_union_area_km2": float(union.sum() * pixel_area_km2),
        "high_jaccard": float(high_jaccard),
        "score_out": str(score_out),
        "class_out": str(class_out),
    }

    print(f"等级变化率：{change_ratio:.4f}")
    print(f"等级变化面积：{summary['changed_area_km2']:.2f} km²")
    print(f"高适宜区 Jaccard 重叠度：{high_jaccard:.4f}")
    print(f"平均绝对得分差异：{summary['score_diff_abs_mean']:.4f}")

    return summary


def plot_results(df):
    print("\n=== 绘制敏感性分析图 ===")

    plot_df = df.sort_values("change_ratio", ascending=False)

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["removed_factor_cn"], plot_df["change_ratio"])
    plt.ylabel("等级变化率")
    plt.xlabel("去除因子")
    plt.title("单因子去除敏感性分析：等级变化率")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(BAR_MAP, dpi=300, bbox_inches="tight")
    plt.close()

    print("已输出：", BAR_MAP)

    plot_df2 = df.sort_values("high_jaccard", ascending=True)

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df2["removed_factor_cn"], plot_df2["high_jaccard"])
    plt.ylabel("高适宜区 Jaccard 重叠度")
    plt.xlabel("去除因子")
    plt.title("单因子去除敏感性分析：高适宜区重叠度")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(JACCARD_MAP, dpi=300, bbox_inches="tight")
    plt.close()

    print("已输出：", JACCARD_MAP)


def main():
    for path in [WEIGHTS_CSV, BASE_SCORE, BASE_CLASS]:
        if not path.exists():
            raise FileNotFoundError(f"找不到文件：{path}")

    weights = read_weights()
    arrays, meta, valid_mask = read_factor_arrays()
    base_score, base_class, base_valid = read_base_result()

    thresholds = get_base_thresholds(base_score, base_valid)

    print("\n=== 原始 LLM-AHP 权重 ===")
    for f, w in weights.items():
        print(f"{f}: {w:.6f}")

    print("\n=== 原始 LLM-AHP 分级阈值 ===")
    print(
        f"Q20={thresholds[0]:.4f}, Q40={thresholds[1]:.4f}, Q60={thresholds[2]:.4f}, Q80={thresholds[3]:.4f}"
    )

    summaries = []

    for remove_factor in factor_paths.keys():
        result = run_remove_one_factor(
            remove_factor=remove_factor,
            arrays=arrays,
            weights=weights,
            meta=meta,
            valid_mask=valid_mask,
            base_score=base_score,
            base_class=base_class,
            base_valid=base_valid,
            thresholds=thresholds,
        )

        summaries.append(result)

    df = pd.DataFrame(summaries)
    df = df.sort_values("change_ratio", ascending=False).reset_index(drop=True)

    df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\n=== 敏感性分析汇总 ===")
    print(
        df[
            [
                "removed_factor",
                "removed_factor_cn",
                "original_weight",
                "change_ratio",
                "changed_area_km2",
                "score_diff_abs_mean",
                "high_jaccard",
            ]
        ]
    )

    print("\n已输出敏感性分析表：", SUMMARY_CSV)

    plot_results(df)

    print("\n敏感性分析完成。")


if __name__ == "__main__":
    main()

"""LLM-AHP 多情景加权叠加脚本"""
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

BASE_DIR = Path(r"D:\Paper\毕设")

FACTOR_DIR = BASE_DIR / "data" / "factors"
WEIGHT_ROOT = BASE_DIR / "results" / "weights" / "llm_scenarios"

RESULT_ROOT = BASE_DIR / "results" / "suitability" / "llm_scenarios"
TABLE_ROOT = BASE_DIR / "results" / "tables" / "llm_scenarios"

RESULT_ROOT.mkdir(parents=True, exist_ok=True)
TABLE_ROOT.mkdir(parents=True, exist_ok=True)

OVERLAY_SUMMARY_CSV = TABLE_ROOT / "llm_scenario_overlay_summary.csv"

factor_paths = {
    "road": FACTOR_DIR / "road_score.tif",
    "poi": FACTOR_DIR / "poi_score.tif",
    "substation": FACTOR_DIR / "substation_score.tif",
    "landuse": FACTOR_DIR / "landuse_score.tif",
    "slope": FACTOR_DIR / "slope_score.tif",
}


def load_factor_arrays():
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


def read_scenario_weights(scenario_dir):
    weight_csv = scenario_dir / "llm_ahp_weights.csv"
    consistency_csv = scenario_dir / "llm_ahp_consistency.csv"

    if not weight_csv.exists():
        raise FileNotFoundError(f"找不到权重文件：{weight_csv}")

    weights_df = pd.read_csv(weight_csv)
    consistency_df = pd.read_csv(consistency_csv)

    passed = bool(consistency_df.loc[0, "passed"])

    if not passed:
        raise ValueError(f"{scenario_dir.name} 一致性未通过，不建议叠加。")

    weights = dict(zip(weights_df["factor"], weights_df["weight"]))

    weights = {f: float(weights[f]) for f in factor_paths.keys()}

    total = sum(weights.values())
    weights = {f: v / total for f, v in weights.items()}

    scenario_name_cn = weights_df.loc[0, "scenario_name_cn"]

    return weights, scenario_name_cn


def save_score_raster(score, meta, out_path):
    out_meta = meta.copy()
    out_meta.update({"dtype": "float32", "nodata": -9999.0, "compress": "lzw"})

    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(score.astype("float32"), 1)


def save_class_raster(classes, meta, out_path):
    out_meta = meta.copy()
    out_meta.update({"dtype": "uint8", "nodata": 0, "compress": "lzw"})

    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(classes.astype("uint8"), 1)


def run_overlay_for_scenario(scenario_dir, arrays, meta, valid_mask):
    scenario_id = scenario_dir.name

    weights, scenario_name_cn = read_scenario_weights(scenario_dir)

    print("\n" + "=" * 60)
    print(f"加权叠加情景：{scenario_id} / {scenario_name_cn}")
    print("=" * 60)

    print("权重：")
    for f, w in weights.items():
        print(f"  {f}: {w:.6f}")

    result_dir = RESULT_ROOT / scenario_id
    table_dir = TABLE_ROOT / scenario_id

    result_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    score = np.full(next(iter(arrays.values())).shape, -9999.0, dtype="float32")
    weighted_sum = np.zeros(score.shape, dtype="float32")

    for f, w in weights.items():
        weighted_sum += arrays[f] * w

    score[valid_mask] = weighted_sum[valid_mask]

    valid_values = score[valid_mask]

    q20, q40, q60, q80 = np.quantile(valid_values, [0.2, 0.4, 0.6, 0.8])

    classes = np.zeros(score.shape, dtype="uint8")
    classes[valid_mask & (score <= q20)] = 1
    classes[valid_mask & (score > q20) & (score <= q40)] = 2
    classes[valid_mask & (score > q40) & (score <= q60)] = 3
    classes[valid_mask & (score > q60) & (score <= q80)] = 4
    classes[valid_mask & (score > q80)] = 5

    score_out = result_dir / "suitability_score.tif"
    class_out = result_dir / "suitability_class.tif"

    save_score_raster(score, meta, score_out)
    save_class_raster(classes, meta, class_out)

    pixel_area_m2 = abs(meta["transform"].a * meta["transform"].e)
    pixel_area_km2 = pixel_area_m2 / 1_000_000

    label_map = {
        1: "不适宜区",
        2: "较低适宜区",
        3: "中等适宜区",
        4: "较高适宜区",
        5: "高适宜区",
    }

    rows = []
    total_valid = int(valid_mask.sum())

    for cls in [1, 2, 3, 4, 5]:
        count = int((classes == cls).sum())
        rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_name_cn": scenario_name_cn,
                "class": cls,
                "label": label_map[cls],
                "pixel_count": count,
                "area_km2": count * pixel_area_km2,
                "ratio": count / total_valid if total_valid > 0 else 0,
            }
        )

    area_df = pd.DataFrame(rows)
    area_df.to_csv(
        table_dir / "suitability_area_stats.csv", index=False, encoding="utf-8-sig"
    )

    weight_df = pd.DataFrame(
        [
            {
                "scenario_id": scenario_id,
                "scenario_name_cn": scenario_name_cn,
                "factor": f,
                "weight": w,
            }
            for f, w in weights.items()
        ]
    )
    weight_df.to_csv(table_dir / "weights_used.csv", index=False, encoding="utf-8-sig")

    summary = {
        "scenario_id": scenario_id,
        "scenario_name_cn": scenario_name_cn,
        "score_min": float(valid_values.min()),
        "score_max": float(valid_values.max()),
        "score_mean": float(valid_values.mean()),
        "score_std": float(valid_values.std()),
        "q20": float(q20),
        "q40": float(q40),
        "q60": float(q60),
        "q80": float(q80),
        "high_suitability_area_km2": float((classes == 5).sum() * pixel_area_km2),
        "score_out": str(score_out),
        "class_out": str(class_out),
    }

    for f, w in weights.items():
        summary[f"weight_{f}"] = w

    print("得分统计：")
    print(
        f"  min={summary['score_min']:.4f}, max={summary['score_max']:.4f}, mean={summary['score_mean']:.4f}, std={summary['score_std']:.4f}"
    )
    print(f"  Q20={q20:.4f}, Q40={q40:.4f}, Q60={q60:.4f}, Q80={q80:.4f}")
    print("已输出：", score_out)
    print("已输出：", class_out)

    return summary


def main():
    arrays, meta, valid_mask = load_factor_arrays()

    scenario_dirs = [
        p
        for p in WEIGHT_ROOT.iterdir()
        if p.is_dir() and (p / "llm_ahp_weights.csv").exists()
    ]

    if not scenario_dirs:
        raise FileNotFoundError(f"没有找到多情景权重目录：{WEIGHT_ROOT}")

    summaries = []

    for scenario_dir in sorted(scenario_dirs):
        summary = run_overlay_for_scenario(scenario_dir, arrays, meta, valid_mask)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OVERLAY_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("多情景加权叠加汇总")
    print("=" * 60)

    print(
        summary_df[
            [
                "scenario_id",
                "scenario_name_cn",
                "weight_poi",
                "weight_road",
                "weight_substation",
                "weight_landuse",
                "weight_slope",
                "score_mean",
                "score_std",
                "high_suitability_area_km2",
            ]
        ]
    )

    print("\n已输出汇总：", OVERLAY_SUMMARY_CSV)


if __name__ == "__main__":
    main()

"""LLM-AHP 多情景对比分析脚本"""
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

BASE_DIR = Path(r"D:\Paper\毕设")

BOUNDARY = BASE_DIR / "data" / "boundary" / "study_area_utm.geojson"

WEIGHT_SUMMARY = (
    BASE_DIR
    / "results"
    / "weights"
    / "llm_scenarios"
    / "llm_scenario_weights_summary.csv"
)
OVERLAY_SUMMARY = (
    BASE_DIR
    / "results"
    / "tables"
    / "llm_scenarios"
    / "llm_scenario_overlay_summary.csv"
)

SCENARIO_RESULT_ROOT = BASE_DIR / "results" / "suitability" / "llm_scenarios"

OUT_TABLE_DIR = BASE_DIR / "results" / "tables" / "llm_scenarios"
OUT_MAP_DIR = BASE_DIR / "results" / "maps" / "llm_scenarios"
OUT_COMPARE_DIR = BASE_DIR / "results" / "comparison" / "llm_scenarios"

OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_MAP_DIR.mkdir(parents=True, exist_ok=True)
OUT_COMPARE_DIR.mkdir(parents=True, exist_ok=True)

WEIGHT_BAR = OUT_MAP_DIR / "llm_scenario_weight_comparison.png"
CLASS_3PANEL = OUT_MAP_DIR / "llm_scenario_class_3panel.png"
HIGH_FREQ_MAP = OUT_MAP_DIR / "llm_scenario_high_suitability_frequency.png"
PAIRWISE_CSV = OUT_TABLE_DIR / "llm_scenario_pairwise_comparison.csv"
HIGH_FREQ_TIF = OUT_COMPARE_DIR / "high_suitability_frequency.tif"

SCENARIO_LABELS = {
    "scenario_01_demand_priority": "需求优先型",
    "scenario_02_traffic_priority": "交通优先型",
    "scenario_03_construction_constraint_priority": "建设约束优先型",
}

FACTOR_LABELS = {
    "weight_poi": "POI密度",
    "weight_road": "道路距离",
    "weight_substation": "变电站距离",
    "weight_landuse": "土地利用",
    "weight_slope": "坡度",
}

plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def read_class_raster(scenario_id):
    path = SCENARIO_RESULT_ROOT / scenario_id / "suitability_class.tif"

    if not path.exists():
        raise FileNotFoundError(f"找不到情景分级图：{path}")

    with rasterio.open(path) as src:
        arr = src.read(1)
        meta = src.meta.copy()
        nodata = src.nodata
        bounds = src.bounds
        crs = src.crs

    valid = arr > 0
    if nodata is not None:
        valid &= arr != nodata

    return arr, valid, meta, bounds, crs


def get_extent(bounds):
    return [bounds.left, bounds.right, bounds.bottom, bounds.top]


def plot_weight_comparison():
    print("\n=== 绘制多情景权重对比图 ===")

    if not WEIGHT_SUMMARY.exists():
        raise FileNotFoundError(f"找不到权重汇总表：{WEIGHT_SUMMARY}")

    df = pd.read_csv(WEIGHT_SUMMARY)

    scenario_ids = list(SCENARIO_LABELS.keys())
    factor_cols = list(FACTOR_LABELS.keys())

    x = np.arange(len(factor_cols))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, scenario_id in enumerate(scenario_ids):
        row = df[df["scenario_id"] == scenario_id].iloc[0]
        values = [row[col] for col in factor_cols]
        ax.bar(
            x + (i - 1) * width, values, width=width, label=SCENARIO_LABELS[scenario_id]
        )

    ax.set_xticks(x)
    ax.set_xticklabels([FACTOR_LABELS[c] for c in factor_cols])
    ax.set_ylabel("AHP 权重")
    ax.set_title("LLM-AHP 多自然语言情景权重对比")
    ax.legend()
    ax.set_ylim(0, 0.5)

    plt.tight_layout()
    plt.savefig(WEIGHT_BAR, dpi=300, bbox_inches="tight")
    plt.close()

    print("已输出：", WEIGHT_BAR)


def plot_class_3panel():
    print("\n=== 绘制多情景适宜性分级对比图 ===")

    scenario_ids = list(SCENARIO_LABELS.keys())

    colors = [
        "#d73027",
        "#fc8d59",
        "#fee08b",
        "#d9ef8b",
        "#1a9850",
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, scenario_id in zip(axes, scenario_ids):
        arr, valid, meta, bounds, crs = read_class_raster(scenario_id)

        display = np.where(valid, arr, np.nan)

        boundary = gpd.read_file(BOUNDARY).to_crs(crs)

        ax.imshow(
            display, extent=get_extent(bounds), origin="upper", cmap=cmap, norm=norm
        )

        boundary.boundary.plot(ax=ax, linewidth=1.0, edgecolor="black")

        ax.set_title(SCENARIO_LABELS[scenario_id], fontsize=13)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")

    legend_items = [
        Patch(facecolor=colors[0], label="1 不适宜区"),
        Patch(facecolor=colors[1], label="2 较低适宜区"),
        Patch(facecolor=colors[2], label="3 中等适宜区"),
        Patch(facecolor=colors[3], label="4 较高适宜区"),
        Patch(facecolor=colors[4], label="5 高适宜区"),
    ]

    fig.legend(
        handles=legend_items, loc="lower center", ncol=5, frameon=True, fontsize=10
    )

    plt.suptitle("LLM-AHP 多自然语言情景适宜性分级对比", fontsize=16)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(CLASS_3PANEL, dpi=300, bbox_inches="tight")
    plt.close()

    print("已输出：", CLASS_3PANEL)


def pairwise_comparison():
    print("\n=== 计算多情景两两对比 ===")

    scenario_ids = list(SCENARIO_LABELS.keys())

    class_data = {}
    valid_data = {}
    meta_ref = None

    for scenario_id in scenario_ids:
        arr, valid, meta, bounds, crs = read_class_raster(scenario_id)
        class_data[scenario_id] = arr
        valid_data[scenario_id] = valid

        if meta_ref is None:
            meta_ref = meta
        else:
            if (
                meta_ref["width"] != meta["width"]
                or meta_ref["height"] != meta["height"]
                or meta_ref["transform"] != meta["transform"]
                or meta_ref["crs"] != meta["crs"]
            ):
                raise ValueError("多情景栅格不对齐。")

    pixel_area_m2 = abs(meta_ref["transform"].a * meta_ref["transform"].e)
    pixel_area_km2 = pixel_area_m2 / 1_000_000

    rows = []

    for i in range(len(scenario_ids)):
        for j in range(i + 1, len(scenario_ids)):
            s1 = scenario_ids[i]
            s2 = scenario_ids[j]

            arr1 = class_data[s1]
            arr2 = class_data[s2]

            valid = valid_data[s1] & valid_data[s2]

            same = valid & (arr1 == arr2)
            changed = valid & (arr1 != arr2)

            high1 = valid & (arr1 == 5)
            high2 = valid & (arr2 == 5)

            overlap = high1 & high2
            union = high1 | high2

            total_pixels = int(valid.sum())
            same_pixels = int(same.sum())
            changed_pixels = int(changed.sum())

            rows.append(
                {
                    "scenario_1": s1,
                    "scenario_1_cn": SCENARIO_LABELS[s1],
                    "scenario_2": s2,
                    "scenario_2_cn": SCENARIO_LABELS[s2],
                    "total_pixels": total_pixels,
                    "same_class_pixels": same_pixels,
                    "changed_class_pixels": changed_pixels,
                    "same_class_ratio": (
                        same_pixels / total_pixels if total_pixels else np.nan
                    ),
                    "changed_class_ratio": (
                        changed_pixels / total_pixels if total_pixels else np.nan
                    ),
                    "changed_area_km2": changed_pixels * pixel_area_km2,
                    "high_overlap_area_km2": float(overlap.sum() * pixel_area_km2),
                    "high_union_area_km2": float(union.sum() * pixel_area_km2),
                    "high_jaccard": (
                        float(overlap.sum() / union.sum()) if union.sum() else np.nan
                    ),
                    "scenario_1_high_area_km2": float(high1.sum() * pixel_area_km2),
                    "scenario_2_high_area_km2": float(high2.sum() * pixel_area_km2),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(PAIRWISE_CSV, index=False, encoding="utf-8-sig")

    print("已输出：", PAIRWISE_CSV)
    print("\n两两对比结果：")
    print(
        df[
            [
                "scenario_1_cn",
                "scenario_2_cn",
                "same_class_ratio",
                "changed_class_ratio",
                "changed_area_km2",
                "high_overlap_area_km2",
                "high_jaccard",
            ]
        ]
    )

    return class_data, valid_data, meta_ref, bounds, crs


def high_frequency_analysis(class_data, valid_data, meta, bounds, crs):
    print("\n=== 计算高适宜区出现频次 ===")

    scenario_ids = list(SCENARIO_LABELS.keys())

    valid_all = None
    high_count = None

    for scenario_id in scenario_ids:
        arr = class_data[scenario_id]
        valid = valid_data[scenario_id]

        if high_count is None:
            high_count = np.zeros(arr.shape, dtype="uint8")
            valid_all = valid.copy()
        else:
            valid_all &= valid

        high_count += ((arr == 5) & valid).astype("uint8")

    high_count[~valid_all] = 0

    out_meta = meta.copy()
    out_meta.update({"dtype": "uint8", "nodata": 0, "compress": "lzw"})

    with rasterio.open(HIGH_FREQ_TIF, "w", **out_meta) as dst:
        dst.write(high_count, 1)

    print("已输出：", HIGH_FREQ_TIF)

    unique, counts = np.unique(high_count[valid_all], return_counts=True)

    pixel_area_m2 = abs(meta["transform"].a * meta["transform"].e)
    pixel_area_km2 = pixel_area_m2 / 1_000_000

    freq_rows = []
    for u, c in zip(unique, counts):
        freq_rows.append(
            {
                "high_frequency": int(u),
                "pixel_count": int(c),
                "area_km2": float(c * pixel_area_km2),
                "ratio": float(c / valid_all.sum()),
            }
        )

    freq_df = pd.DataFrame(freq_rows)
    freq_csv = OUT_TABLE_DIR / "high_suitability_frequency_stats.csv"
    freq_df.to_csv(freq_csv, index=False, encoding="utf-8-sig")

    print("已输出：", freq_csv)
    print("\n高适宜频次统计：")
    print(freq_df)

    boundary = gpd.read_file(BOUNDARY).to_crs(crs)

    display = np.where(valid_all & (high_count > 0), high_count, np.nan)

    colors = [
        "#fee08b",  # 1
        "#66bd63",  # 2
        "#1a9850",  # 3
    ]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(10, 9))

    ax.imshow(display, extent=get_extent(bounds), origin="upper", cmap=cmap, norm=norm)

    boundary.boundary.plot(ax=ax, linewidth=1.2, edgecolor="black")

    legend_items = [
        Patch(facecolor=colors[0], label="1 个情景为高适宜"),
        Patch(facecolor=colors[1], label="2 个情景为高适宜"),
        Patch(facecolor=colors[2], label="3 个情景均为高适宜"),
    ]

    ax.legend(handles=legend_items, loc="lower left", frameon=True, fontsize=9)

    ax.set_title("LLM-AHP 多情景高适宜区出现频次", fontsize=14)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(HIGH_FREQ_MAP, dpi=300, bbox_inches="tight")
    plt.close()

    print("已输出：", HIGH_FREQ_MAP)


def main():
    plot_weight_comparison()
    plot_class_3panel()

    class_data, valid_data, meta, bounds, crs = pairwise_comparison()

    high_frequency_analysis(class_data, valid_data, meta, bounds, crs)

    print("\nLLM 多情景对比分析完成。")


if __name__ == "__main__":
    main()

# 对比 baseline 和 manual AHP 的适宜性结果，分析差异并可视化
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import geopandas as gpd

BASE_DIR = Path(r"D:\Paper\毕设")

BOUNDARY = BASE_DIR / "data" / "boundary" / "study_area_utm.geojson"

BASELINE_SCORE = BASE_DIR / "results" / "suitability" / "suitability_score.tif"
BASELINE_CLASS = BASE_DIR / "results" / "suitability" / "suitability_class.tif"

AHP_SCORE = (
    BASE_DIR / "results" / "suitability" / "manual_ahp" / "suitability_score.tif"
)
AHP_CLASS = (
    BASE_DIR / "results" / "suitability" / "manual_ahp" / "suitability_class.tif"
)

COMPARE_DIR = BASE_DIR / "results" / "comparison" / "baseline_vs_manual_ahp"
MAP_DIR = BASE_DIR / "results" / "maps" / "comparison"

COMPARE_DIR.mkdir(parents=True, exist_ok=True)
MAP_DIR.mkdir(parents=True, exist_ok=True)

SCORE_DIFF_TIF = COMPARE_DIR / "score_difference_ahp_minus_baseline.tif"
CLASS_DIFF_TIF = COMPARE_DIR / "class_difference_ahp_minus_baseline.tif"
HIGH_OVERLAP_TIF = COMPARE_DIR / "high_suitability_overlap.tif"

SUMMARY_CSV = COMPARE_DIR / "comparison_summary.csv"
TRANSITION_CSV = COMPARE_DIR / "class_transition_matrix.csv"

SCORE_DIFF_MAP = MAP_DIR / "score_difference_map.png"
CLASS_DIFF_MAP = MAP_DIR / "class_difference_map.png"
HIGH_OVERLAP_MAP = MAP_DIR / "high_suitability_overlap_map.png"

plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        meta = src.meta.copy()
        nodata = src.nodata
        bounds = src.bounds
        crs = src.crs

    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata

    return arr, valid, meta, nodata, bounds, crs


def save_float_raster(arr, meta, out_path, nodata=-9999.0):
    out_meta = meta.copy()
    out_meta.update({"dtype": "float32", "nodata": nodata, "compress": "lzw"})

    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(arr.astype("float32"), 1)

    print("已输出：", out_path)


def save_int_raster(arr, meta, out_path, dtype="int16", nodata=-9999):
    out_meta = meta.copy()
    out_meta.update({"dtype": dtype, "nodata": nodata, "compress": "lzw"})

    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(arr.astype(dtype), 1)

    print("已输出：", out_path)


def check_alignment(meta1, meta2):
    keys = ["width", "height", "transform", "crs"]
    for k in keys:
        if meta1[k] != meta2[k]:
            raise ValueError(f"栅格不对齐：{k} 不一致。")


def get_extent(bounds):
    return [bounds.left, bounds.right, bounds.bottom, bounds.top]


def make_comparison():
    baseline_score, valid_bs, meta_score, nodata_bs, bounds, crs = read_raster(
        BASELINE_SCORE
    )
    ahp_score, valid_as, meta_ahp_score, nodata_as, _, _ = read_raster(AHP_SCORE)

    baseline_class, valid_bc, meta_class, nodata_bc, _, _ = read_raster(BASELINE_CLASS)
    ahp_class, valid_ac, meta_ahp_class, nodata_ac, _, _ = read_raster(AHP_CLASS)

    check_alignment(meta_score, meta_ahp_score)
    check_alignment(meta_class, meta_ahp_class)

    valid_score = valid_bs & valid_as
    valid_class = valid_bc & valid_ac & (baseline_class > 0) & (ahp_class > 0)

    print("有效得分像元数：", int(valid_score.sum()))
    print("有效分级像元数：", int(valid_class.sum()))

    # 1. 得分差异：AHP - baseline
    score_diff = np.full(baseline_score.shape, -9999.0, dtype="float32")
    score_diff[valid_score] = ahp_score[valid_score] - baseline_score[valid_score]

    save_float_raster(score_diff, meta_score, SCORE_DIFF_TIF)

    # 2. 等级差异：AHP等级 - baseline等级
    class_diff = np.full(baseline_class.shape, -9999, dtype="int16")
    class_diff[valid_class] = ahp_class[valid_class].astype("int16") - baseline_class[
        valid_class
    ].astype("int16")

    save_int_raster(class_diff, meta_class, CLASS_DIFF_TIF, dtype="int16", nodata=-9999)

    # 3. 高适宜区重叠
    baseline_high = valid_class & (baseline_class == 5)
    ahp_high = valid_class & (ahp_class == 5)

    overlap = baseline_high & ahp_high
    baseline_only = baseline_high & (~ahp_high)
    ahp_only = ahp_high & (~baseline_high)
    union = baseline_high | ahp_high

    # 0: 非高适宜区
    # 1: baseline only
    # 2: AHP only
    # 3: both
    overlap_arr = np.zeros(baseline_class.shape, dtype="uint8")
    overlap_arr[baseline_only] = 1
    overlap_arr[ahp_only] = 2
    overlap_arr[overlap] = 3

    overlap_meta = meta_class.copy()
    overlap_meta.update({"dtype": "uint8", "nodata": 0, "compress": "lzw"})

    with rasterio.open(HIGH_OVERLAP_TIF, "w", **overlap_meta) as dst:
        dst.write(overlap_arr, 1)

    print("已输出：", HIGH_OVERLAP_TIF)

    # 面积计算
    pixel_area_m2 = abs(meta_score["transform"].a * meta_score["transform"].e)
    pixel_area_km2 = pixel_area_m2 / 1_000_000

    total_valid = int(valid_class.sum())
    total_area_km2 = total_valid * pixel_area_km2

    same_class = valid_class & (baseline_class == ahp_class)
    changed_class = valid_class & (baseline_class != ahp_class)

    summary = {
        "total_valid_pixels": total_valid,
        "total_area_km2": total_area_km2,
        "score_diff_min": float(score_diff[valid_score].min()),
        "score_diff_max": float(score_diff[valid_score].max()),
        "score_diff_mean": float(score_diff[valid_score].mean()),
        "score_diff_abs_mean": float(np.abs(score_diff[valid_score]).mean()),
        "same_class_pixels": int(same_class.sum()),
        "changed_class_pixels": int(changed_class.sum()),
        "same_class_ratio": float(same_class.sum() / total_valid),
        "changed_class_ratio": float(changed_class.sum() / total_valid),
        "baseline_high_area_km2": float(baseline_high.sum() * pixel_area_km2),
        "ahp_high_area_km2": float(ahp_high.sum() * pixel_area_km2),
        "high_overlap_area_km2": float(overlap.sum() * pixel_area_km2),
        "baseline_only_high_area_km2": float(baseline_only.sum() * pixel_area_km2),
        "ahp_only_high_area_km2": float(ahp_only.sum() * pixel_area_km2),
        "high_union_area_km2": float(union.sum() * pixel_area_km2),
        "high_jaccard": (
            float(overlap.sum() / union.sum()) if union.sum() > 0 else np.nan
        ),
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\n=== 对比汇总 ===")
    print(summary_df.T)
    print("已输出：", SUMMARY_CSV)

    # 4. 等级转移矩阵
    matrix = np.zeros((5, 5), dtype=int)

    for i in range(1, 6):
        for j in range(1, 6):
            matrix[i - 1, j - 1] = int(
                ((baseline_class == i) & (ahp_class == j) & valid_class).sum()
            )

    transition_df = pd.DataFrame(
        matrix,
        index=[f"baseline_{i}" for i in range(1, 6)],
        columns=[f"ahp_{j}" for j in range(1, 6)],
    )

    transition_area_df = transition_df * pixel_area_km2

    # 保存像元数和面积放在同一个 csv 中
    transition_df.to_csv(TRANSITION_CSV, encoding="utf-8-sig")

    area_csv = COMPARE_DIR / "class_transition_matrix_area_km2.csv"
    transition_area_df.to_csv(area_csv, encoding="utf-8-sig")

    print("\n=== 等级转移矩阵：像元数 ===")
    print(transition_df)
    print("已输出：", TRANSITION_CSV)
    print("已输出面积矩阵：", area_csv)

    return {
        "score_diff": score_diff,
        "class_diff": class_diff,
        "overlap_arr": overlap_arr,
        "bounds": bounds,
        "crs": crs,
        "summary": summary,
    }


def plot_score_diff(result):
    print("\n=== 绘制得分差异图 ===")

    arr = result["score_diff"]
    bounds = result["bounds"]
    crs = result["crs"]

    display = np.where(arr != -9999.0, arr, np.nan)

    boundary = gpd.read_file(BOUNDARY).to_crs(crs)

    vmax = np.nanmax(np.abs(display))
    vmax = max(vmax, 0.01)

    fig, ax = plt.subplots(figsize=(10, 9))

    im = ax.imshow(
        display,
        extent=get_extent(bounds),
        origin="upper",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )

    boundary.boundary.plot(ax=ax, linewidth=1.2, edgecolor="black")

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("AHP Score - Baseline Score")

    ax.set_title("AHP与Baseline综合得分差异图", fontsize=14)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(SCORE_DIFF_MAP, dpi=300, bbox_inches="tight")
    plt.close()

    print("已输出：", SCORE_DIFF_MAP)


def plot_class_diff(result):
    print("\n=== 绘制等级差异图 ===")

    arr = result["class_diff"]
    bounds = result["bounds"]
    crs = result["crs"]

    display = np.where(arr != -9999, arr, np.nan)

    boundary = gpd.read_file(BOUNDARY).to_crs(crs)

    # 差异范围一般是 -4 到 4
    colors = [
        "#2166ac",
        "#67a9cf",
        "#d1e5f0",
        "#f7f7f7",
        "#fddbc7",
        "#ef8a62",
        "#b2182b",
    ]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    clipped = np.clip(display, -3, 3)

    fig, ax = plt.subplots(figsize=(10, 9))

    im = ax.imshow(
        clipped, extent=get_extent(bounds), origin="upper", cmap=cmap, norm=norm
    )

    boundary.boundary.plot(ax=ax, linewidth=1.2, edgecolor="black")

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("AHP Class - Baseline Class")

    ax.set_title("AHP与Baseline适宜性等级差异图", fontsize=14)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(CLASS_DIFF_MAP, dpi=300, bbox_inches="tight")
    plt.close()

    print("已输出：", CLASS_DIFF_MAP)


def plot_high_overlap(result):
    print("\n=== 绘制高适宜区重叠图 ===")

    arr = result["overlap_arr"]
    bounds = result["bounds"]
    crs = result["crs"]

    display = np.where(arr > 0, arr, np.nan)

    boundary = gpd.read_file(BOUNDARY).to_crs(crs)

    colors = [
        "#fdae61",  # 1 baseline only
        "#2c7bb6",  # 2 AHP only
        "#1a9850",  # 3 both
    ]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(10, 9))

    ax.imshow(display, extent=get_extent(bounds), origin="upper", cmap=cmap, norm=norm)

    boundary.boundary.plot(ax=ax, linewidth=1.2, edgecolor="black")

    legend_items = [
        Patch(facecolor=colors[0], label="仅Baseline高适宜"),
        Patch(facecolor=colors[1], label="仅AHP高适宜"),
        Patch(facecolor=colors[2], label="二者共同高适宜"),
    ]

    ax.legend(handles=legend_items, loc="lower left", frameon=True, fontsize=9)

    ax.set_title("Baseline与AHP高适宜区重叠图", fontsize=14)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(HIGH_OVERLAP_MAP, dpi=300, bbox_inches="tight")
    plt.close()

    print("已输出：", HIGH_OVERLAP_MAP)


def main():
    for path in [BASELINE_SCORE, BASELINE_CLASS, AHP_SCORE, AHP_CLASS]:
        if not path.exists():
            raise FileNotFoundError(f"找不到文件：{path}")

    result = make_comparison()

    plot_score_diff(result)
    plot_class_diff(result)
    plot_high_overlap(result)

    print("\nbaseline 与 manual AHP 对比完成。")


if __name__ == "__main__":
    main()

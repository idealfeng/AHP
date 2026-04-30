from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

BASE_DIR = Path(r"D:\Paper\毕设")

SCENARIO_NAME = "manual_ahp"

BOUNDARY = BASE_DIR / "data" / "boundary" / "study_area_utm.geojson"

SCORE_TIF = (
    BASE_DIR / "results" / "suitability" / SCENARIO_NAME / "suitability_score.tif"
)
CLASS_TIF = (
    BASE_DIR / "results" / "suitability" / SCENARIO_NAME / "suitability_class.tif"
)

MAP_DIR = BASE_DIR / "results" / "maps" / SCENARIO_NAME
MAP_DIR.mkdir(parents=True, exist_ok=True)

SCORE_MAP = MAP_DIR / "suitability_score_map.png"
CLASS_MAP = MAP_DIR / "suitability_class_map.png"

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
        nodata = src.nodata
        bounds = src.bounds
        crs = src.crs

    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata

    return arr, valid, bounds, crs


def get_extent(bounds):
    return [bounds.left, bounds.right, bounds.bottom, bounds.top]


def plot_score_map():
    print("\n=== 绘制 manual_ahp 综合适宜性得分图 ===")

    arr, valid, bounds, crs = read_raster(SCORE_TIF)
    boundary = gpd.read_file(BOUNDARY).to_crs(crs)

    display = np.where(valid, arr, np.nan)

    fig, ax = plt.subplots(figsize=(10, 9))

    im = ax.imshow(display, extent=get_extent(bounds), origin="upper", cmap="YlOrRd")

    boundary.boundary.plot(ax=ax, linewidth=1.2, edgecolor="black")

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Suitability Score")

    ax.set_title("AHP综合适宜性得分图 / AHP Suitability Score", fontsize=14)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(SCORE_MAP, dpi=300, bbox_inches="tight")
    plt.close()

    values = arr[valid]

    print("已输出：", SCORE_MAP)
    print(f"得分范围：{values.min():.4f} - {values.max():.4f}")
    print(f"得分均值：{values.mean():.4f}")


def plot_class_map():
    print("\n=== 绘制 manual_ahp 适宜性等级图 ===")

    arr, valid, bounds, crs = read_raster(CLASS_TIF)
    boundary = gpd.read_file(BOUNDARY).to_crs(crs)

    display = np.where(valid & (arr > 0), arr, np.nan)

    colors = [
        "#d73027",
        "#fc8d59",
        "#fee08b",
        "#d9ef8b",
        "#1a9850",
    ]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)

    fig, ax = plt.subplots(figsize=(10, 9))

    ax.imshow(display, extent=get_extent(bounds), origin="upper", cmap=cmap, norm=norm)

    boundary.boundary.plot(ax=ax, linewidth=1.2, edgecolor="black")

    legend_items = [
        Patch(facecolor=colors[0], label="1 不适宜区"),
        Patch(facecolor=colors[1], label="2 较低适宜区"),
        Patch(facecolor=colors[2], label="3 中等适宜区"),
        Patch(facecolor=colors[3], label="4 较高适宜区"),
        Patch(facecolor=colors[4], label="5 高适宜区"),
    ]

    ax.legend(handles=legend_items, loc="lower left", frameon=True, fontsize=9)

    ax.set_title("AHP适宜性分级图 / AHP Suitability Classification", fontsize=14)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(CLASS_MAP, dpi=300, bbox_inches="tight")
    plt.close()

    print("已输出：", CLASS_MAP)

    unique, counts = np.unique(arr[valid & (arr > 0)], return_counts=True)
    print("等级统计：")
    for u, c in zip(unique, counts):
        print(f"等级 {int(u)}: {int(c)} 个像元")


if __name__ == "__main__":
    if not SCORE_TIF.exists():
        raise FileNotFoundError(f"找不到：{SCORE_TIF}")

    if not CLASS_TIF.exists():
        raise FileNotFoundError(f"找不到：{CLASS_TIF}")

    plot_score_map()
    plot_class_map()

    print("\nmanual_ahp 综合适宜性出图完成。")

"""综合适宜性出图脚本：根据加权叠加分析结果，绘制综合适宜性得分图和适宜性分级图，并统计相关信息。"""
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

BASE_DIR = Path(r"D:\Paper\毕设")

BOUNDARY = BASE_DIR / "data" / "boundary" / "study_area_utm.geojson"

SCORE_TIF = BASE_DIR / "results" / "suitability" / "suitability_score.tif"
CLASS_TIF = BASE_DIR / "results" / "suitability" / "suitability_class.tif"

MAP_DIR = BASE_DIR / "results" / "maps"
MAP_DIR.mkdir(parents=True, exist_ok=True)

SCORE_MAP = MAP_DIR / "suitability_score_map.png"
CLASS_MAP = MAP_DIR / "suitability_class_map.png"

# 中文显示设置。如果你的环境没有这些字体，标题可能显示为方框，不影响数据结果。
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

    if nodata is not None:
        valid = (arr != nodata) & np.isfinite(arr)
    else:
        valid = np.isfinite(arr)

    return arr, valid, bounds, crs, nodata


def get_extent(bounds):
    return [bounds.left, bounds.right, bounds.bottom, bounds.top]


def plot_score_map():
    print("\n=== 绘制综合适宜性得分图 ===")

    arr, valid, bounds, crs, nodata = read_raster(SCORE_TIF)
    boundary = gpd.read_file(BOUNDARY).to_crs(crs)

    display = np.where(valid, arr, np.nan)

    fig, ax = plt.subplots(figsize=(10, 9))

    im = ax.imshow(display, extent=get_extent(bounds), origin="upper", cmap="YlOrRd")

    boundary.boundary.plot(ax=ax, linewidth=1.2, edgecolor="black")

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Suitability Score")

    ax.set_title("综合适宜性得分图 / Suitability Score", fontsize=14)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(SCORE_MAP, dpi=300, bbox_inches="tight")
    plt.close()

    print("已输出：", SCORE_MAP)

    values = arr[valid]
    print(f"得分范围：{values.min():.4f} - {values.max():.4f}")
    print(f"得分均值：{values.mean():.4f}")


def plot_class_map():
    print("\n=== 绘制适宜性等级图 ===")

    arr, valid, bounds, crs, nodata = read_raster(CLASS_TIF)
    boundary = gpd.read_file(BOUNDARY).to_crs(crs)

    display = np.where(valid & (arr > 0), arr, np.nan)

    # 1-5 五个等级
    colors = [
        "#d73027",  # 1 不适宜区
        "#fc8d59",  # 2 较低适宜区
        "#fee08b",  # 3 中等适宜区
        "#d9ef8b",  # 4 较高适宜区
        "#1a9850",  # 5 高适宜区
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

    ax.set_title("适宜性分级图 / Suitability Classification", fontsize=14)
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
        raise FileNotFoundError(f"找不到综合适宜性得分图：{SCORE_TIF}")

    if not CLASS_TIF.exists():
        raise FileNotFoundError(f"找不到适宜性分级图：{CLASS_TIF}")

    plot_score_map()
    plot_class_map()

    print("\n综合适宜性出图完成。")

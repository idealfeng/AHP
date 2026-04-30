# 从 llm_ahp 的适宜性分级图中提取高适宜区，生成候选区面和中心点，并计算统计指标
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from rasterio.mask import mask
from shapely.geometry import shape, mapping
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

BASE_DIR = Path(r"D:\Paper\毕设")

SCENARIO_NAME = "llm_ahp"

BOUNDARY = BASE_DIR / "data" / "boundary" / "study_area_utm.geojson"

SCORE_TIF = (
    BASE_DIR / "results" / "suitability" / SCENARIO_NAME / "suitability_score.tif"
)
CLASS_TIF = (
    BASE_DIR / "results" / "suitability" / SCENARIO_NAME / "suitability_class.tif"
)

CANDIDATE_DIR = BASE_DIR / "results" / "candidates" / SCENARIO_NAME
TABLE_DIR = BASE_DIR / "results" / "tables" / SCENARIO_NAME
MAP_DIR = BASE_DIR / "results" / "maps" / SCENARIO_NAME

CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
MAP_DIR.mkdir(parents=True, exist_ok=True)

HIGH_AREAS_UTM = CANDIDATE_DIR / "high_suitability_areas.geojson"
HIGH_AREAS_WGS84 = CANDIDATE_DIR / "high_suitability_areas_wgs84.geojson"
CANDIDATE_POINTS_UTM = CANDIDATE_DIR / "candidate_points.geojson"
CANDIDATE_POINTS_WGS84 = CANDIDATE_DIR / "candidate_points_wgs84.geojson"

STATS_CSV = TABLE_DIR / "candidate_sites_stats.csv"
CANDIDATE_MAP = MAP_DIR / "candidate_sites_map.png"

MIN_AREA_M2 = 30000

plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def raster_to_high_polygons():
    print("\n=== 提取 llm_ahp 高适宜区 ===")

    with rasterio.open(CLASS_TIF) as src:
        class_arr = src.read(1)
        transform = src.transform
        crs = src.crs

    high_mask = class_arr == 5
    high_pixel_count = int(high_mask.sum())

    print("高适宜区像元数：", high_pixel_count)

    if high_pixel_count == 0:
        raise ValueError("没有 class = 5 的高适宜区像元。")

    results = shapes(high_mask.astype("uint8"), mask=high_mask, transform=transform)

    polygons = []

    for geom, value in results:
        if int(value) == 1:
            polygons.append(shape(geom))

    gdf = gpd.GeoDataFrame({"class": [5] * len(polygons)}, geometry=polygons, crs=crs)

    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf["area_m2"] = gdf.geometry.area

    before_count = len(gdf)
    gdf = gdf[gdf["area_m2"] >= MIN_AREA_M2].copy()
    after_count = len(gdf)

    print(f"过滤前斑块数：{before_count}")
    print(f"过滤后斑块数：{after_count}")
    print(f"最小面积阈值：{MIN_AREA_M2} m²")

    if gdf.empty:
        raise ValueError("面积过滤后没有候选区。可以降低 MIN_AREA_M2。")

    gdf = gdf.sort_values("area_m2", ascending=False).reset_index(drop=True)
    gdf["candidate_id"] = range(1, len(gdf) + 1)
    gdf["area_km2"] = gdf["area_m2"] / 1_000_000

    return gdf


def calculate_zonal_score(gdf):
    print("\n=== 计算候选区平均适宜性得分 ===")

    mean_scores = []
    max_scores = []
    min_scores = []

    with rasterio.open(SCORE_TIF) as src:
        nodata = src.nodata

        for geom in gdf.geometry:
            out_image, _ = mask(src, [mapping(geom)], crop=True, nodata=nodata)

            arr = out_image[0].astype("float32")

            valid = np.isfinite(arr)
            if nodata is not None:
                valid &= arr != nodata

            values = arr[valid]

            if values.size == 0:
                mean_scores.append(np.nan)
                max_scores.append(np.nan)
                min_scores.append(np.nan)
            else:
                mean_scores.append(float(values.mean()))
                max_scores.append(float(values.max()))
                min_scores.append(float(values.min()))

    gdf["mean_score"] = mean_scores
    gdf["max_score"] = max_scores
    gdf["min_score"] = min_scores

    return gdf


def create_candidate_points(gdf):
    print("\n=== 生成候选区中心点 ===")

    points = gdf.copy()
    points["geometry"] = points.geometry.representative_point()

    points_wgs = points.to_crs(epsg=4326)

    points["center_x"] = points.geometry.x
    points["center_y"] = points.geometry.y
    points["lon"] = points_wgs.geometry.x.values
    points["lat"] = points_wgs.geometry.y.values

    return points


def save_outputs(areas, points):
    print("\n=== 保存 llm_ahp 候选区结果 ===")

    areas.to_file(HIGH_AREAS_UTM, driver="GeoJSON", encoding="utf-8")
    areas.to_crs(epsg=4326).to_file(
        HIGH_AREAS_WGS84, driver="GeoJSON", encoding="utf-8"
    )

    points.to_file(CANDIDATE_POINTS_UTM, driver="GeoJSON", encoding="utf-8")
    points.to_crs(epsg=4326).to_file(
        CANDIDATE_POINTS_WGS84, driver="GeoJSON", encoding="utf-8"
    )

    stats_cols = [
        "candidate_id",
        "area_m2",
        "area_km2",
        "mean_score",
        "max_score",
        "min_score",
        "center_x",
        "center_y",
        "lon",
        "lat",
    ]

    stats = points[stats_cols].copy()

    stats = stats.sort_values(
        ["mean_score", "area_km2"], ascending=[False, False]
    ).reset_index(drop=True)

    stats.to_csv(STATS_CSV, index=False, encoding="utf-8-sig")

    print("已输出高适宜区面：", HIGH_AREAS_UTM)
    print("已输出高适宜区面 WGS84：", HIGH_AREAS_WGS84)
    print("已输出候选点：", CANDIDATE_POINTS_UTM)
    print("已输出候选点 WGS84：", CANDIDATE_POINTS_WGS84)
    print("已输出统计表：", STATS_CSV)

    print("\n候选区数量：", len(areas))
    print(f"候选区总面积：{areas['area_km2'].sum():.2f} km²")
    print("\n前 10 个候选区：")
    print(stats.head(10))

    return stats


def plot_candidate_map(areas, points):
    print("\n=== 绘制 llm_ahp 候选区分布图 ===")

    with rasterio.open(CLASS_TIF) as src:
        class_arr = src.read(1)
        nodata = src.nodata
        bounds = src.bounds
        crs = src.crs

    valid = class_arr > 0
    if nodata is not None:
        valid &= class_arr != nodata

    display = np.where(valid, class_arr, np.nan)

    boundary = gpd.read_file(BOUNDARY).to_crs(crs)
    areas = areas.to_crs(crs)
    points = points.to_crs(crs)

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

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

    ax.imshow(display, extent=extent, origin="upper", cmap=cmap, norm=norm, alpha=0.9)

    boundary.boundary.plot(ax=ax, linewidth=1.2, edgecolor="black")

    areas.boundary.plot(ax=ax, linewidth=0.8, edgecolor="blue")

    # 只显示前 20 个候选区中心点，避免图太乱
    top_points = points.sort_values(
        ["mean_score", "area_km2"], ascending=[False, False]
    ).head(20)

    top_points.plot(ax=ax, markersize=18, color="blue")

    legend_items = [
        Patch(facecolor=colors[0], label="1 不适宜区"),
        Patch(facecolor=colors[1], label="2 较低适宜区"),
        Patch(facecolor=colors[2], label="3 中等适宜区"),
        Patch(facecolor=colors[3], label="4 较高适宜区"),
        Patch(facecolor=colors[4], label="5 高适宜区"),
        Patch(facecolor="blue", label="高适宜区边界 / 前20候选点"),
    ]

    ax.legend(handles=legend_items, loc="lower left", frameon=True, fontsize=8)

    ax.set_title("AHP高适宜区与候选点分布图 / AHP Candidate Sites", fontsize=14)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(CANDIDATE_MAP, dpi=300, bbox_inches="tight")
    plt.close()

    print("已输出候选区地图：", CANDIDATE_MAP)


def main():
    if not CLASS_TIF.exists():
        raise FileNotFoundError(f"找不到适宜性分级图：{CLASS_TIF}")

    if not SCORE_TIF.exists():
        raise FileNotFoundError(f"找不到综合适宜性得分图：{SCORE_TIF}")

    areas = raster_to_high_polygons()
    areas = calculate_zonal_score(areas)
    points = create_candidate_points(areas)

    save_outputs(areas, points)
    plot_candidate_map(areas, points)

    print("\nllm_ahp 高适宜区提取完成。")


if __name__ == "__main__":
    main()

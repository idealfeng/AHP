"""根据矢量数据生成距离因子和密度因子，并进行 min-max 归一化。"""
from pathlib import Path
import csv
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree

BASE_DIR = Path(r"D:\Paper\毕设")

TEMPLATE = BASE_DIR / "data" / "factors" / "template_100m.tif"

ROADS = BASE_DIR / "data" / "processed" / "roads_utm.geojson"
POI = BASE_DIR / "data" / "processed" / "poi_utm.geojson"
SUBSTATION = BASE_DIR / "data" / "processed" / "substation_utm.geojson"

FACTOR_DIR = BASE_DIR / "data" / "factors"
TABLE_DIR = BASE_DIR / "results" / "tables"

FACTOR_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

ROAD_DISTANCE = FACTOR_DIR / "road_distance.tif"
ROAD_SCORE = FACTOR_DIR / "road_score.tif"

POI_COUNT = FACTOR_DIR / "poi_count_100m.tif"
POI_DENSITY = FACTOR_DIR / "poi_density.tif"
POI_SCORE = FACTOR_DIR / "poi_score.tif"

SUBSTATION_DISTANCE = FACTOR_DIR / "substation_distance.tif"
SUBSTATION_SCORE = FACTOR_DIR / "substation_score.tif"

STATS_CSV = TABLE_DIR / "vector_factor_stats.csv"

NODATA_FLOAT = -9999.0
NODATA_INT = 0

# POI 核密度带宽，单位：米
POI_BANDWIDTH_M = 1000


def load_template():
    with rasterio.open(TEMPLATE) as src:
        mask = src.read(1)
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height
        res_x = abs(transform.a)
        res_y = abs(transform.e)

    valid_mask = mask == 1

    return {
        "mask": valid_mask,
        "meta": meta,
        "transform": transform,
        "crs": crs,
        "width": width,
        "height": height,
        "res_x": res_x,
        "res_y": res_y,
    }


def save_float_raster(array, output_path, template_info):
    meta = template_info["meta"].copy()
    meta.update(
        {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "nodata": NODATA_FLOAT,
            "compress": "lzw",
        }
    )

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(array.astype("float32"), 1)

    print("已输出：", output_path)


def save_int_raster(array, output_path, template_info):
    meta = template_info["meta"].copy()
    meta.update(
        {
            "driver": "GTiff",
            "dtype": "uint16",
            "count": 1,
            "nodata": NODATA_INT,
            "compress": "lzw",
        }
    )

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(array.astype("uint16"), 1)

    print("已输出：", output_path)


def inverse_minmax_score(distance_array, valid_mask):
    """
    距离越小，得分越高。
    使用研究区内有效像元的 min-max 归一化。
    """
    score = np.full(distance_array.shape, NODATA_FLOAT, dtype="float32")

    valid = valid_mask & np.isfinite(distance_array) & (distance_array != NODATA_FLOAT)

    values = distance_array[valid]

    if values.size == 0:
        raise ValueError("没有有效距离值，无法归一化。")

    min_v = float(np.min(values))
    max_v = float(np.max(values))

    if max_v - min_v < 1e-9:
        score[valid] = 1.0
    else:
        score[valid] = 1 - (distance_array[valid] - min_v) / (max_v - min_v)
        score[valid] = np.clip(score[valid], 0, 1)

    return score, min_v, max_v


def positive_minmax_score(value_array, valid_mask):
    """
    数值越大，得分越高。
    适用于 POI 密度。
    """
    score = np.full(value_array.shape, NODATA_FLOAT, dtype="float32")

    valid = valid_mask & np.isfinite(value_array) & (value_array != NODATA_FLOAT)

    values = value_array[valid]

    if values.size == 0:
        raise ValueError("没有有效数值，无法归一化。")

    min_v = float(np.min(values))
    max_v = float(np.max(values))

    if max_v - min_v < 1e-9:
        score[valid] = 1.0
    else:
        score[valid] = (value_array[valid] - min_v) / (max_v - min_v)
        score[valid] = np.clip(score[valid], 0, 1)

    return score, min_v, max_v


def make_road_factor(template_info):
    print("\n==============================")
    print("生成道路距离因子")
    print("==============================")

    if not ROADS.exists():
        raise FileNotFoundError(f"找不到道路文件：{ROADS}")

    roads = gpd.read_file(ROADS).to_crs(template_info["crs"])
    roads = roads[roads.geometry.notna()]
    roads = roads[~roads.geometry.is_empty]

    print("道路数量：", len(roads))

    if roads.empty:
        raise ValueError("道路数据为空。")

    shapes = ((geom, 1) for geom in roads.geometry)

    road_binary = rasterize(
        shapes,
        out_shape=(template_info["height"], template_info["width"]),
        transform=template_info["transform"],
        fill=0,
        dtype="uint8",
        all_touched=True,
    )

    if road_binary.sum() == 0:
        raise ValueError("道路没有成功栅格化，请检查坐标系或模板范围。")

    # distance_transform_edt 会计算到最近 0/False 的距离
    # 所以这里 road_binary == 0 表示非道路区域，到道路的距离
    distance = distance_transform_edt(
        road_binary == 0, sampling=(template_info["res_y"], template_info["res_x"])
    ).astype("float32")

    distance[~template_info["mask"]] = NODATA_FLOAT

    score, min_d, max_d = inverse_minmax_score(distance, template_info["mask"])

    save_float_raster(distance, ROAD_DISTANCE, template_info)
    save_float_raster(score, ROAD_SCORE, template_info)

    print(f"道路距离范围：{min_d:.2f} m - {max_d:.2f} m")

    return {
        "factor": "road",
        "input_count": len(roads),
        "raw_min": min_d,
        "raw_max": max_d,
        "score_min": 0.0,
        "score_max": 1.0,
        "output": str(ROAD_SCORE),
    }


def make_poi_factor(template_info):
    print("\n==============================")
    print("生成 POI 密度因子")
    print("==============================")

    if not POI.exists():
        raise FileNotFoundError(f"找不到 POI 文件：{POI}")

    poi = gpd.read_file(POI).to_crs(template_info["crs"])
    poi = poi[poi.geometry.notna()]
    poi = poi[~poi.geometry.is_empty]

    # 如果存在非点几何，转为代表点
    poi = poi.copy()
    poi["geometry"] = poi.geometry.representative_point()

    print("POI 数量：", len(poi))

    if poi.empty:
        raise ValueError("POI 数据为空。")

    transform = template_info["transform"]
    width = template_info["width"]
    height = template_info["height"]

    counts = np.zeros((height, width), dtype="uint16")

    xs = poi.geometry.x.values
    ys = poi.geometry.y.values

    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    rows = np.array(rows)
    cols = np.array(cols)

    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)

    np.add.at(counts, (rows[inside], cols[inside]), 1)

    save_int_raster(counts, POI_COUNT, template_info)

    sigma_pixels = POI_BANDWIDTH_M / template_info["res_x"]

    density = gaussian_filter(counts.astype("float32"), sigma=sigma_pixels)

    density[~template_info["mask"]] = NODATA_FLOAT

    score, min_density, max_density = positive_minmax_score(
        density, template_info["mask"]
    )

    save_float_raster(density, POI_DENSITY, template_info)
    save_float_raster(score, POI_SCORE, template_info)

    print(f"POI 栅格内数量：{int(counts.sum())}")
    print(f"POI 密度范围：{min_density:.6f} - {max_density:.6f}")
    print(f"核密度带宽：{POI_BANDWIDTH_M} m")

    return {
        "factor": "poi",
        "input_count": len(poi),
        "raw_min": min_density,
        "raw_max": max_density,
        "score_min": 0.0,
        "score_max": 1.0,
        "output": str(POI_SCORE),
    }


def make_substation_factor(template_info):
    print("\n==============================")
    print("生成变电站距离因子")
    print("==============================")

    if not SUBSTATION.exists():
        raise FileNotFoundError(f"找不到变电站文件：{SUBSTATION}")

    substation = gpd.read_file(SUBSTATION).to_crs(template_info["crs"])
    substation = substation[substation.geometry.notna()]
    substation = substation[~substation.geometry.is_empty]

    substation = substation.copy()
    substation["geometry"] = substation.geometry.representative_point()

    print("变电站数量：", len(substation))

    if substation.empty:
        raise ValueError("变电站数据为空。")

    # 注意：这里不用 rasterize，而是 KDTree 直接计算每个像元中心到最近变电站的距离。
    # 这样可以保留研究区外缓冲区内的变电站对边界像元的影响。
    station_coords = np.column_stack(
        [substation.geometry.x.values, substation.geometry.y.values]
    )

    tree = cKDTree(station_coords)

    height = template_info["height"]
    width = template_info["width"]
    transform = template_info["transform"]
    mask = template_info["mask"]

    cols, rows = np.meshgrid(np.arange(width), np.arange(height))

    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e

    valid_coords = np.column_stack([xs[mask].ravel(), ys[mask].ravel()])

    distances, _ = tree.query(valid_coords, k=1)

    distance_array = np.full((height, width), NODATA_FLOAT, dtype="float32")
    distance_array[mask] = distances.astype("float32")

    score, min_d, max_d = inverse_minmax_score(distance_array, mask)

    save_float_raster(distance_array, SUBSTATION_DISTANCE, template_info)
    save_float_raster(score, SUBSTATION_SCORE, template_info)

    print(f"变电站距离范围：{min_d:.2f} m - {max_d:.2f} m")

    return {
        "factor": "substation",
        "input_count": len(substation),
        "raw_min": min_d,
        "raw_max": max_d,
        "score_min": 0.0,
        "score_max": 1.0,
        "output": str(SUBSTATION_SCORE),
    }


def save_stats(stats):
    with open(STATS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "factor",
                "input_count",
                "raw_min",
                "raw_max",
                "score_min",
                "score_max",
                "output",
            ],
        )
        writer.writeheader()
        writer.writerows(stats)

    print("\n已输出统计表：", STATS_CSV)


def main():
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"找不到模板：{TEMPLATE}")

    template_info = load_template()

    print("模板 CRS：", template_info["crs"])
    print("模板大小：", template_info["width"], template_info["height"])
    print("模板分辨率：", template_info["res_x"], template_info["res_y"])
    print("研究区有效像元数：", int(template_info["mask"].sum()))

    stats = []

    stats.append(make_road_factor(template_info))
    stats.append(make_poi_factor(template_info))
    stats.append(make_substation_factor(template_info))

    save_stats(stats)

    print("\n三个矢量因子生成完成。")
    print("关键输出：")
    print(" -", ROAD_SCORE)
    print(" -", POI_SCORE)
    print(" -", SUBSTATION_SCORE)


if __name__ == "__main__":
    main()

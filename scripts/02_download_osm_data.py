"""下载 OSM 数据的脚本，包含道路、POI、电力设施等"""
from pathlib import Path
import geopandas as gpd
import osmnx as ox

BASE_DIR = Path(r"D:\Paper\毕设")

BOUNDARY_WGS = BASE_DIR / "data" / "boundary" / "study_area_clean.geojson"
BOUNDARY_UTM = BASE_DIR / "data" / "boundary" / "study_area_utm.geojson"

RAW_OSM_DIR = BASE_DIR / "data" / "raw" / "osm"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_OSM_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.timeout = 300


def get_union_geometry(gdf):
    """兼容不同 GeoPandas 版本的合并几何写法"""
    try:
        return gdf.geometry.union_all()
    except Exception:
        return gdf.geometry.unary_union


def clean_for_geojson(gdf):
    """避免 list/dict 类型字段导致 GeoJSON 保存失败"""
    gdf = gdf.copy()
    for col in gdf.columns:
        if col == "geometry":
            continue
        gdf[col] = gdf[col].astype(str)
    return gdf


def get_buffered_polygon_wgs(buffer_m=1000):
    area_utm = gpd.read_file(BOUNDARY_UTM)

    area_buffer_utm = area_utm.copy()
    area_buffer_utm["geometry"] = area_buffer_utm.geometry.buffer(buffer_m)

    area_buffer_wgs = area_buffer_utm.to_crs(epsg=4326)
    return get_union_geometry(area_buffer_wgs)


def remove_empty_geometry(gdf):
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]
    return gdf


def download_roads(area_utm):
    print("\n=== 正在下载道路数据 ===")

    polygon = get_buffered_polygon_wgs(buffer_m=1000)

    G = ox.graph_from_polygon(
        polygon,
        network_type="drive",
        simplify=True,
        retain_all=False,
        truncate_by_edge=True,
    )

    nodes, edges = ox.graph_to_gdfs(G)

    keep_cols = [
        col
        for col in ["osmid", "name", "highway", "oneway", "length", "geometry"]
        if col in edges.columns
    ]

    roads = edges[keep_cols].copy()
    roads = remove_empty_geometry(roads)
    roads = clean_for_geojson(roads)

    raw_path = RAW_OSM_DIR / "roads_raw.geojson"
    processed_path = PROCESSED_DIR / "roads_utm.geojson"

    roads.to_file(raw_path, driver="GeoJSON", encoding="utf-8")

    roads_utm = roads.to_crs(epsg=32648)
    roads_utm = gpd.clip(roads_utm, area_utm)
    roads_utm = remove_empty_geometry(roads_utm)
    roads_utm = clean_for_geojson(roads_utm)

    roads_utm.to_file(processed_path, driver="GeoJSON", encoding="utf-8")

    print("已保存原始道路：", raw_path)
    print("已保存处理后道路：", processed_path)
    print("道路数量：", len(roads_utm))


def download_poi(area_utm):
    print("\n=== 正在下载 POI 数据 ===")

    area_wgs = gpd.read_file(BOUNDARY_WGS).to_crs(epsg=4326)
    polygon = get_union_geometry(area_wgs)

    tags = {
        "amenity": True,
        "shop": True,
        "tourism": True,
        "office": True,
        "leisure": True,
        "public_transport": True,
    }

    poi = ox.features_from_polygon(polygon, tags=tags)

    if poi is None or poi.empty:
        print("警告：没有获取到 POI")
        return

    poi = poi.reset_index(drop=True)

    useful_cols = [
        "name",
        "amenity",
        "shop",
        "tourism",
        "office",
        "leisure",
        "public_transport",
        "geometry",
    ]
    keep_cols = [c for c in useful_cols if c in poi.columns]
    poi = poi[keep_cols].copy()
    poi = remove_empty_geometry(poi)
    poi = clean_for_geojson(poi)

    raw_path = RAW_OSM_DIR / "poi_raw.geojson"
    processed_path = PROCESSED_DIR / "poi_utm.geojson"

    poi.to_file(raw_path, driver="GeoJSON", encoding="utf-8")

    # 转 UTM，并把点、线、面统一变成代表点
    poi_utm = poi.to_crs(epsg=32648)
    poi_point = poi_utm.copy()
    poi_point["geometry"] = poi_point.geometry.representative_point()

    poi_point = gpd.clip(poi_point, area_utm)
    poi_point = remove_empty_geometry(poi_point)
    poi_point = clean_for_geojson(poi_point)

    poi_point.to_file(processed_path, driver="GeoJSON", encoding="utf-8")

    print("已保存原始 POI：", raw_path)
    print("已保存处理后 POI：", processed_path)
    print("POI 数量：", len(poi_point))


def download_power(area_utm):
    print("\n=== 正在下载电力设施数据 ===")

    # 电力设施保留 5km 缓冲区，因为研究区外近处电力设施也会影响接入距离
    polygon = get_buffered_polygon_wgs(buffer_m=5000)

    tags = {"power": True}

    power = ox.features_from_polygon(polygon, tags=tags)

    if power is None or power.empty:
        print("警告：没有获取到 power 数据")
        return

    power = power.reset_index(drop=True)

    if "power" in power.columns:
        power = power[
            power["power"].isin(
                ["substation", "station", "transformer", "plant", "generator"]
            )
        ].copy()

    if power.empty:
        print("警告：筛选后没有变电站/电力设施")
        return

    useful_cols = ["name", "power", "operator", "geometry"]
    keep_cols = [c for c in useful_cols if c in power.columns]
    power = power[keep_cols].copy()
    power = remove_empty_geometry(power)
    power = clean_for_geojson(power)

    raw_path = RAW_OSM_DIR / "power_raw.geojson"
    processed_path = PROCESSED_DIR / "power_utm.geojson"

    power.to_file(raw_path, driver="GeoJSON", encoding="utf-8")

    power_utm = power.to_crs(epsg=32648)
    power_point = power_utm.copy()
    power_point["geometry"] = power_point.geometry.representative_point()

    # 这里不裁剪到研究区，保留研究区周边 5km 内电力设施
    power_point = remove_empty_geometry(power_point)
    power_point = clean_for_geojson(power_point)

    power_point.to_file(processed_path, driver="GeoJSON", encoding="utf-8")

    print("已保存原始电力设施：", raw_path)
    print("已保存处理后电力设施：", processed_path)
    print("电力设施数量：", len(power_point))


if __name__ == "__main__":
    area_utm = gpd.read_file(BOUNDARY_UTM)

    download_roads(area_utm)
    download_poi(area_utm)
    download_power(area_utm)

    print("\n全部 OSM 数据下载完成。")

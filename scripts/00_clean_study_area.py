import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from pathlib import Path

in_path = Path("data/boundary/study_area.geojson")
out_path = Path("data/boundary/study_area_clean.geojson")

gdf = gpd.read_file(in_path)

# 转成投影坐标，用平方米判断面积
gdf_utm = gdf.to_crs("EPSG:32648")


def remove_small_parts(geom, min_area_m2=1_000_000):
    """
    删除小于 min_area_m2 的碎片面。
    默认 1,000,000 m² = 1 km²。
    """
    if geom.geom_type == "Polygon":
        return geom

    if geom.geom_type == "MultiPolygon":
        large_parts = [part for part in geom.geoms if part.area >= min_area_m2]
        if len(large_parts) == 1:
            return large_parts[0]
        else:
            return MultiPolygon(large_parts)

    return geom


gdf_utm["geometry"] = gdf_utm["geometry"].apply(remove_small_parts)

# 再合并一次
gdf_utm = gdf_utm.dissolve()

# 转回 WGS84
gdf_clean = gdf_utm.to_crs("EPSG:4326")

gdf_clean.to_file(out_path, driver="GeoJSON", encoding="utf-8")

print("已输出清理后边界：", out_path)

# 面积检查
area_km2 = gdf_utm.geometry.area.sum() / 1_000_000
print(f"清理后面积：{area_km2:.2f} km²")

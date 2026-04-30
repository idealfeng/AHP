from pathlib import Path
import geopandas as gpd

BASE_DIR = Path(r"D:\Paper\毕设")

input_path = BASE_DIR / "data" / "boundary" / "study_area_clean.geojson"
output_path = BASE_DIR / "data" / "boundary" / "study_area_utm.geojson"

# 读取边界
gdf = gpd.read_file(input_path)

print("原始坐标系：", gdf.crs)
print("原始范围：", gdf.total_bounds)

# 确保是 WGS84
if gdf.crs is None:
    gdf = gdf.set_crs(epsg=4326)

gdf_wgs = gdf.to_crs(epsg=4326)

# 投影到 UTM 48N
gdf_utm = gdf_wgs.to_crs(epsg=32648)

# 修复可能的几何错误
gdf_utm["geometry"] = gdf_utm.geometry.buffer(0)

# 保存
gdf_utm.to_file(output_path, driver="GeoJSON", encoding="utf-8")

area_km2 = gdf_utm.geometry.area.sum() / 1_000_000

print("已输出：", output_path)
print(f"投影后面积：{area_km2:.2f} km²")
print("投影后范围：", gdf_utm.total_bounds)

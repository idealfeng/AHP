"""检查原始数据文件的脚本，确保数据完整且格式正确"""
from pathlib import Path
import geopandas as gpd

BASE_DIR = Path(r"D:\Paper\毕设")

files = {
    "研究区": BASE_DIR / "data" / "boundary" / "study_area_utm.geojson",
    "道路": BASE_DIR / "data" / "processed" / "roads_utm.geojson",
    "POI": BASE_DIR / "data" / "processed" / "poi_utm.geojson",
    "电力设施": BASE_DIR / "data" / "processed" / "power_utm.geojson",
}

for name, path in files.items():
    print(f"\n=== {name} ===")

    if not path.exists():
        print("文件不存在：", path)
        continue

    gdf = gpd.read_file(path)

    print("文件路径：", path)
    print("要素数量：", len(gdf))
    print("坐标系：", gdf.crs)
    print("字段：", list(gdf.columns))
    print("范围：", gdf.total_bounds)

    if name == "研究区":
        area_km2 = gdf.geometry.area.sum() / 1_000_000
        print(f"面积：{area_km2:.2f} km²")

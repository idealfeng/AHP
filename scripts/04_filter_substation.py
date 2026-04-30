"""从电力设施数据中筛选出变电站的脚本"""
from pathlib import Path
import geopandas as gpd

BASE_DIR = Path(r"D:\Paper\毕设")

input_path = BASE_DIR / "data" / "processed" / "power_utm.geojson"
output_path = BASE_DIR / "data" / "processed" / "substation_utm.geojson"

power = gpd.read_file(input_path)

print("原始电力设施数量：", len(power))
print("power 字段统计：")
print(power["power"].value_counts(dropna=False))

# 只保留变电站
substation = power[power["power"] == "substation"].copy()

# 去掉空几何
substation = substation[substation.geometry.notna()]
substation = substation[~substation.geometry.is_empty]

substation.to_file(output_path, driver="GeoJSON", encoding="utf-8")

print("\n已输出：", output_path)
print("变电站数量：", len(substation))
print("坐标系：", substation.crs)
print("范围：", substation.total_bounds)

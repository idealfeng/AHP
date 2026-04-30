import geopandas as gpd
from pathlib import Path

# 输出路径
out_dir = Path("data\\boundary")
out_dir.mkdir(parents=True, exist_ok=True)

# DataV 成都市区县边界
# 510100 是成都市 adcode，_full 表示包含下级区县边界
url = "https://geo.datav.aliyun.com/areas/bound/510100_full.json"

# 读取成都市区县 GeoJSON
chengdu = gpd.read_file(url)

print("字段：")
print(chengdu.columns)

print("区县名称：")
print(chengdu["name"].tolist())

# 你的五城区
target_names = ["锦江区", "青羊区", "金牛区", "武侯区", "成华区"]

study_area = chengdu[chengdu["name"].isin(target_names)].copy()

print("筛选结果：")
print(study_area[["name", "adcode"]])

# 保存五城区边界
study_area.to_file(
    out_dir / "study_area_5districts_raw.geojson",
    driver="GeoJSON",
    encoding="utf-8"
)

# 合并成一个总研究区
study_area_union = study_area.dissolve()

study_area_union.to_file(
    out_dir / "study_area.geojson",
    driver="GeoJSON",
    encoding="utf-8"
)

print("已生成：../data/boundary/study_area.geojson")

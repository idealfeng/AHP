from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt

# 你可以根据实际路径二选一
possible_paths = [
    Path("data/boundary/study_area.geojson"),
    Path("../data/boundary/study_area.geojson"),
]

study_path = None

for p in possible_paths:
    print(f"检查路径：{p.resolve()}  是否存在：{p.exists()}")
    if p.exists():
        study_path = p
        break

if study_path is None:
    raise FileNotFoundError("没有找到 study_area.geojson，请先确认文件保存位置。")

# 读取研究区
study_area = gpd.read_file(study_path)

print("\n=== 基本信息 ===")
print("文件路径：", study_path.resolve())
print("行数：", study_area.shape[0])
print("字段：", study_area.columns.tolist())
print("坐标系：", study_area.crs)
print("边界范围：", study_area.total_bounds)

# 如果没有 CRS，先指定为 WGS84
if study_area.crs is None:
    study_area = study_area.set_crs("EPSG:4326")

# 转成成都所在的 UTM 48N，用来算面积
study_area_utm = study_area.to_crs("EPSG:32648")
area_km2 = study_area_utm.geometry.area.sum() / 1_000_000

print("\n=== 面积检查 ===")
print(f"研究区面积约为：{area_km2:.2f} km²")

# 画图检查
out_dir = Path("results/maps")
out_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 8))
study_area.plot(ax=ax, edgecolor="black", facecolor="lightblue")
ax.set_title("Study Area Check")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.savefig(out_dir / "study_area_check.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n已输出检查图：results/maps/study_area_check.png")

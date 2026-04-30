from pathlib import Path
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

boundary_path = ROOT / "data" / "boundary" / "study_area_clean.geojson"
power_path = ROOT / "data" / "processed" / "power_utm.geojson"

out_dir = ROOT / "outputs" / "checks"
out_dir.mkdir(parents=True, exist_ok=True)

boundary = gpd.read_file(boundary_path)
power = gpd.read_file(power_path)

print("=== 边界信息 ===")
print("CRS:", boundary.crs)
print("范围:", boundary.total_bounds)
print("数量:", len(boundary))

print("\n=== 电力设施信息 ===")
print("CRS:", power.crs)
print("范围:", power.total_bounds)
print("数量:", len(power))
print("字段:", list(power.columns))
print("几何类型：")
print(power.geom_type.value_counts())

# 统一坐标系
if boundary.crs != power.crs:
    boundary = boundary.to_crs(power.crs)

# 输出字段统计
summary_rows = []

for col in power.columns:
    if col == "geometry":
        continue
    vc = power[col].value_counts(dropna=False).head(20)
    if len(vc) > 0:
        tmp = vc.reset_index()
        tmp.columns = ["value", "count"]
        tmp.insert(0, "field", col)
        summary_rows.append(tmp)

if summary_rows:
    summary = pd.concat(summary_rows, ignore_index=True)
    summary.to_csv(
        out_dir / "power_field_summary.csv", index=False, encoding="utf-8-sig"
    )
    print("\n已输出字段统计：", out_dir / "power_field_summary.csv")

# 绘图检查
fig, ax = plt.subplots(figsize=(8, 8))

boundary.boundary.plot(ax=ax, linewidth=1.2)

# 点、线、面分开画，避免报错
power_point = power[power.geom_type.isin(["Point", "MultiPoint"])]
power_line = power[power.geom_type.isin(["LineString", "MultiLineString"])]
power_poly = power[power.geom_type.isin(["Polygon", "MultiPolygon"])]

if len(power_poly) > 0:
    power_poly.plot(ax=ax, alpha=0.4)

if len(power_line) > 0:
    power_line.plot(ax=ax, linewidth=0.8)

if len(power_point) > 0:
    power_point.plot(ax=ax, markersize=8)

ax.set_title(f"Power Facilities in Study Area, n={len(power)}")
ax.set_axis_off()

plt.tight_layout()
plt.savefig(out_dir / "power_check_map.png", dpi=300)
plt.close()

print("已输出检查图：", out_dir / "power_check_map.png")
print("\n检查完成。")

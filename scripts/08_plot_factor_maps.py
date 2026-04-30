"""绘制五个因子的空间分布图，并计算每个因子的统计信息。"""
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

BASE_DIR = Path(r"D:\Paper\毕设")

FACTOR_DIR = BASE_DIR / "data" / "factors"
MAP_DIR = BASE_DIR / "results" / "maps"
TABLE_DIR = BASE_DIR / "results" / "tables"

MAP_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

factor_files = {
    "road_score": FACTOR_DIR / "road_score.tif",
    "poi_score": FACTOR_DIR / "poi_score.tif",
    "substation_score": FACTOR_DIR / "substation_score.tif",
    "landuse_score": FACTOR_DIR / "landuse_score.tif",
    "slope_score": FACTOR_DIR / "slope_score.tif",
}

summary_rows = []

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (name, path) in enumerate(factor_files.items()):
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata

    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata

    valid_values = arr[valid]

    if valid_values.size == 0:
        raise ValueError(f"{name} 没有有效像元。")

    summary_rows.append(
        {
            "factor": name,
            "min": float(valid_values.min()),
            "max": float(valid_values.max()),
            "mean": float(valid_values.mean()),
            "std": float(valid_values.std()),
            "count": int(valid_values.size),
        }
    )

    display_arr = np.where(valid, arr, np.nan)

    ax = axes[idx]
    im = ax.imshow(display_arr)
    ax.set_title(
        f"{name}\nmin={valid_values.min():.3f}, "
        f"max={valid_values.max():.3f}, "
        f"mean={valid_values.mean():.3f}"
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# 最后一个空白子图关掉
if len(factor_files) < len(axes):
    for j in range(len(factor_files), len(axes)):
        axes[j].axis("off")

plt.tight_layout()

out_map = MAP_DIR / "factor_check_5panel.png"
plt.savefig(out_map, dpi=300, bbox_inches="tight")
plt.close()

summary_df = pd.DataFrame(summary_rows)
out_csv = TABLE_DIR / "factor_summary_stats.csv"
summary_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

print("已输出五因子检查图：", out_map)
print("已输出因子统计表：", out_csv)
print("\n因子统计：")
print(summary_df)

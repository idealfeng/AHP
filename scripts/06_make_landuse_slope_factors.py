"""生成土地利用和坡度适宜性因子，输出与分析模板一致的栅格数据。"""
from pathlib import Path
import math
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling

BASE_DIR = Path(r"D:\Paper\毕设")

BOUNDARY_UTM = BASE_DIR / "data" / "boundary" / "study_area_utm.geojson"

DEM_CLIP = BASE_DIR / "data" / "processed" / "dem_clip_utm.tif"
LANDUSE_CLIP = BASE_DIR / "data" / "processed" / "landuse_clip_utm.tif"

FACTOR_DIR = BASE_DIR / "data" / "factors"
FACTOR_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE = FACTOR_DIR / "template_100m.tif"

DEM_100M = FACTOR_DIR / "dem_100m.tif"
LANDUSE_100M = FACTOR_DIR / "landuse_100m.tif"

SLOPE_NATIVE = FACTOR_DIR / "slope_native.tif"
SLOPE_100M = FACTOR_DIR / "slope_100m.tif"
SLOPE_SCORE = FACTOR_DIR / "slope_score.tif"

LANDUSE_SCORE = FACTOR_DIR / "landuse_score.tif"

TARGET_CRS = "EPSG:32648"
RESOLUTION = 100
NODATA_FLOAT = -9999.0


def create_template():
    print("\n=== 创建 100m 分析模板 ===")

    boundary = gpd.read_file(BOUNDARY_UTM).to_crs(TARGET_CRS)

    xmin, ymin, xmax, ymax = boundary.total_bounds

    xmin = math.floor(xmin / RESOLUTION) * RESOLUTION
    ymin = math.floor(ymin / RESOLUTION) * RESOLUTION
    xmax = math.ceil(xmax / RESOLUTION) * RESOLUTION
    ymax = math.ceil(ymax / RESOLUTION) * RESOLUTION

    width = int((xmax - xmin) / RESOLUTION)
    height = int((ymax - ymin) / RESOLUTION)

    transform = from_origin(xmin, ymax, RESOLUTION, RESOLUTION)

    mask_array = rasterize(
        [(geom, 1) for geom in boundary.geometry],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": TARGET_CRS,
        "transform": transform,
        "nodata": 0,
    }

    with rasterio.open(TEMPLATE, "w", **meta) as dst:
        dst.write(mask_array, 1)

    print("已输出模板：", TEMPLATE)
    print("模板宽高：", width, height)
    print("模板范围：", xmin, ymin, xmax, ymax)


def reproject_to_template(input_path, output_path, resampling_method, dtype, nodata):
    with rasterio.open(TEMPLATE) as tmpl:
        dst_meta = tmpl.meta.copy()
        dst_meta.update({"dtype": dtype, "nodata": nodata})

        dst_array = np.full((tmpl.height, tmpl.width), nodata, dtype=dtype)

        with rasterio.open(input_path) as src:
            src_nodata = src.nodata

            reproject(
                source=rasterio.band(src, 1),
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=tmpl.transform,
                dst_crs=tmpl.crs,
                dst_nodata=nodata,
                resampling=resampling_method,
            )

        with rasterio.open(output_path, "w", **dst_meta) as dst:
            dst.write(dst_array, 1)

    print("已重采样到模板：", output_path)


def calculate_slope_native():
    print("\n=== 计算坡度 ===")

    with rasterio.open(DEM_CLIP) as src:
        dem = src.read(1).astype("float32")
        meta = src.meta.copy()

        nodata = src.nodata
        if nodata is not None:
            dem[dem == nodata] = np.nan

        xres = abs(src.transform.a)
        yres = abs(src.transform.e)

        dz_dy, dz_dx = np.gradient(dem, yres, xres)

        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.degrees(slope_rad).astype("float32")

        slope_deg[np.isnan(slope_deg)] = NODATA_FLOAT

        meta.update({"dtype": "float32", "nodata": NODATA_FLOAT})

        with rasterio.open(SLOPE_NATIVE, "w", **meta) as dst:
            dst.write(slope_deg, 1)

    print("已输出原始坡度：", SLOPE_NATIVE)


def make_slope_score():
    print("\n=== 生成坡度适宜性因子 ===")

    calculate_slope_native()

    reproject_to_template(
        SLOPE_NATIVE,
        SLOPE_100M,
        Resampling.bilinear,
        dtype="float32",
        nodata=NODATA_FLOAT,
    )

    with rasterio.open(TEMPLATE) as tmpl:
        mask = tmpl.read(1)

    with rasterio.open(SLOPE_100M) as src:
        slope = src.read(1).astype("float32")
        meta = src.meta.copy()

    valid = (mask == 1) & (slope != NODATA_FLOAT) & np.isfinite(slope)

    score = np.full(slope.shape, NODATA_FLOAT, dtype="float32")

    # 坡度越小越适合。
    # 这里设置 10 度以上基本视为不适宜，适合城市平原区实验。
    score[valid] = 1 - np.clip(slope[valid] / 10.0, 0, 1)

    meta.update({"dtype": "float32", "nodata": NODATA_FLOAT})

    with rasterio.open(SLOPE_SCORE, "w", **meta) as dst:
        dst.write(score, 1)

    print("已输出坡度得分：", SLOPE_SCORE)

    if valid.any():
        print("坡度最小值：", float(np.nanmin(slope[valid])))
        print("坡度最大值：", float(np.nanmax(slope[valid])))
        print("坡度得分最小值：", float(np.nanmin(score[valid])))
        print("坡度得分最大值：", float(np.nanmax(score[valid])))


def make_landuse_score():
    print("\n=== 生成土地利用适宜性因子 ===")

    reproject_to_template(
        LANDUSE_CLIP, LANDUSE_100M, Resampling.nearest, dtype="uint8", nodata=0
    )

    with rasterio.open(TEMPLATE) as tmpl:
        mask = tmpl.read(1)

    with rasterio.open(LANDUSE_100M) as src:
        landuse = src.read(1)
        meta = src.meta.copy()

    # ESA WorldCover 分类赋分
    # 10 Tree cover
    # 20 Shrubland
    # 30 Grassland
    # 40 Cropland
    # 50 Built-up
    # 60 Bare / sparse vegetation
    # 70 Snow and ice
    # 80 Permanent water bodies
    # 90 Herbaceous wetland
    # 95 Mangroves
    # 100 Moss and lichen
    score_map = {
        10: 0.2,
        20: 0.2,
        30: 0.4,
        40: 0.3,
        50: 1.0,
        60: 0.6,
        70: 0.0,
        80: 0.0,
        90: 0.0,
        95: 0.0,
        100: 0.1,
    }

    score = np.full(landuse.shape, NODATA_FLOAT, dtype="float32")

    valid_area = mask == 1

    # 先给研究区内未知类别一个保守分数
    score[valid_area] = 0.3

    for code, value in score_map.items():
        score[(landuse == code) & valid_area] = value

    meta.update({"dtype": "float32", "nodata": NODATA_FLOAT})

    with rasterio.open(LANDUSE_SCORE, "w", **meta) as dst:
        dst.write(score, 1)

    print("已输出土地利用得分：", LANDUSE_SCORE)

    unique, counts = np.unique(landuse[valid_area], return_counts=True)
    print("\n研究区内土地利用类别统计：")
    for u, c in zip(unique, counts):
        print(f"类别 {int(u)}: {int(c)} 个像元")


def main():
    if not DEM_CLIP.exists():
        raise FileNotFoundError(f"找不到 DEM：{DEM_CLIP}")

    if not LANDUSE_CLIP.exists():
        raise FileNotFoundError(f"找不到土地利用：{LANDUSE_CLIP}")

    create_template()

    reproject_to_template(
        DEM_CLIP, DEM_100M, Resampling.bilinear, dtype="float32", nodata=NODATA_FLOAT
    )

    make_slope_score()
    make_landuse_score()

    print("\n土地利用因子和坡度因子生成完成。")


if __name__ == "__main__":
    main()

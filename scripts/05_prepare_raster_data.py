"""处理原始 DEM 和土地利用数据，重投影到 UTM 坐标系，并裁剪到研究区范围内。"""

from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling

BASE_DIR = Path(r"D:\Paper\毕设")

BOUNDARY_UTM = BASE_DIR / "data" / "boundary" / "study_area_utm.geojson"

DEM_RAW = BASE_DIR / "data" / "raw" / "dem" / "dem_raw.tif"
LANDUSE_RAW = BASE_DIR / "data" / "raw" / "landuse" / "landuse_raw.tif"

DEM_TEMP = BASE_DIR / "data" / "processed" / "temp_dem_utm.tif"
LANDUSE_TEMP = BASE_DIR / "data" / "processed" / "temp_landuse_utm.tif"

DEM_OUT = BASE_DIR / "data" / "processed" / "dem_clip_utm.tif"
LANDUSE_OUT = BASE_DIR / "data" / "processed" / "landuse_clip_utm.tif"

TARGET_CRS = "EPSG:32648"


def reproject_to_utm(input_path, output_path, resampling_method, nodata_value):
    with rasterio.open(input_path) as src:
        print("\n原始文件：", input_path)
        print("原始 CRS：", src.crs)
        print("原始范围：", src.bounds)
        print("原始分辨率：", src.res)
        print("原始宽高：", src.width, src.height)
        print("原始 NoData：", src.nodata)

        src_nodata = src.nodata if src.nodata is not None else nodata_value

        transform, width, height = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": TARGET_CRS,
                "transform": transform,
                "width": width,
                "height": height,
                "nodata": nodata_value,
            }
        )

        with rasterio.open(output_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src_nodata,
                    dst_transform=transform,
                    dst_crs=TARGET_CRS,
                    dst_nodata=nodata_value,
                    resampling=resampling_method,
                )

    print("已重投影：", output_path)


def clip_by_boundary(input_path, output_path, nodata_value):
    boundary = gpd.read_file(BOUNDARY_UTM).to_crs(TARGET_CRS)

    with rasterio.open(input_path) as src:
        geoms = [geom for geom in boundary.geometry]

        out_image, out_transform = mask(src, geoms, crop=True, nodata=nodata_value)

        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "crs": TARGET_CRS,
                "nodata": nodata_value,
            }
        )

        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(out_image)

    print("已裁剪：", output_path)

    with rasterio.open(output_path) as src:
        print("输出 CRS：", src.crs)
        print("输出范围：", src.bounds)
        print("输出分辨率：", src.res)
        print("输出宽高：", src.width, src.height)
        print("输出 NoData：", src.nodata)


def process_dem():
    print("\n==============================")
    print("处理 DEM")
    print("==============================")

    if not DEM_RAW.exists():
        raise FileNotFoundError(f"找不到 DEM 文件：{DEM_RAW}")

    reproject_to_utm(DEM_RAW, DEM_TEMP, Resampling.bilinear, nodata_value=-9999)

    clip_by_boundary(DEM_TEMP, DEM_OUT, nodata_value=-9999)

    if DEM_TEMP.exists():
        DEM_TEMP.unlink()


def process_landuse():
    print("\n==============================")
    print("处理土地利用")
    print("==============================")

    if not LANDUSE_RAW.exists():
        raise FileNotFoundError(f"找不到土地利用文件：{LANDUSE_RAW}")

    reproject_to_utm(LANDUSE_RAW, LANDUSE_TEMP, Resampling.nearest, nodata_value=0)

    clip_by_boundary(LANDUSE_TEMP, LANDUSE_OUT, nodata_value=0)

    if LANDUSE_TEMP.exists():
        LANDUSE_TEMP.unlink()


if __name__ == "__main__":
    process_dem()
    process_landuse()

    print("\nDEM 和土地利用预处理完成。")

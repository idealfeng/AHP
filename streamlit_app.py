"""stramlit_app.py"""
from pathlib import Path
from fractions import Fraction
import os
import re
import json
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from rasterio.mask import mask
from shapely.geometry import shape, mapping

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

import streamlit as st
from openai import OpenAI

# ==========================================================
# 基本路径
# ==========================================================
BASE_DIR = Path(r"D:\Paper\毕设")

FACTOR_DIR = BASE_DIR / "data" / "factors"
BOUNDARY = BASE_DIR / "data" / "boundary" / "study_area_utm.geojson"

RUN_ROOT = BASE_DIR / "results" / "streamlit_runs"
RUN_ROOT.mkdir(parents=True, exist_ok=True)

FACTOR_PATHS = {
    "road": FACTOR_DIR / "road_score.tif",
    "poi": FACTOR_DIR / "poi_score.tif",
    "substation": FACTOR_DIR / "substation_score.tif",
    "landuse": FACTOR_DIR / "landuse_score.tif",
    "slope": FACTOR_DIR / "slope_score.tif",
}

FACTORS = ["poi", "road", "substation", "landuse", "slope"]

FACTOR_CN = {
    "poi": "POI密度 / 充电需求强度",
    "road": "道路距离 / 交通可达性",
    "substation": "变电站距离 / 电网接入条件",
    "landuse": "土地利用适宜性",
    "slope": "坡度 / 地形条件",
}

RI_TABLE = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}

NODATA_FLOAT = -9999.0


# ==========================================================
# 页面设置
# ==========================================================
st.set_page_config(page_title="成都市充电桩选址智能体", page_icon="⚡", layout="wide")

plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# ==========================================================
# 工具函数：LLM-AHP
# ==========================================================
def parse_number(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        x = x.strip()
        if "/" in x:
            return float(Fraction(x))
        return float(x)
    raise ValueError(f"无法解析矩阵元素：{x}")


def extract_json(text):
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("模型输出中没有找到 JSON 对象。")

    return json.loads(match.group(0))


def build_prompt(user_preference):
    factor_desc = "\n".join([f"- {f}: {FACTOR_CN[f]}" for f in FACTORS])

    return f"""
你需要根据用户的自然语言偏好，构建 AHP 两两比较判断矩阵。

因子顺序固定为：
{FACTORS}

因子含义：
{factor_desc}

Saaty 1-9 标度含义：
1 表示同等重要；
2 表示稍微重要；
3 表示略重要；
5 表示明显重要；
7 表示强烈重要；
9 表示极端重要；
4、6、8 为中间值。

规则：
1. matrix 必须是 5×5 矩阵。
2. 对角线必须全为 1。
3. 必须满足互反性：a_ij = 1 / a_ji。
4. 使用数字小数，不要使用 "1/2" 这类字符串。
5. 尽量保证 AHP 一致性检验 CR < 0.1。
6. 只输出 JSON，不要输出解释文字。

用户偏好：
{user_preference}

输出格式：
{{
  "factors": ["poi", "road", "substation", "landuse", "slope"],
  "matrix": [
    [1, 2, 3, 4, 5],
    [0.5, 1, 2, 3, 4],
    [0.333333, 0.5, 1, 2, 3],
    [0.25, 0.333333, 0.5, 1, 2],
    [0.2, 0.25, 0.333333, 0.5, 1]
  ],
  "reason": "一句话说明偏好映射逻辑"
}}
"""


def call_deepseek(user_preference, api_key, model):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是严谨的 GIS-MCDA 与 AHP 专家，只输出合法 JSON。",
            },
            {"role": "user", "content": build_prompt(user_preference)},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        stream=False,
    )

    content = response.choices[0].message.content
    return content, extract_json(content)


def matrix_from_result(result):
    if "matrix" not in result:
        raise ValueError("模型 JSON 中没有 matrix 字段。")

    raw = result["matrix"]

    A = np.array([[parse_number(x) for x in row] for row in raw], dtype=float)

    if A.shape != (len(FACTORS), len(FACTORS)):
        raise ValueError(f"矩阵形状错误：{A.shape}")

    if np.any(A <= 0):
        raise ValueError("判断矩阵元素必须为正数。")

    # 自动修正轻微互反误差
    n = A.shape[0]
    for i in range(n):
        A[i, i] = 1.0
        for j in range(i + 1, n):
            A[j, i] = 1.0 / A[i, j]

    return A


def calculate_ahp(A):
    eigvals, eigvecs = np.linalg.eig(A)

    max_idx = np.argmax(eigvals.real)
    lambda_max = float(eigvals[max_idx].real)

    w = eigvecs[:, max_idx].real
    w = np.abs(w)
    w = w / w.sum()

    n = A.shape[0]
    CI = (lambda_max - n) / (n - 1)
    RI = RI_TABLE[n]
    CR = CI / RI if RI != 0 else 0.0

    return w, lambda_max, CI, RI, CR


# ==========================================================
# 工具函数：栅格叠加
# ==========================================================
def load_factor_arrays():
    arrays = {}
    meta = None
    valid_mask = None

    for name, path in FACTOR_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"找不到因子文件：{path}")

        with rasterio.open(path) as src:
            arr = src.read(1).astype("float32")
            nodata = src.nodata

            if meta is None:
                meta = src.meta.copy()
            else:
                if (
                    src.width != meta["width"]
                    or src.height != meta["height"]
                    or src.transform != meta["transform"]
                    or src.crs != meta["crs"]
                ):
                    raise ValueError(f"{name} 与其他因子栅格不对齐。")

        valid = np.isfinite(arr)
        if nodata is not None:
            valid &= arr != nodata

        arrays[name] = arr

        if valid_mask is None:
            valid_mask = valid
        else:
            valid_mask &= valid

    return arrays, meta, valid_mask


def weighted_overlay(weights, run_dir):
    arrays, meta, valid_mask = load_factor_arrays()

    score = np.full(next(iter(arrays.values())).shape, NODATA_FLOAT, dtype="float32")
    weighted_sum = np.zeros(score.shape, dtype="float32")

    for f, w in weights.items():
        weighted_sum += arrays[f] * w

    score[valid_mask] = weighted_sum[valid_mask]

    valid_values = score[valid_mask]

    q20, q40, q60, q80 = np.quantile(valid_values, [0.2, 0.4, 0.6, 0.8])

    classes = np.zeros(score.shape, dtype="uint8")
    classes[valid_mask & (score <= q20)] = 1
    classes[valid_mask & (score > q20) & (score <= q40)] = 2
    classes[valid_mask & (score > q40) & (score <= q60)] = 3
    classes[valid_mask & (score > q60) & (score <= q80)] = 4
    classes[valid_mask & (score > q80)] = 5

    score_path = run_dir / "suitability_score.tif"
    class_path = run_dir / "suitability_class.tif"

    score_meta = meta.copy()
    score_meta.update({"dtype": "float32", "nodata": NODATA_FLOAT, "compress": "lzw"})

    with rasterio.open(score_path, "w", **score_meta) as dst:
        dst.write(score.astype("float32"), 1)

    class_meta = meta.copy()
    class_meta.update({"dtype": "uint8", "nodata": 0, "compress": "lzw"})

    with rasterio.open(class_path, "w", **class_meta) as dst:
        dst.write(classes.astype("uint8"), 1)

    pixel_area_km2 = abs(meta["transform"].a * meta["transform"].e) / 1_000_000

    area_rows = []
    label_map = {
        1: "不适宜区",
        2: "较低适宜区",
        3: "中等适宜区",
        4: "较高适宜区",
        5: "高适宜区",
    }

    total_valid = int(valid_mask.sum())

    for cls in [1, 2, 3, 4, 5]:
        count = int((classes == cls).sum())
        area_rows.append(
            {
                "class": cls,
                "label": label_map[cls],
                "pixel_count": count,
                "area_km2": count * pixel_area_km2,
                "ratio": count / total_valid if total_valid > 0 else 0,
            }
        )

    area_df = pd.DataFrame(area_rows)
    area_df.to_csv(
        run_dir / "suitability_area_stats.csv", index=False, encoding="utf-8-sig"
    )

    stats = {
        "score_min": float(valid_values.min()),
        "score_max": float(valid_values.max()),
        "score_mean": float(valid_values.mean()),
        "score_std": float(valid_values.std()),
        "q20": float(q20),
        "q40": float(q40),
        "q60": float(q60),
        "q80": float(q80),
        "valid_pixels": total_valid,
        "score_path": str(score_path),
        "class_path": str(class_path),
    }

    return score, classes, meta, valid_mask, stats, area_df, score_path, class_path


# ==========================================================
# 工具函数：候选区提取
# ==========================================================
def extract_candidates(score_path, class_path, run_dir, min_area_m2=30000):
    with rasterio.open(class_path) as src:
        class_arr = src.read(1)
        transform = src.transform
        crs = src.crs

    high_mask = class_arr == 5

    results = shapes(high_mask.astype("uint8"), mask=high_mask, transform=transform)

    polygons = []
    for geom, value in results:
        if int(value) == 1:
            polygons.append(shape(geom))

    if not polygons:
        return gpd.GeoDataFrame(), gpd.GeoDataFrame(), pd.DataFrame()

    areas = gpd.GeoDataFrame({"class": [5] * len(polygons)}, geometry=polygons, crs=crs)

    areas["geometry"] = areas.geometry.buffer(0)
    areas["area_m2"] = areas.geometry.area
    areas = areas[areas["area_m2"] >= min_area_m2].copy()

    if areas.empty:
        return areas, gpd.GeoDataFrame(), pd.DataFrame()

    areas = areas.sort_values("area_m2", ascending=False).reset_index(drop=True)
    areas["candidate_id"] = range(1, len(areas) + 1)
    areas["area_km2"] = areas["area_m2"] / 1_000_000

    mean_scores = []
    max_scores = []
    min_scores = []

    with rasterio.open(score_path) as src:
        nodata = src.nodata

        for geom in areas.geometry:
            out_image, _ = mask(src, [mapping(geom)], crop=True, nodata=nodata)

            arr = out_image[0].astype("float32")
            valid = np.isfinite(arr)
            if nodata is not None:
                valid &= arr != nodata

            values = arr[valid]

            if values.size == 0:
                mean_scores.append(np.nan)
                max_scores.append(np.nan)
                min_scores.append(np.nan)
            else:
                mean_scores.append(float(values.mean()))
                max_scores.append(float(values.max()))
                min_scores.append(float(values.min()))

    areas["mean_score"] = mean_scores
    areas["max_score"] = max_scores
    areas["min_score"] = min_scores

    points = areas.copy()
    points["geometry"] = points.geometry.representative_point()

    points_wgs = points.to_crs(epsg=4326)

    points["center_x"] = points.geometry.x
    points["center_y"] = points.geometry.y
    points["lon"] = points_wgs.geometry.x.values
    points["lat"] = points_wgs.geometry.y.values

    stats_cols = [
        "candidate_id",
        "area_m2",
        "area_km2",
        "mean_score",
        "max_score",
        "min_score",
        "center_x",
        "center_y",
        "lon",
        "lat",
    ]

    stats_df = (
        points[stats_cols]
        .sort_values(["mean_score", "area_km2"], ascending=[False, False])
        .reset_index(drop=True)
    )

    areas.to_file(
        run_dir / "high_suitability_areas.geojson", driver="GeoJSON", encoding="utf-8"
    )
    areas.to_crs(epsg=4326).to_file(
        run_dir / "high_suitability_areas_wgs84.geojson",
        driver="GeoJSON",
        encoding="utf-8",
    )

    points.to_file(
        run_dir / "candidate_points.geojson", driver="GeoJSON", encoding="utf-8"
    )
    points.to_crs(epsg=4326).to_file(
        run_dir / "candidate_points_wgs84.geojson", driver="GeoJSON", encoding="utf-8"
    )

    stats_df.to_csv(
        run_dir / "candidate_sites_stats.csv", index=False, encoding="utf-8-sig"
    )

    return areas, points, stats_df


# ==========================================================
# 工具函数：绘图
# ==========================================================
def get_extent(bounds):
    return [bounds.left, bounds.right, bounds.bottom, bounds.top]


def plot_score_map(score_path, out_path):
    with rasterio.open(score_path) as src:
        arr = src.read(1)
        nodata = src.nodata
        bounds = src.bounds
        crs = src.crs

    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata

    display = np.where(valid, arr, np.nan)

    boundary = gpd.read_file(BOUNDARY).to_crs(crs)

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(display, extent=get_extent(bounds), origin="upper", cmap="YlOrRd")

    boundary.boundary.plot(ax=ax, linewidth=1.0, edgecolor="black")

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Suitability Score")

    ax.set_title("综合适宜性得分图")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_class_map(class_path, out_path):
    with rasterio.open(class_path) as src:
        arr = src.read(1)
        nodata = src.nodata
        bounds = src.bounds
        crs = src.crs

    valid = arr > 0
    if nodata is not None:
        valid &= arr != nodata

    display = np.where(valid, arr, np.nan)

    boundary = gpd.read_file(BOUNDARY).to_crs(crs)

    colors = [
        "#d73027",
        "#fc8d59",
        "#fee08b",
        "#d9ef8b",
        "#1a9850",
    ]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.imshow(display, extent=get_extent(bounds), origin="upper", cmap=cmap, norm=norm)

    boundary.boundary.plot(ax=ax, linewidth=1.0, edgecolor="black")

    legend_items = [
        Patch(facecolor=colors[0], label="1 不适宜区"),
        Patch(facecolor=colors[1], label="2 较低适宜区"),
        Patch(facecolor=colors[2], label="3 中等适宜区"),
        Patch(facecolor=colors[3], label="4 较高适宜区"),
        Patch(facecolor=colors[4], label="5 高适宜区"),
    ]

    ax.legend(handles=legend_items, loc="lower left", frameon=True, fontsize=8)

    ax.set_title("适宜性分级图")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_candidate_map(class_path, areas, points, out_path):
    with rasterio.open(class_path) as src:
        arr = src.read(1)
        nodata = src.nodata
        bounds = src.bounds
        crs = src.crs

    valid = arr > 0
    if nodata is not None:
        valid &= arr != nodata

    display = np.where(valid, arr, np.nan)

    boundary = gpd.read_file(BOUNDARY).to_crs(crs)

    colors = [
        "#d73027",
        "#fc8d59",
        "#fee08b",
        "#d9ef8b",
        "#1a9850",
    ]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.imshow(
        display,
        extent=get_extent(bounds),
        origin="upper",
        cmap=cmap,
        norm=norm,
        alpha=0.9,
    )

    boundary.boundary.plot(ax=ax, linewidth=1.0, edgecolor="black")

    if areas is not None and not areas.empty:
        areas.to_crs(crs).boundary.plot(ax=ax, linewidth=0.8, edgecolor="blue")

    if points is not None and not points.empty:
        top_points = points.sort_values(
            ["mean_score", "area_km2"], ascending=[False, False]
        ).head(20)
        top_points.to_crs(crs).plot(ax=ax, markersize=18, color="blue")

    legend_items = [
        Patch(facecolor=colors[0], label="1 不适宜区"),
        Patch(facecolor=colors[1], label="2 较低适宜区"),
        Patch(facecolor=colors[2], label="3 中等适宜区"),
        Patch(facecolor=colors[3], label="4 较高适宜区"),
        Patch(facecolor=colors[4], label="5 高适宜区"),
        Patch(facecolor="blue", label="高适宜区边界 / 前20候选点"),
    ]

    ax.legend(handles=legend_items, loc="lower left", frameon=True, fontsize=8)

    ax.set_title("高适宜区与候选点")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ==========================================================
# Streamlit UI
# ==========================================================
st.title("⚡ 基于 LLM-AHP 的成都市充电桩选址智能体原型")

st.markdown("""
这个原型用于演示：**自然语言偏好 → LLM 生成 AHP 权重 → GIS 加权叠加 → 高适宜区提取**。
""")

with st.sidebar:
    st.header("运行设置")

    api_key_input = st.text_input(
        "DeepSeek API Key",
        value=os.getenv("DEEPSEEK_API_KEY", ""),
        type="password",
        help="可以留空，但需要提前设置环境变量 DEEPSEEK_API_KEY。",
    )

    model = st.text_input(
        "DeepSeek 模型名", value=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    )

    min_area_m2 = st.number_input(
        "最小候选区面积 m²", min_value=10000, max_value=1000000, value=30000, step=10000
    )

    st.caption("默认 30000 m²，约等于 3 个 100m 像元。")

default_preference = "我认为充电需求密度最重要，道路交通可达性次之，电网接入条件也很重要，土地利用适宜性有一定影响，坡度影响最小。"

user_preference = st.text_area(
    "请输入自然语言选址偏好", value=default_preference, height=120
)

run_name = st.text_input("本次运行名称", value="demo_llm_ahp")

start = st.button("🚀 开始运行选址智能体", type="primary")

if start:
    api_key = api_key_input.strip()

    if not api_key:
        st.error("请在侧边栏输入 DeepSeek API Key，或先设置环境变量 DEEPSEEK_API_KEY。")
        st.stop()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_run_name = re.sub(r"[^a-zA-Z0-9_\u4e00-\u9fa5-]", "_", run_name)
    run_dir = RUN_ROOT / f"{timestamp}_{safe_run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        with st.status("正在调用 LLM 生成 AHP 判断矩阵...", expanded=True) as status:
            raw_text, result = call_deepseek(user_preference, api_key, model)

            with open(run_dir / "llm_raw_response.txt", "w", encoding="utf-8") as f:
                f.write(raw_text)

            A = matrix_from_result(result)
            weights_arr, lambda_max, CI, RI, CR = calculate_ahp(A)

            weights = {f: float(w) for f, w in zip(FACTORS, weights_arr)}

            matrix_df = pd.DataFrame(A, index=FACTORS, columns=FACTORS)
            weight_df = pd.DataFrame(
                {
                    "factor": FACTORS,
                    "factor_cn": [FACTOR_CN[f] for f in FACTORS],
                    "weight": weights_arr,
                }
            )

            matrix_df.to_csv(run_dir / "ahp_matrix.csv", encoding="utf-8-sig")
            weight_df.to_csv(
                run_dir / "ahp_weights.csv", index=False, encoding="utf-8-sig"
            )

            consistency_df = pd.DataFrame(
                [
                    {
                        "lambda_max": lambda_max,
                        "CI": CI,
                        "RI": RI,
                        "CR": CR,
                        "passed": CR < 0.1,
                    }
                ]
            )
            consistency_df.to_csv(
                run_dir / "ahp_consistency.csv", index=False, encoding="utf-8-sig"
            )

            st.write("AHP 权重已生成。")
            st.write(f"CR = {CR:.6f}")

            if CR >= 0.1:
                st.warning(
                    "CR >= 0.1，一致性未通过。当前 Demo 将停止，你可以修改偏好或增加自动修正机制。"
                )
                status.update(label="一致性检验未通过", state="error")
                st.stop()

            status.update(label="AHP 权重生成完成", state="complete")

        with st.status("正在执行 GIS 加权叠加...", expanded=True) as status:
            (
                score,
                classes,
                meta,
                valid_mask,
                overlay_stats,
                area_df,
                score_path,
                class_path,
            ) = weighted_overlay(weights=weights, run_dir=run_dir)
            st.write("综合适宜性得分与分级完成。")
            status.update(label="GIS 加权叠加完成", state="complete")

        with st.status("正在提取高适宜区候选区域...", expanded=True) as status:
            areas, points, candidate_stats = extract_candidates(
                score_path=score_path,
                class_path=class_path,
                run_dir=run_dir,
                min_area_m2=min_area_m2,
            )
            st.write("高适宜区候选区域提取完成。")
            status.update(label="候选区提取完成", state="complete")

        with st.status("正在生成地图...", expanded=True) as status:
            score_map = run_dir / "suitability_score_map.png"
            class_map = run_dir / "suitability_class_map.png"
            candidate_map = run_dir / "candidate_sites_map.png"

            plot_score_map(score_path, score_map)
            plot_class_map(class_path, class_map)
            plot_candidate_map(class_path, areas, points, candidate_map)

            status.update(label="地图生成完成", state="complete")

        # 保存运行摘要
        run_summary = {
            "run_name": run_name,
            "timestamp": timestamp,
            "user_preference": user_preference,
            "model": model,
            "weights": weights,
            "lambda_max": lambda_max,
            "CI": CI,
            "RI": RI,
            "CR": CR,
            "passed": CR < 0.1,
            "llm_reason": result.get("reason", ""),
            "overlay_stats": overlay_stats,
            "candidate_count": (
                int(len(candidate_stats)) if candidate_stats is not None else 0
            ),
            "candidate_total_area_km2": (
                float(candidate_stats["area_km2"].sum())
                if candidate_stats is not None and not candidate_stats.empty
                else 0.0
            ),
            "run_dir": str(run_dir),
        }

        with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(run_summary, f, ensure_ascii=False, indent=2)

        # 页面展示
        st.success("选址智能体运行完成！")

        st.subheader("1. 自然语言偏好")
        st.write(user_preference)

        st.subheader("2. LLM-AHP 权重结果")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.dataframe(weight_df, use_container_width=True)

        with col2:
            st.metric("CR 一致性比率", f"{CR:.6f}")
            st.metric("综合适宜性均值", f"{overlay_stats['score_mean']:.4f}")
            if candidate_stats is not None and not candidate_stats.empty:
                st.metric("候选区数量", len(candidate_stats))
                st.metric(
                    "候选区总面积 km²", f"{candidate_stats['area_km2'].sum():.2f}"
                )
            else:
                st.metric("候选区数量", 0)

        st.subheader("3. AHP 判断矩阵")
        st.dataframe(matrix_df, use_container_width=True)

        st.subheader("4. 综合适宜性结果图")

        img_col1, img_col2 = st.columns(2)

        with img_col1:
            st.image(
                str(score_map), caption="综合适宜性得分图", use_container_width=True
            )

        with img_col2:
            st.image(str(class_map), caption="适宜性分级图", use_container_width=True)

        st.subheader("5. 高适宜区与候选点")
        st.image(
            str(candidate_map),
            caption="高适宜区与候选点分布图",
            use_container_width=True,
        )

        st.subheader("6. 面积统计")
        st.dataframe(area_df, use_container_width=True)

        st.subheader("7. 候选区统计")

        if candidate_stats is not None and not candidate_stats.empty:
            st.dataframe(candidate_stats.head(20), use_container_width=True)
        else:
            st.warning("当前面积阈值下没有提取到候选区，可以降低最小候选区面积。")

        st.subheader("8. 输出目录")
        st.code(str(run_dir))

    except Exception as e:
        st.error(f"运行失败：{e}")
        st.exception(e)

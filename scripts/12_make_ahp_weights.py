"""AHP 权重计算脚本"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE_DIR = Path(r"D:\Paper\毕设")

WEIGHT_DIR = BASE_DIR / "results" / "weights"
WEIGHT_DIR.mkdir(parents=True, exist_ok=True)

MATRIX_CSV = WEIGHT_DIR / "manual_ahp_matrix.csv"
WEIGHTS_CSV = WEIGHT_DIR / "manual_ahp_weights.csv"
CONSISTENCY_CSV = WEIGHT_DIR / "manual_ahp_consistency.csv"
REPORT_JSON = WEIGHT_DIR / "manual_ahp_report.json"

# 因子顺序很重要，后面矩阵按这个顺序来
FACTORS = ["poi", "road", "substation", "landuse", "slope"]

FACTOR_CN = {
    "poi": "POI密度 / 充电需求强度",
    "road": "道路距离 / 交通可达性",
    "substation": "变电站距离 / 电网接入条件",
    "landuse": "土地利用适宜性",
    "slope": "坡度 / 地形条件",
}

# Saaty 随机一致性指标 RI
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

# ==========================================================
# 手动 AHP 判断矩阵
# 因子顺序：
# [poi, road, substation, landuse, slope]
#
# 含义：
# POI需求最重要，道路次之，变电站再次，土地利用和坡度影响较小。
#
# 注意：
# a_ij 表示第 i 个因子相对第 j 个因子的重要程度。
# ==========================================================
A = np.array(
    [
        [1, 2, 2, 3, 4],
        [1 / 2, 1, 2, 3, 4],
        [1 / 2, 1 / 2, 1, 2, 3],
        [1 / 3, 1 / 3, 1 / 2, 1, 2],
        [1 / 4, 1 / 4, 1 / 3, 1 / 2, 1],
    ],
    dtype=float,
)


def check_matrix(A):
    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("判断矩阵必须是方阵。")

    if not np.allclose(np.diag(A), np.ones(n)):
        raise ValueError("判断矩阵对角线必须全为 1。")

    if not np.allclose(A * A.T, np.ones((n, n)), atol=1e-8):
        raise ValueError("判断矩阵必须满足互反性：a_ij = 1 / a_ji。")

    if np.any(A <= 0):
        raise ValueError("判断矩阵元素必须为正数。")


def calculate_weights_eigen(A):
    eigvals, eigvecs = np.linalg.eig(A)

    max_idx = np.argmax(eigvals.real)
    lambda_max = float(eigvals[max_idx].real)

    w = eigvecs[:, max_idx].real
    w = np.abs(w)
    w = w / w.sum()

    return w, lambda_max


def calculate_weights_geometric_mean(A):
    gm = np.prod(A, axis=1) ** (1 / A.shape[0])
    w = gm / gm.sum()
    return w


def calculate_consistency(A, weights, lambda_max):
    n = A.shape[0]

    CI = (lambda_max - n) / (n - 1)
    RI = RI_TABLE[n]
    CR = CI / RI if RI != 0 else 0.0

    return CI, RI, CR


def main():
    check_matrix(A)

    n = A.shape[0]

    weights_eigen, lambda_max = calculate_weights_eigen(A)
    weights_gm = calculate_weights_geometric_mean(A)
    CI, RI, CR = calculate_consistency(A, weights_eigen, lambda_max)

    matrix_df = pd.DataFrame(A, index=FACTORS, columns=FACTORS)
    matrix_df.to_csv(MATRIX_CSV, encoding="utf-8-sig")

    weights_df = pd.DataFrame(
        {
            "factor": FACTORS,
            "factor_cn": [FACTOR_CN[f] for f in FACTORS],
            "weight": weights_eigen,
            "weight_geometric_mean": weights_gm,
        }
    )

    weights_df.to_csv(WEIGHTS_CSV, index=False, encoding="utf-8-sig")

    consistency_df = pd.DataFrame(
        [
            {
                "n": n,
                "lambda_max": lambda_max,
                "CI": CI,
                "RI": RI,
                "CR": CR,
                "passed": CR < 0.1,
            }
        ]
    )
    consistency_df.to_csv(CONSISTENCY_CSV, index=False, encoding="utf-8-sig")

    report = {
        "factor_order": FACTORS,
        "factor_cn": FACTOR_CN,
        "matrix": A.tolist(),
        "weights": {
            factor: float(weight) for factor, weight in zip(FACTORS, weights_eigen)
        },
        "lambda_max": lambda_max,
        "CI": CI,
        "RI": RI,
        "CR": CR,
        "passed": bool(CR < 0.1),
    }

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== AHP 判断矩阵 ===")
    print(matrix_df)

    print("\n=== AHP 权重结果 ===")
    print(weights_df[["factor", "factor_cn", "weight"]])

    print("\n=== 一致性检验 ===")
    print(f"lambda_max = {lambda_max:.6f}")
    print(f"CI = {CI:.6f}")
    print(f"RI = {RI:.6f}")
    print(f"CR = {CR:.6f}")

    if CR < 0.1:
        print("一致性检验通过：CR < 0.1")
    else:
        print("一致性检验未通过：CR >= 0.1，需要调整判断矩阵")

    print("\n已输出：")
    print(" -", MATRIX_CSV)
    print(" -", WEIGHTS_CSV)
    print(" -", CONSISTENCY_CSV)
    print(" -", REPORT_JSON)


if __name__ == "__main__":
    main()

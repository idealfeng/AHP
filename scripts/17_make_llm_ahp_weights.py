"""LLM 输出判断矩阵，然后本地计算权重和 CR"""

from pathlib import Path
import os
from dotenv import load_dotenv
import json
import re
import numpy as np
import pandas as pd
from openai import OpenAI

BASE_DIR = Path(r"D:\Paper\毕设")
load_dotenv(BASE_DIR / ".env")

OUT_DIR = BASE_DIR / "results" / "weights" / "llm_ahp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_RESPONSE_TXT = OUT_DIR / "llm_raw_response.txt"
MATRIX_CSV = OUT_DIR / "llm_ahp_matrix.csv"
WEIGHTS_CSV = OUT_DIR / "llm_ahp_weights.csv"
CONSISTENCY_CSV = OUT_DIR / "llm_ahp_consistency.csv"
REPORT_JSON = OUT_DIR / "llm_ahp_report.json"

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

USER_PREFERENCE = """
我认为充电需求密度最重要，道路交通可达性次之，电网接入条件也很重要，
土地利用适宜性有一定影响，坡度影响最小。
"""


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


def build_prompt():
    factor_desc = "\n".join([
        f"- {f}: {FACTOR_CN[f]}"
        for f in FACTORS
    ])

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
如果 i 相对 j 的重要性为 a，则 j 相对 i 的重要性必须为 1/a。

用户偏好：
{USER_PREFERENCE}

请只输出 JSON，不要输出解释文字。格式如下：
{{
  "factors": ["poi", "road", "substation", "landuse", "slope"],
  "matrix": [
    [1, 2, 3, 4, 5],
    [0.5, 1, 2, 3, 4],
    [0.333333, 0.5, 1, 2, 3],
    [0.25, 0.333333, 0.5, 1, 2],
    [0.2, 0.25, 0.333333, 0.5, 1]
  ],
  "reason": "一句话说明权重偏好逻辑"
}}
"""


def call_deepseek():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("请先设置环境变量 DEEPSEEK_API_KEY。")

    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是一个严谨的 GIS-MCDA 与 AHP 专家，只输出合法 JSON。"
            },
            {
                "role": "user",
                "content": build_prompt()
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        stream=False
    )

    content = response.choices[0].message.content

    with open(RAW_RESPONSE_TXT, "w", encoding="utf-8") as f:
        f.write(content)

    return extract_json(content)


def check_matrix(A):
    n = A.shape[0]

    if A.shape != (n, n):
        raise ValueError("判断矩阵必须是方阵。")

    if n != len(FACTORS):
        raise ValueError("判断矩阵维度与因子数量不一致。")

    if not np.allclose(np.diag(A), np.ones(n), atol=1e-6):
        raise ValueError("判断矩阵对角线必须为 1。")

    if np.any(A <= 0):
        raise ValueError("判断矩阵元素必须为正数。")

    # 自动修正轻微互反误差
    for i in range(n):
        A[i, i] = 1.0
        for j in range(i + 1, n):
            A[j, i] = 1.0 / A[i, j]

    return A


def calculate_weights_eigen(A):
    eigvals, eigvecs = np.linalg.eig(A)
    max_idx = np.argmax(eigvals.real)

    lambda_max = float(eigvals[max_idx].real)
    w = eigvecs[:, max_idx].real
    w = np.abs(w)
    w = w / w.sum()

    return w, lambda_max


def calculate_consistency(A, lambda_max):
    n = A.shape[0]
    CI = (lambda_max - n) / (n - 1)
    RI = RI_TABLE[n]
    CR = CI / RI if RI != 0 else 0.0
    return CI, RI, CR


def main():
    result = call_deepseek()

    if "matrix" not in result:
        raise ValueError("模型 JSON 中没有 matrix 字段。")

    A = np.array(result["matrix"], dtype=float)
    A = check_matrix(A)

    weights, lambda_max = calculate_weights_eigen(A)
    CI, RI, CR = calculate_consistency(A, lambda_max)

    matrix_df = pd.DataFrame(A, index=FACTORS, columns=FACTORS)
    matrix_df.to_csv(MATRIX_CSV, encoding="utf-8-sig")

    weights_df = pd.DataFrame({
        "factor": FACTORS,
        "factor_cn": [FACTOR_CN[f] for f in FACTORS],
        "weight": weights
    })
    weights_df.to_csv(WEIGHTS_CSV, index=False, encoding="utf-8-sig")

    consistency_df = pd.DataFrame([{
        "lambda_max": lambda_max,
        "CI": CI,
        "RI": RI,
        "CR": CR,
        "passed": CR < 0.1
    }])
    consistency_df.to_csv(CONSISTENCY_CSV, index=False, encoding="utf-8-sig")

    report = {
        "user_preference": USER_PREFERENCE.strip(),
        "factors": FACTORS,
        "factor_cn": FACTOR_CN,
        "matrix": A.tolist(),
        "weights": {
            f: float(w)
            for f, w in zip(FACTORS, weights)
        },
        "lambda_max": lambda_max,
        "CI": CI,
        "RI": RI,
        "CR": CR,
        "passed": bool(CR < 0.1),
        "llm_reason": result.get("reason", "")
    }

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== LLM-AHP 判断矩阵 ===")
    print(matrix_df)

    print("\n=== LLM-AHP 权重 ===")
    print(weights_df)

    print("\n=== 一致性检验 ===")
    print(f"lambda_max = {lambda_max:.6f}")
    print(f"CI = {CI:.6f}")
    print(f"RI = {RI:.6f}")
    print(f"CR = {CR:.6f}")

    if CR < 0.1:
        print("一致性检验通过：CR < 0.1")
    else:
        print("一致性检验未通过：后续需要加入自动修正机制。")

    print("\n已输出：")
    print(" -", RAW_RESPONSE_TXT)
    print(" -", MATRIX_CSV)
    print(" -", WEIGHTS_CSV)
    print(" -", CONSISTENCY_CSV)
    print(" -", REPORT_JSON)


if __name__ == "__main__":
    main()

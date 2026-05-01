"""LLM-AHP 多情景权重生成脚本"""
from pathlib import Path
import os
import json
import re
import time
from fractions import Fraction

import numpy as np
import pandas as pd
from openai import OpenAI

BASE_DIR = Path(r"D:\Paper\毕设")

OUT_ROOT = BASE_DIR / "results" / "weights" / "llm_scenarios"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUT_ROOT / "llm_scenario_weights_summary.csv"

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

SCENARIOS = {
    "scenario_01_demand_priority": {
        "name_cn": "需求优先型",
        "preference": "我认为充电需求密度最重要，道路交通可达性次之，电网接入条件也需要考虑，土地利用适宜性有一定影响，坡度影响最小。"
    },
    "scenario_02_traffic_priority": {
        "name_cn": "交通优先型",
        "preference": "我认为道路交通可达性最重要，充电需求密度次之，电网接入条件和土地利用适宜性都需要考虑，坡度影响最小。"
    },
    "scenario_03_construction_constraint_priority": {
        "name_cn": "建设约束优先型",
        "preference": "我认为电网接入条件和土地利用适宜性最重要，道路交通可达性次之，充电需求密度有一定影响，坡度影响最小。"
    }
}

MAX_ATTEMPTS = 3


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
        raise ValueError("模型输出中没有找到 JSON。")

    return json.loads(match.group(0))


def build_initial_prompt(preference):
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

规则：
1. matrix 必须是 5×5 矩阵。
2. 对角线必须全为 1。
3. 必须满足互反性：a_ij = 1 / a_ji。
4. 请使用数字小数，不要使用 "1/2" 这类字符串。
5. 尽量保证 AHP 一致性检验 CR < 0.1。
6. 只输出 JSON，不要输出解释文字。

用户偏好：
{preference}

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


def build_repair_prompt(preference, previous_matrix, previous_cr):
    return f"""
你之前生成的 AHP 判断矩阵一致性检验未通过或需要进一步优化。

用户偏好：
{preference}

上一版矩阵：
{json.dumps(previous_matrix, ensure_ascii=False)}

上一版 CR：
{previous_cr}

请在保持用户偏好顺序不变的前提下，调整判断矩阵，使 CR < 0.1。
要求：
1. 因子顺序仍为 ["poi", "road", "substation", "landuse", "slope"]。
2. 输出 5×5 互反矩阵。
3. 对角线全为 1。
4. 使用数字小数，不要使用分数字符串。
5. 只输出 JSON，不要输出解释文字。

输出格式：
{{
  "factors": ["poi", "road", "substation", "landuse", "slope"],
  "matrix": [[...], [...], [...], [...], [...]],
  "reason": "一句话说明修正逻辑"
}}
"""


def call_deepseek(prompt):
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
                "content": "你是严谨的 GIS-MCDA 与 AHP 专家，只输出合法 JSON。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        stream=False
    )

    return response.choices[0].message.content


def matrix_from_result(result):
    if "matrix" not in result:
        raise ValueError("JSON 中没有 matrix 字段。")

    raw = result["matrix"]
    A = np.array([
        [parse_number(x) for x in row]
        for row in raw
    ], dtype=float)

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


def calculate_weights(A):
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


def process_scenario(scenario_id, scenario_info):
    print("\n" + "=" * 60)
    print(f"处理情景：{scenario_id} / {scenario_info['name_cn']}")
    print("=" * 60)

    scenario_dir = OUT_ROOT / scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)

    preference = scenario_info["preference"]

    prompt = build_initial_prompt(preference)

    final_result = None
    final_A = None
    final_weights = None
    final_lambda = None
    final_CI = None
    final_RI = None
    final_CR = None
    final_attempt = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n第 {attempt} 次调用 LLM...")

        raw_text = call_deepseek(prompt)

        raw_path = scenario_dir / f"raw_response_attempt_{attempt}.txt"
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(raw_text)

        result = extract_json(raw_text)
        A = matrix_from_result(result)

        weights, lambda_max = calculate_weights(A)
        CI, RI, CR = calculate_consistency(A, lambda_max)

        print("权重：")
        for f, w in zip(FACTORS, weights):
            print(f"  {f}: {w:.6f}")

        print(f"CR = {CR:.6f}")

        final_result = result
        final_A = A
        final_weights = weights
        final_lambda = lambda_max
        final_CI = CI
        final_RI = RI
        final_CR = CR
        final_attempt = attempt

        if CR < 0.1:
            print("一致性通过。")
            break

        print("一致性未通过，准备让 LLM 修正。")
        prompt = build_repair_prompt(preference, A.tolist(), CR)
        time.sleep(1)

    matrix_df = pd.DataFrame(final_A, index=FACTORS, columns=FACTORS)
    matrix_df.to_csv(scenario_dir / "llm_ahp_matrix.csv", encoding="utf-8-sig")

    weights_df = pd.DataFrame({
        "scenario_id": scenario_id,
        "scenario_name_cn": scenario_info["name_cn"],
        "factor": FACTORS,
        "factor_cn": [FACTOR_CN[f] for f in FACTORS],
        "weight": final_weights
    })
    weights_df.to_csv(scenario_dir / "llm_ahp_weights.csv", index=False, encoding="utf-8-sig")

    consistency_df = pd.DataFrame([{
        "scenario_id": scenario_id,
        "scenario_name_cn": scenario_info["name_cn"],
        "lambda_max": final_lambda,
        "CI": final_CI,
        "RI": final_RI,
        "CR": final_CR,
        "passed": final_CR < 0.1,
        "attempts": final_attempt
    }])
    consistency_df.to_csv(scenario_dir / "llm_ahp_consistency.csv", index=False, encoding="utf-8-sig")

    report = {
        "scenario_id": scenario_id,
        "scenario_name_cn": scenario_info["name_cn"],
        "preference": preference,
        "factors": FACTORS,
        "factor_cn": FACTOR_CN,
        "matrix": final_A.tolist(),
        "weights": {
            f: float(w)
            for f, w in zip(FACTORS, final_weights)
        },
        "lambda_max": final_lambda,
        "CI": final_CI,
        "RI": final_RI,
        "CR": final_CR,
        "passed": bool(final_CR < 0.1),
        "attempts": final_attempt,
        "llm_reason": final_result.get("reason", "")
    }

    with open(scenario_dir / "llm_ahp_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    summary_row = {
        "scenario_id": scenario_id,
        "scenario_name_cn": scenario_info["name_cn"],
        "preference": preference,
        "CR": final_CR,
        "passed": final_CR < 0.1,
        "attempts": final_attempt,
    }

    for f, w in zip(FACTORS, final_weights):
        summary_row[f"weight_{f}"] = float(w)

    return summary_row


def main():
    rows = []

    for scenario_id, scenario_info in SCENARIOS.items():
        row = process_scenario(scenario_id, scenario_info)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("多情景 LLM-AHP 权重汇总")
    print("=" * 60)
    print(summary_df[[
        "scenario_id",
        "scenario_name_cn",
        "weight_poi",
        "weight_road",
        "weight_substation",
        "weight_landuse",
        "weight_slope",
        "CR",
        "passed",
        "attempts"
    ]])

    print("\n已输出汇总：", SUMMARY_CSV)


if __name__ == "__main__":
    main()
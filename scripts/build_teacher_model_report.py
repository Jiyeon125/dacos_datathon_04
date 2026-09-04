from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "teacher_model"


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def main() -> None:
    summary = json.loads((REPORT_DIR / "teacher_model_summary.json").read_text(encoding="utf-8"))
    model_cv = pd.read_csv(REPORT_DIR / "model_cv_results.csv")
    paired = pd.read_csv(REPORT_DIR / "paired_improvement.csv")
    segments = pd.read_csv(REPORT_DIR / "segment_errors.csv")
    outliers = pd.read_csv(REPORT_DIR / "class_model_outliers.csv").head(12)

    model_cv = model_cv.rename(
        columns={"model": "모델", "mae": "MAE", "rmse": "RMSE", "r2": "R2"}
    )
    paired = paired.rename(
        columns={
            "model": "모델",
            "mean_improvement": "MAE개선",
            "ci_low": "95CI하한",
            "ci_high": "95CI상한",
        }
    )
    segments = segments.rename(
        columns={
            "model": "모델",
            "segment": "학교집단",
            "mae": "MAE",
            "median_ae": "중앙절대오차",
            "n": "학교수",
        }
    )

    outlier_keep = [
        c
        for c in [
            "학교명",
            "학교규모집단",
            "학급수_20251001",
            "교원수_20251001",
            "학급수모형_예측교원수",
            "잔차",
            "절대잔차",
        ]
        if c in outliers.columns
    ]
    outliers = outliers[outlier_keep]

    generated_at = datetime.now(timezone.utc).isoformat()
    headline = pd.DataFrame(
        [
            {
                "학급수모형_MAE": round(summary["class_only_mae"], 3),
                "학급수모형_R2": round(summary["class_only_r2"], 4),
                "오차10분위": round(summary["class_residual_q10"], 2),
                "오차90분위": round(summary["class_residual_q90"], 2),
                "분석학교수": summary["n_schools"],
                "소규모학교수": summary["n_small_schools"],
            }
        ]
    )

    source = {
        "id": "teacher_model_analysis",
        "label": "2025 부산 초등학교 교원 회귀 탐색",
        "path": "notebooks/teacher_regression_value.ipynb",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "description": "부산 공립·운영 중·본교 초등학교를 대상으로 반복 교차검증한 교원 수 예측 결과입니다.",
            "sql": "SELECT * FROM read_csv_auto('data/processed/school_master_2025.csv') WHERE \"시도교육청명\" = '부산광역시교육청' AND \"학교급\" = '초등학교' AND \"설립유형\" = '공립' AND \"운영상태\" = '운영' AND \"본교분교구분\" = '본교'",
            "executed_at": generated_at,
            "tables_used": ["data/processed/school_master_2025.csv"],
            "filters": [
                "시도교육청명=부산광역시교육청",
                "학교급=초등학교",
                "설립유형=공립",
                "운영상태=운영",
                "본교/분교=본교",
            ],
            "metric_definitions": [
                "MAE = 학교별 |실제 교원 수 - 교차검증 예측 교원 수|의 평균",
                "R² = 반복 5겹 교차검증 전체 예측의 결정계수",
                "잔차 = 실제 교원 수 - 예측 교원 수",
                "교원 수 = 2025-10-01 정규교원 수 + 기간제교원 수",
            ],
        },
    }

    manifest = {
        "version": 1,
        "surface": "report",
        "title": "교원 회귀모델, 어디까지 쓸 수 있나",
        "description": "부산 초등학교 통합 시뮬레이터에서 교원 회귀모델을 사용할 가치와 한계를 검증한 기술 보고서",
        "generatedAt": generated_at,
        "cards": [
            {
                "id": "mae_card",
                "dataset": "headline",
                "sourceId": "teacher_model_analysis",
                "description": "학급 수만 사용한 선형모형의 학교별 평균 절대오차",
                "metrics": [{"label": "평균 오차", "field": "학급수모형_MAE", "format": "number"}],
            },
            {
                "id": "r2_card",
                "dataset": "headline",
                "sourceId": "teacher_model_analysis",
                "description": "현재 학교 간 교원 수 차이를 학급 수 모형이 설명하는 정도",
                "metrics": [{"label": "교차검증 R²", "field": "학급수모형_R2", "format": "number"}],
            },
            {
                "id": "school_count_card",
                "dataset": "headline",
                "sourceId": "teacher_model_analysis",
                "description": "모델 학습과 검증에 포함된 부산 공립 운영 본교 초등학교 수",
                "metrics": [{"label": "분석 학교", "field": "분석학교수", "format": "number"}],
            },
        ],
        "charts": [
            {
                "id": "model_mae_chart",
                "title": "모델별 교차검증 평균 절대오차",
                "subtitle": "낮을수록 현재 교원 수 예측이 정확합니다.",
                "intent": "comparison",
                "question": "복잡한 변수 추가가 학급 수 단독모형보다 오차를 줄이는가?",
                "rationale": "모델별 성능 차이를 동일 척도에서 비교합니다.",
                "type": "horizontalBar",
                "dataset": "model_cv",
                "sourceId": "teacher_model_analysis",
                "encodings": {
                    "x": {"field": "모델", "type": "nominal", "label": "모델"},
                    "y": {"field": "MAE", "type": "quantitative", "label": "평균 절대오차", "unit": "명"},
                    "tooltip": [
                        {"field": "RMSE", "type": "quantitative", "label": "RMSE"},
                        {"field": "R2", "type": "quantitative", "label": "R²"},
                    ],
                },
                "xAxisTitle": "평균 절대오차(명)",
                "valueFormat": "number",
                "layout": "full",
            },
            {
                "id": "segment_mae_chart",
                "title": "학교 규모 집단별 평균 절대오차",
                "subtitle": "소규모학교와 그 외 학교에서 모델 오차가 어떻게 다른지 비교합니다.",
                "intent": "comparison",
                "question": "소규모학교에서도 모델 오차가 허용 가능한가?",
                "rationale": "통합 검토 대상과 수용학교 집단의 정확도를 분리해 확인합니다.",
                "type": "bar",
                "dataset": "segment_errors",
                "sourceId": "teacher_model_analysis",
                "encodings": {
                    "x": {"field": "모델", "type": "nominal", "label": "모델"},
                    "y": {"field": "MAE", "type": "quantitative", "label": "평균 절대오차", "unit": "명"},
                    "color": {"field": "학교집단", "type": "nominal", "label": "학교 집단"},
                    "tooltip": [
                        {"field": "학교수", "type": "quantitative", "label": "학교 수"},
                        {"field": "중앙절대오차", "type": "quantitative", "label": "중앙 절대오차"},
                    ],
                },
                "yAxisTitle": "평균 절대오차(명)",
                "valueFormat": "number",
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "paired_table",
                "title": "학급 수 단독모형 대비 모델별 MAE 개선",
                "subtitle": "양수면 개선, 음수면 악화입니다. 신뢰구간이 0을 포함하면 우열이 확실하지 않습니다.",
                "dataset": "paired_improvement",
                "sourceId": "teacher_model_analysis",
                "density": "compact",
                "columns": [
                    {"field": "모델", "label": "모델"},
                    {"field": "MAE개선", "label": "MAE 개선(명)", "format": "number", "movement": True},
                    {"field": "95CI하한", "label": "95% CI 하한", "format": "number"},
                    {"field": "95CI상한", "label": "95% CI 상한", "format": "number"},
                ],
            },
            {
                "id": "outlier_table",
                "title": "학급 수 모형의 오차가 큰 학교",
                "subtitle": "회귀값을 정원으로 쓰기 어려운 이유를 보여주는 사례입니다.",
                "dataset": "outliers",
                "sourceId": "teacher_model_analysis",
                "density": "compact",
                "columns": [
                    {"field": c, "label": c, "format": "number" if c not in {"학교명", "학교규모집단"} else None}
                    for c in outlier_keep
                ],
            },
        ],
        "sources": [source],
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# 교원 회귀모델, 어디까지 쓸 수 있나"},
            {
                "id": "executive_summary",
                "type": "markdown",
                "body": "## Executive Summary\n\n**결론: 보조 지표로는 가치가 있지만, 통합 후 교원 정원을 계산하는 핵심 로직으로 쓰면 안 됩니다.** 2025년 부산 공립 운영 본교 초등학교 296개를 반복 교차검증한 결과, 학급 수만 사용한 단순 선형모형이 가장 정확했습니다. 학생 수·특수학급·교실·교지·지역을 더 넣은 모델은 오차를 줄이지 못했습니다. 따라서 대시보드의 주 계산은 `학년별 학생 합산 → 학급 재편성 → 공식 직위 규칙 적용` 순서로 두고, 회귀값은 현재 배치 패턴과 얼마나 다른지 보여주는 참고 범위나 이상치 점검에만 사용해야 합니다.",
                "sourceId": "teacher_model_analysis",
            },
            {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["mae_card", "r2_card", "school_count_card"]},
            {
                "id": "method",
                "type": "markdown",
                "body": "## 분석 설계\n\n- **대상:** 2025-10-01 부산 공립·운영 중·본교 초등학교 296개\n- **목표변수:** 교원 수(정규교원 + 기간제교원)\n- **검증:** 5겹 교차검증을 10회 반복해 총 50개 폴드 평가\n- **비교모델:** 평균 예측, 학급 수 선형, 학급 수+학생 수 선형, 확장 Ridge, Random Forest\n- **누수 방지:** 학생/교원처럼 목표값으로 계산되는 변수는 입력에서 제외\n\n이 평가는 현재 단면의 배치 패턴을 얼마나 재현하는지 보는 것이며, 통합이 교원 배치에 미치는 인과효과나 공식 정원 규칙을 추정하는 분석은 아닙니다.",
                "sourceId": "teacher_model_analysis",
            },
            {"id": "model_mae_block", "type": "chart", "chartId": "model_mae_chart"},
            {
                "id": "finding_one",
                "type": "markdown",
                "body": "## 핵심 결과 1 — 복잡한 모델의 추가가치는 확인되지 않음\n\n학급 수 단독 선형모형의 MAE는 **1.78명**으로 가장 낮았습니다. 학급 수에 학생 수를 추가한 선형모형은 1.80명, 확장 Ridge는 1.85명, Random Forest는 1.95명이었습니다. 같은 학교 예측을 짝지어 비교한 부트스트랩에서도 어느 확장모델도 학급 수 단독모형보다 안정적으로 개선되지 않았습니다. 현재 데이터에서 교원 수 차이의 대부분은 이미 학급 수에 담겨 있습니다.",
                "sourceId": "teacher_model_analysis",
            },
            {"id": "paired_block", "type": "table", "tableId": "paired_table"},
            {"id": "segment_mae_block", "type": "chart", "chartId": "segment_mae_chart"},
            {
                "id": "finding_two",
                "type": "markdown",
                "body": "## 핵심 결과 2 — 평균 성능이 좋아도 정원 산정에는 부족함\n\n학급 수 모형의 중앙 80% 잔차는 약 **-2.80명~+2.86명**이지만, 일부 학교는 실제 교원 수와 예측치가 약 8~9명 차이 납니다. 교장·교감·보건·영양·특수·상담 등 학교별 고정 또는 선택 직위와 예외가 단순 학급 수 관계에 모두 설명되지 않기 때문입니다. 또 세부 직위 컬럼은 서로 중복되는 분류축이라 단순 합산할 수 없습니다.",
                "sourceId": "teacher_model_analysis",
            },
            {"id": "outlier_block", "type": "table", "tableId": "outlier_table"},
            {
                "id": "recommendation",
                "type": "markdown",
                "body": "## 대시보드 적용 권고\n\n1. **핵심 계산에는 미적용:** `A+B 학년별 학생 수 → 기준인원으로 일반학급 재편성`을 먼저 계산합니다.\n2. **교원은 규칙 기반으로 분해:** 필요 학급 담당 + 고정 직위 + 자료로 확인 가능한 선택 직위를 합산합니다. 공식 규칙이 확보되지 않은 직위는 미산정으로 표시합니다.\n3. **회귀는 보조 표시만:** 필요하면 `수용학교 현재 교원 / 두 학교 현재 교원 합 / 학급 수 기반 관측패턴 참고값`을 함께 보여주되, 참고값에는 약 ±3명 수준의 경험적 범위와 비공식 추정이라는 라벨을 붙입니다.\n4. **가장 적합한 용도:** 규칙 기반 결과가 현재 학교들의 일반적인 배치 패턴에서 크게 벗어나는지 점검하는 QA 또는 이상치 탐색입니다.\n\n회귀 예측값을 반올림해 ‘필요 교원 수’로 제시하거나, 값이 줄었다는 이유만으로 통합이 좋다고 판단하는 방식은 사용하지 않습니다.",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": "## 한계와 다음 검증\n\n- 2025년 한 시점의 부산 학교만 사용해 연도 변화와 실제 통합 이후 재배치를 학습하지 못했습니다.\n- 수용학교는 소규모학교가 아닐 수 있어 전체 학교 오차를 함께 봐야 합니다.\n- 4월 학생·특수학급·교실 자료와 10월 교원·전체 학급 자료의 시점 차이가 있습니다.\n- 통합 시나리오에 모델을 적용할 때는 재편성 일반학급에 두 학교의 현재 특수학급 합을 더해 10월 전체 학급 정의와 맞췄지만, 이것도 특수학급 재편성을 예측한 것은 아닙니다.\n- 향후 실제 교원 정원 배정표, 직위별 배치 규칙, 과거 통합학교의 전후 자료가 확보되면 규칙 기반 결과와 회귀 잔차를 다시 검증해야 합니다.",
                "sourceId": "teacher_model_analysis",
            },
        ],
    }

    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": records(headline),
                "model_cv": records(model_cv),
                "paired_improvement": records(paired),
                "segment_errors": records(segments),
                "outliers": records(outliers),
            },
        },
        "sources": [source],
    }

    (REPORT_DIR / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(REPORT_DIR / "artifact.json")


if __name__ == "__main__":
    main()

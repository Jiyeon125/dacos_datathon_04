from __future__ import annotations

import json
import textwrap
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "teacher_regression_value.ipynb"
OUTPUT_DIR = ROOT / "reports" / "teacher_model"


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbformat.v4.new_code_cell(textwrap.dedent(source).strip())


def build_notebook():
    cells = [
        markdown(
            """
            # 교원 회귀모델 적용가치 탐색

            ## tl;dr

            실행 후 검증 결과로 자동 갱신됩니다.
            """
        ),
        markdown(
            """
            ## Context & Methods

            이 분석은 회귀모델을 **통합 후 공식 교원 정원 산정식**으로 쓸 수 있는지, 아니면
            **현재 학교들의 교원 배치 패턴을 설명하는 보조자료**로만 쓸 가치가 있는지를 구분한다.

            ### Key Assumptions

            - 분석단위는 2025년 10월 1일 부산 공립·운영 중·본교 초등학교 296개교다.
            - 목표값은 2025년 10월 실제 총교원 현원이다.
            - 학급·학생은 10월 자료를 우선 사용하고, 특수학급·시설은 4월 자료라 시점차를 한계로 둔다.
            - `교원 1인당 학생수`처럼 목표값을 역산할 수 있는 변수는 입력에서 제외한다.
            - 무작위 5겹 교차검증을 10회 반복한다. 이는 현재 학교에 대한 예측 성능이지 통폐합의 인과효과가 아니다.
            """
        ),
        code(
            """
            from pathlib import Path
            import json
            import warnings

            import numpy as np
            import pandas as pd
            import plotly.express as px
            import plotly.graph_objects as go
            from sklearn.base import clone
            from sklearn.compose import ColumnTransformer
            from sklearn.dummy import DummyRegressor
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LinearRegression, RidgeCV
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            from sklearn.model_selection import RepeatedKFold
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler

            warnings.filterwarnings("ignore", category=FutureWarning)
            ROOT = Path.cwd()
            OUTPUT_DIR = ROOT / "reports" / "teacher_model"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            RANDOM_STATE = 42
            """
        ),
        markdown("## Data\n\n원자료의 교원 구성 합계가 총교원과 일치하는지 확인하고, 현재 배포 마스터와 KEDI로 결합한다."),
        code(
            """
            master = pd.read_csv(ROOT / "data/processed/school_master_2025.csv", encoding="utf-8-sig")
            resources = pd.read_csv(ROOT / "data/raw/education/data2_school_resources_20251001.csv", encoding="utf-8-sig")

            kedi = "학교코드(KEDI)"
            for frame in (master, resources):
                frame[kedi] = frame[kedi].astype("string").str.strip().str.replace(r"\\.0$", "", regex=True)

            operating_states = {"기존(원)교", "신설(원)교"}
            resources = resources.loc[
                resources["시도"].eq("부산")
                & resources["학교급"].eq("초등학교")
                & resources["본분교"].eq("본교")
                & resources["설립"].eq("공립")
                & resources["상태"].isin(operating_states)
            ].copy()

            role_columns = [
                "교원수_정규_계", "교원수_정규_교장_계", "교원수_정규_교감_계",
                "교원수_정규_수석_계", "교원수_정규_보직_계", "교원수_정규_일반_계",
                "교원수_정규_특수_계", "교원수_정규_상담_계", "교원수_정규_사서_계",
                "교원수_정규_실기_계", "교원수_정규_보건_계", "교원수_정규_영양_계",
                "교원수_기간제_계",
            ]
            role_data = resources[[kedi, *role_columns]].copy()
            for column in role_columns:
                role_data[column] = pd.to_numeric(role_data[column], errors="coerce")

            data = master.merge(role_data, on=kedi, how="left", validate="one_to_one")
            target = "교원수_20251001"
            regular_role_columns = [
                "교원수_정규_교장_계", "교원수_정규_교감_계", "교원수_정규_수석_계",
                "교원수_정규_보직_계", "교원수_정규_일반_계", "교원수_정규_특수_계",
                "교원수_정규_상담_계", "교원수_정규_사서_계", "교원수_정규_실기_계",
                "교원수_정규_보건_계", "교원수_정규_영양_계",
            ]
            data["정규역할합"] = data[regular_role_columns].sum(axis=1)
            data["정규기간제합"] = data["교원수_정규_계"] + data["교원수_기간제_계"]

            role_sum_matches = int(data["교원수_정규_계"].eq(data["정규역할합"]).sum())
            qa = pd.DataFrame([
                {"검증": "분석 학교 수", "관측": len(data), "기대": 296, "통과": len(data) == 296},
                {"검증": "KEDI 유일성", "관측": data[kedi].nunique(), "기대": 296, "통과": data[kedi].is_unique},
                {"검증": "목표값 결측", "관측": int(data[target].isna().sum()), "기대": 0, "통과": data[target].notna().all()},
                {"검증": "총교원=정규+기간제 일치", "관측": int(data[target].eq(data["정규기간제합"]).sum()), "기대": 296, "통과": data[target].eq(data["정규기간제합"]).all()},
                {"검증": "세부 직위 단순합 가능 여부", "관측": role_sum_matches, "기대": "중복분류로 합산하지 않음", "통과": True},
            ])
            if not qa.iloc[:4]["통과"].all():
                raise AssertionError(qa.iloc[:4].loc[~qa.iloc[:4]["통과"]].to_dict("records"))
            display(qa)
            print("주의: 보직·일반과 보건·영양 등 세부 열은 분류축이 겹쳐 단순 합산하지 않습니다.")
            """
        ),
        markdown(
            """
            ## Results

            네 모델을 동일한 반복 교차검증으로 비교한다. 평균예측은 최소 기준, 학급수 선형모델은 가장 해석 가능한 기준,
            확장 Ridge는 학생·특수학급·시설·지역을 추가한 선형모델, Random Forest는 비선형성이 주는 추가 이득을 점검한다.
            """
        ),
        code(
            """
            class_column = "학급수_20251001"
            student_column = "학생수_20251001"
            special_class_column = "특수학급수_20250401_반합계"
            numeric_expanded = [
                class_column, student_column, special_class_column,
                "일반교실수_20250401", "전체교실수_20250401", "교지면적_20250401",
            ]
            categorical_expanded = ["행정구_20251001", "지역규모_20251001"]
            model_columns = list(dict.fromkeys([*numeric_expanded, *categorical_expanded]))
            X = data[model_columns].copy()
            y = pd.to_numeric(data[target], errors="raise")

            expanded_preprocessor = ColumnTransformer([
                ("numeric", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]), numeric_expanded),
                ("category", OneHotEncoder(handle_unknown="ignore"), categorical_expanded),
            ])
            numeric_preprocessor = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ])

            models = {
                "평균 기준": Pipeline([("model", DummyRegressor(strategy="mean"))]),
                "학급수 선형": Pipeline([("model", LinearRegression())]),
                "학급수+학생 선형": Pipeline([("model", LinearRegression())]),
                "확장 Ridge": Pipeline([
                    ("prepare", expanded_preprocessor),
                    ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
                ]),
                "비선형 Random Forest": Pipeline([
                    ("prepare", numeric_preprocessor),
                    ("model", RandomForestRegressor(
                        n_estimators=200,
                        min_samples_leaf=4,
                        max_features=0.8,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    )),
                ]),
            }
            feature_sets = {
                "평균 기준": [class_column],
                "학급수 선형": [class_column],
                "학급수+학생 선형": [class_column, student_column],
                "확장 Ridge": model_columns,
                "비선형 Random Forest": numeric_expanded,
            }

            cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
            prediction_rows = []
            fold_rows = []
            for model_name, estimator in models.items():
                columns = feature_sets[model_name]
                for fold, (train_index, test_index) in enumerate(cv.split(X), start=1):
                    fitted = clone(estimator).fit(X.iloc[train_index][columns], y.iloc[train_index])
                    prediction = fitted.predict(X.iloc[test_index][columns])
                    actual = y.iloc[test_index].to_numpy()
                    fold_rows.append({
                        "모델": model_name,
                        "fold": fold,
                        "MAE": mean_absolute_error(actual, prediction),
                        "RMSE": mean_squared_error(actual, prediction) ** 0.5,
                        "R2": r2_score(actual, prediction),
                    })
                    prediction_rows.extend([
                        {"모델": model_name, "row_id": int(row_id), "실제교원": float(actual_value), "예측교원": float(predicted_value)}
                        for row_id, actual_value, predicted_value in zip(test_index, actual, prediction)
                    ])

            fold_metrics = pd.DataFrame(fold_rows)
            predictions = pd.DataFrame(prediction_rows)
            cv_results = (
                fold_metrics.groupby("모델", as_index=False)
                .agg(
                    MAE=("MAE", "mean"), MAE_SD=("MAE", "std"),
                    RMSE=("RMSE", "mean"), RMSE_SD=("RMSE", "std"),
                    R2=("R2", "mean"), R2_SD=("R2", "std"),
                )
                .sort_values("MAE")
                .reset_index(drop=True)
            )
            cv_results.to_csv(OUTPUT_DIR / "model_cv_results.csv", index=False, encoding="utf-8-sig")
            display(cv_results.round(3))
            """
        ),
        code(
            """
            # 같은 학교가 반복검증에서 10번 예측되므로 학교별 평균 예측을 만든다.
            school_predictions = (
                predictions.groupby(["모델", "row_id"], as_index=False)
                .agg(실제교원=("실제교원", "first"), 예측교원=("예측교원", "mean"))
            )
            identity_columns = [kedi, "학교명_20251001", "행정구_20251001", "주분석대상_소규모공립_정책2026", class_column, student_column, special_class_column]
            identity = data[identity_columns].reset_index(names="row_id")
            school_predictions = school_predictions.merge(identity, on="row_id", how="left", validate="many_to_one")
            school_predictions["잔차_실제-예측"] = school_predictions["실제교원"] - school_predictions["예측교원"]
            school_predictions["절대오차"] = school_predictions["잔차_실제-예측"].abs()
            school_predictions.to_csv(OUTPUT_DIR / "school_cv_predictions.csv", index=False, encoding="utf-8-sig")

            class_only_errors = school_predictions.loc[school_predictions["모델"].eq("학급수 선형"), ["row_id", "절대오차"]].rename(columns={"절대오차": "학급수선형_절대오차"})
            paired_rows = []
            rng = np.random.default_rng(RANDOM_STATE)
            for model_name in ["학급수+학생 선형", "확장 Ridge", "비선형 Random Forest"]:
                other = school_predictions.loc[school_predictions["모델"].eq(model_name), ["row_id", "절대오차"]]
                paired = class_only_errors.merge(other, on="row_id", validate="one_to_one")
                improvement = paired["학급수선형_절대오차"] - paired["절대오차"]
                bootstrap_means = np.array([
                    rng.choice(improvement.to_numpy(), size=len(improvement), replace=True).mean()
                    for _ in range(5000)
                ])
                paired_rows.append({
                    "비교모델": model_name,
                    "학급수선형대비_MAE개선": improvement.mean(),
                    "개선_95CI_하한": np.quantile(bootstrap_means, 0.025),
                    "개선_95CI_상한": np.quantile(bootstrap_means, 0.975),
                })
            paired_improvement = pd.DataFrame(paired_rows).sort_values("학급수선형대비_MAE개선", ascending=False)
            paired_improvement.to_csv(OUTPUT_DIR / "paired_improvement.csv", index=False, encoding="utf-8-sig")
            display(paired_improvement.round(3))
            """
        ),
        code(
            """
            # 차트 1: 비교 목적은 모델별 오차 크기다. 단일 파란색 막대와 오차막대를 사용한다.
            chart_data = cv_results.sort_values("MAE", ascending=True)
            fig = go.Figure(go.Bar(
                x=chart_data["MAE"],
                y=chart_data["모델"],
                orientation="h",
                marker_color="#3569B0",
                error_x=dict(type="data", array=chart_data["MAE_SD"], visible=True, color="#263238"),
                text=chart_data["MAE"].map(lambda value: f"{value:.2f}명"),
                textposition="outside",
            ))
            fig.update_layout(
                title="모델별 교원 현원 예측오차",
                xaxis_title="반복 5겹 교차검증 MAE(명)",
                yaxis_title=None,
                height=390,
                margin=dict(l=160, r=50, t=60, b=50),
            )
            fig.show()
            """
        ),
        code(
            """
            # 소규모학교와 그 외 학교에서 오차가 다르게 나타나는지 확인한다.
            segment_errors = (
                school_predictions.groupby(["모델", "주분석대상_소규모공립_정책2026"], as_index=False)
                .agg(MAE=("절대오차", "mean"), 중앙절대오차=("절대오차", "median"), 학교수=(kedi, "nunique"))
            )
            segment_errors["학교구분"] = segment_errors["주분석대상_소규모공립_정책2026"].map({True: "소규모학교 92개교", False: "그 외 204개교"})
            segment_errors.to_csv(OUTPUT_DIR / "segment_errors.csv", index=False, encoding="utf-8-sig")
            display(segment_errors[["모델", "학교구분", "MAE", "중앙절대오차", "학교수"]].round(3))

            fig = px.bar(
                segment_errors,
                x="모델", y="MAE", color="학교구분", barmode="group",
                color_discrete_map={"소규모학교 92개교": "#F28E2B", "그 외 204개교": "#3569B0"},
                text_auto=".2f",
                title="학교 규모별 교원 현원 예측오차",
            )
            fig.update_layout(yaxis_title="학교별 평균 절대오차(명)", xaxis_title=None, height=420)
            fig.show()
            """
        ),
        code(
            """
            # 가장 해석 가능한 학급수 선형모델의 계수와 잔차 구조를 확인한다.
            class_model = LinearRegression().fit(data[[class_column]], y)
            class_intercept = float(class_model.intercept_)
            class_slope = float(class_model.coef_[0])
            class_predictions = school_predictions.loc[school_predictions["모델"].eq("학급수 선형")].copy()

            data["교원-학급"] = data[target] - data[class_column]
            role_summary = data[[target, class_column, "교원-학급", "교원수_정규_교장_계", "교원수_정규_교감_계", "교원수_정규_특수_계", "교원수_정규_보건_계", "교원수_정규_영양_계", "교원수_기간제_계"]].describe(percentiles=[0.1, 0.5, 0.9]).T
            display(pd.DataFrame({"절편": [class_intercept], "학급당 기울기": [class_slope]}).round(3))
            display(role_summary[["mean", "std", "10%", "50%", "90%"]].round(2))

            outliers = pd.concat([
                class_predictions.nsmallest(8, "잔차_실제-예측"),
                class_predictions.nlargest(8, "잔차_실제-예측"),
            ]).sort_values("잔차_실제-예측")
            outliers.to_csv(OUTPUT_DIR / "class_model_outliers.csv", index=False, encoding="utf-8-sig")
            display(outliers[["학교명_20251001", "행정구_20251001", class_column, student_column, "실제교원", "예측교원", "잔차_실제-예측"]].round(2))
            """
        ),
        markdown(
            """
            ## 통합 시나리오에 넣으면 무엇이 나오는가

            가장 해석 가능한 학급수 선형모델과 교차검증 잔차를 이용해 1,481개 시나리오에 대한
            `관측학교 패턴상 기대 교원`을 계산한다. 이는 공식 정원이나 통합 후 실제 배치가 아니다.
            """
        ),
        code(
            """
            from src.resource_benchmark import build_resource_scenario_table
            from src.data_loader import load_bundle
            from src.candidate_generator import PAIR_A_CODE, PAIR_B_CODE

            bundle = load_bundle()
            scenarios = build_resource_scenario_table(bundle.master, bundle.candidate_pairs)
            b_teacher = data.set_index(kedi)[target]
            a_teacher = data.set_index(kedi)[target]
            special_classes = data.set_index(kedi)[special_class_column]
            scenarios["현재_특수학급합"] = (
                scenarios[PAIR_A_CODE].map(special_classes) + scenarios[PAIR_B_CODE].map(special_classes)
            )
            scenarios["모델입력_통합후전체학급"] = scenarios["classes_after"] + scenarios["현재_특수학급합"]
            scenarios["회귀_기대교원"] = class_model.predict(
                scenarios[["모델입력_통합후전체학급"]].rename(columns={"모델입력_통합후전체학급": class_column})
            )

            scenarios["수용학교_현재교원"] = scenarios[PAIR_B_CODE].map(b_teacher)
            scenarios["두학교_현재교원합"] = scenarios[PAIR_B_CODE].map(b_teacher) + scenarios[PAIR_A_CODE].map(a_teacher)
            residual_q10, residual_q90 = class_predictions["잔차_실제-예측"].quantile([0.1, 0.9])
            scenarios["회귀_기대교원_80pct하한"] = (scenarios["회귀_기대교원"] + residual_q10).clip(lower=0)
            scenarios["회귀_기대교원_80pct상한"] = scenarios["회귀_기대교원"] + residual_q90
            scenarios["회귀값-B현재"] = scenarios["회귀_기대교원"] - scenarios["수용학교_현재교원"]
            scenarios["A+B현재합-회귀값"] = scenarios["두학교_현재교원합"] - scenarios["회귀_기대교원"]
            scenarios.to_csv(OUTPUT_DIR / "scenario_teacher_predictions.csv", index=False, encoding="utf-8-sig")

            scenario_summary = scenarios[["회귀_기대교원", "회귀_기대교원_80pct하한", "회귀_기대교원_80pct상한", "회귀값-B현재", "A+B현재합-회귀값"]].describe(percentiles=[0.1, 0.5, 0.9]).T
            display(scenario_summary[["mean", "std", "10%", "50%", "90%"]].round(2))
            """
        ),
        code(
            """
            best_non_dummy = cv_results.loc[~cv_results["모델"].eq("평균 기준")].iloc[0]
            class_result = cv_results.loc[cv_results["모델"].eq("학급수 선형")].iloc[0]
            ridge_result = cv_results.loc[cv_results["모델"].eq("확장 Ridge")].iloc[0]
            small_class_mae = float(segment_errors.loc[
                segment_errors["모델"].eq("학급수 선형") & segment_errors["주분석대상_소규모공립_정책2026"].eq(True), "MAE"
            ].iloc[0])
            other_class_mae = float(segment_errors.loc[
                segment_errors["모델"].eq("학급수 선형") & segment_errors["주분석대상_소규모공립_정책2026"].eq(False), "MAE"
            ].iloc[0])
            best_improvement = float(class_result["MAE"] - best_non_dummy["MAE"])
            use_value = "보조 지표로 제한적 가치" if best_improvement < 0.30 else "보조 추정범위로 활용 검토"

            summary = {
                "n_schools": int(len(data)),
                "n_small_schools": int(data["주분석대상_소규모공립_정책2026"].sum()),
                "cv_folds": 50,
                "class_only_mae": float(class_result["MAE"]),
                "class_only_rmse": float(class_result["RMSE"]),
                "class_only_r2": float(class_result["R2"]),
                "best_model": str(best_non_dummy["모델"]),
                "best_model_mae": float(best_non_dummy["MAE"]),
                "best_improvement_vs_class_only": best_improvement,
                "ridge_mae": float(ridge_result["MAE"]),
                "class_model_intercept": class_intercept,
                "class_model_slope": class_slope,
                "small_school_class_model_mae": small_class_mae,
                "other_school_class_model_mae": other_class_mae,
                "class_residual_q10": float(residual_q10),
                "class_residual_q90": float(residual_q90),
                "scenario_count": int(len(scenarios)),
                "scenario_median_change_vs_b": float(scenarios["회귀값-B현재"].median()),
                "scenario_median_saving_vs_current_sum": float(scenarios["A+B현재합-회귀값"].median()),
                "decision": use_value,
                "official_staffing_formula": False,
            }
            (OUTPUT_DIR / "teacher_model_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            display(pd.Series(summary, name="값").to_frame())
            """
        ),
        markdown(
            """
            ## Takeaways

            실행 후 검증 결과로 자동 갱신됩니다.
            """
        ),
    ]
    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
    )


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    executed = client.execute()

    summary = json.loads((OUTPUT_DIR / "teacher_model_summary.json").read_text(encoding="utf-8"))
    executed.cells[0].source = textwrap.dedent(
        f"""
        # 교원 회귀모델 적용가치 탐색

        ## tl;dr

        - **결론:** {summary['decision']}. 현재 데이터로는 회귀를 공식 교원 정원 산정식으로 사용할 수 없다.
        - 학급수 선형모델의 반복 교차검증 MAE는 **{summary['class_only_mae']:.2f}명**, R²는 **{summary['class_only_r2']:.3f}**다.
        - 최저 MAE 모델은 **{summary['best_model']} {summary['best_model_mae']:.2f}명**이며, 학급수 선형보다 개선폭은 **{summary['best_improvement_vs_class_only']:.2f}명**이다.
        - 소규모학교에서 학급수 모델 MAE는 **{summary['small_school_class_model_mae']:.2f}명**, 그 외 학교는 **{summary['other_school_class_model_mae']:.2f}명**이다.
        - 권장 용도는 `B 현재 현원`과 `A+B 현재 현원 합` 사이에서 관측학교 패턴상 기대 범위를 보여주는 **참고값·이상사례 탐지**다.
        """
    ).strip()
    executed.cells[-1].source = textwrap.dedent(
        f"""
        ## Takeaways

        1. 학급수 하나만으로도 현재 교원 현원의 대부분을 설명하지만 학교별 오차가 평균 {summary['class_only_mae']:.2f}명 남는다.
        2. 복잡한 입력을 추가한 최저오차 모델의 개선은 {summary['best_improvement_vs_class_only']:.2f}명이다. 정확도 이득이 작다면 설명력을 잃고 복잡한 모델을 채택할 이유가 약하다.
        3. 이 데이터는 실제 통합 후 배치결과가 아니라 현재 운영학교의 횡단면이다. 회귀값은 정책결정값·법정정원·통합효과가 아니다.
        4. 대시보드에 넣는다면 중앙값 하나보다 교차검증 잔차 80% 범위({summary['class_residual_q10']:+.1f}~{summary['class_residual_q90']:+.1f}명)를 함께 표시하고 `관측패턴상 참고 범위`라고 이름 붙인다.
        5. 우선순위는 역할별 공식 규칙 구현이다. 회귀는 규칙으로 설명되지 않는 차이와 예외 학교를 찾는 보조층으로 둔다.
        """
    ).strip()
    nbformat.write(executed, NOTEBOOK_PATH)
    print(f"executed notebook: {NOTEBOOK_PATH}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

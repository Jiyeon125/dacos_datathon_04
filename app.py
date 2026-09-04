from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.candidate_generator import PAIR_A_CODE, PAIR_B_CODE
from src.config import PROCESSED_DIR
from src.data_loader import load_bundle
from src.resource_simulator import resource_comparison_table, simulate_resource_change
from src.scenario_engine import run_scenario
from src.schema import DISTRICT, KEDI, SCHOOL_NAME, STUDENTS


st.set_page_config(page_title="학교 통합 교육여건 시뮬레이터", page_icon="🏫", layout="wide")


@st.cache_resource(show_spinner=False)
def get_bundle(data_dir: str):
    return load_bundle(Path(data_dir))


def format_number(value, digits: int = 1, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "자료 없음"
    return f"{float(value):,.{digits}f}{suffix}"


def class_size_figure(before: float, after: float) -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=["통합 전", "통합 후"],
            y=[before, after],
            text=[f"{before:.1f}명", f"{after:.1f}명"],
            textposition="outside",
            marker_color=["#4878CF", "#F28E2B"],
        )
    )
    figure.add_hline(y=28, line_dash="dash", line_color="#C43C39", annotation_text="28명 참고선")
    figure.update_layout(
        title="학급당 학생 수",
        yaxis_title="명",
        showlegend=False,
        margin=dict(l=20, r=20, t=55, b=20),
        height=330,
    )
    return figure


def add_polygon_boundary(figure: go.Figure, geometry) -> None:
    polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    for polygon in polygons:
        x, y = polygon.exterior.xy
        figure.add_trace(
            go.Scatter(
                x=list(x),
                y=list(y),
                mode="lines",
                line=dict(color="#566573", width=2),
                name="A 통학구역",
                hoverinfo="skip",
            )
        )


def accessibility_figure(bundle, grid, a_code: str, b_code: str) -> go.Figure:
    zone = bundle.catchments.set_index(KEDI).loc[a_code].geometry
    if isinstance(zone, pd.Series):
        zone = zone.iloc[0]
    points = bundle.school_points.set_index(KEDI)
    a_point = points.loc[a_code].geometry
    b_point = points.loc[b_code].geometry
    figure = go.Figure()
    add_polygon_boundary(figure, zone)
    figure.add_trace(
        go.Scatter(
            x=grid.geometry.x,
            y=grid.geometry.y,
            mode="markers",
            marker=dict(
                size=9,
                color=grid["추가접근거리_km"],
                colorscale="RdYlBu_r",
                colorbar=dict(title="추가거리<br>(km)"),
                line=dict(width=0.5, color="white"),
            ),
            text=[
                f"현재 {current:.2f}km<br>통합 후 {after:.2f}km<br>변화 {added:+.2f}km"
                for current, after, added in zip(
                    grid["현재거리_km"], grid["통합후거리_km"], grid["추가접근거리_km"]
                )
            ],
            hovertemplate="%{text}<extra>격자</extra>",
            name="250m 격자",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[a_point.x, b_point.x],
            y=[a_point.y, b_point.y],
            mode="markers+text",
            marker=dict(size=15, color=["#2166AC", "#B2182B"], symbol=["x", "star"]),
            text=["현재 학교 A", "후보 학교 B"],
            textposition=["top left", "top right"],
            name="학교",
        )
    )
    figure.update_layout(
        title="A 통학구역의 접근거리 변화",
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=10, r=10, t=70, b=10),
        height=500,
    )
    return figure


def scenario_comparison(bundle, a_code: str, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, pair in pairs.sort_values("학교간직선거리_km").iterrows():
        resource = simulate_resource_change(bundle.master, a_code, pair[PAIR_B_CODE])
        rows.append(
            {
                "후보 학교": pair["후보학교명"],
                "행정구": pair["후보학교_행정구"],
                "학교간 직선거리(km)": round(pair["학교간직선거리_km"], 2),
                "이동 학생 수": resource["moving_students"],
                "통합 후 학급당 학생 수": round(resource["class_size_after"], 1),
                "학급당 학생 수 변화": round(resource["class_size_delta"], 1),
                "통합 후 학생/교원": round(resource["students_per_teacher_after"], 1),
                "통합 후 학생/교실": round(resource["students_per_classroom_after"], 1),
                "28명 참고선 초과": "예" if resource["overcrowded_28_after"] else "아니오",
            }
        )
    return pd.DataFrame(rows)


data_dir = os.getenv("DACOS_DATA_DIR", str(PROCESSED_DIR))
try:
    bundle = get_bundle(data_dir)
except Exception as error:
    st.error("배포용 데이터가 없습니다. `python -m scripts.build_assets`를 먼저 실행하세요.")
    st.exception(error)
    st.stop()

st.title("학교 통합, 교육여건은 어떻게 달라질까?")
st.caption("부산 소규모 공립초등학교 A를 선택하고, 학교점 직선거리 3km 이내 B와의 가상 통합 결과를 비교합니다.")

counts = bundle.manifest["row_counts"]
overview_cols = st.columns(4)
overview_cols[0].metric("부산 공립 본교", f"{counts['busan_public_elementary_main']:,}개교")
overview_cols[1].metric("소규모 분석대상", f"{counts['small_public_schools']:,}개교")
overview_cols[2].metric("GIS 분석 가능", f"{counts['gis_usable_small_schools']:,}개교")
overview_cols[3].metric("3km 후보 시나리오", f"{counts['candidate_pairs_3km']:,}건")

with st.expander("부산 전체 현황 EDA", expanded=False):
    chart_left, chart_right = st.columns(2)
    with chart_left:
        grade = bundle.grade_summary.sort_values("학년")
        grade_figure = go.Figure(
            go.Bar(
                x=[f"{int(value)}학년" for value in grade["학년"]],
                y=grade["학생수_20250401"],
                marker_color="#3569B0",
                text=[f"{int(value):,}" for value in grade["학생수_20250401"]],
                textposition="outside",
            )
        )
        grade_figure.update_layout(
            title="학년별 학생 수",
            yaxis_title="명",
            showlegend=False,
            height=350,
            margin=dict(l=20, r=20, t=55, b=20),
        )
        st.plotly_chart(grade_figure, width="stretch")
    with chart_right:
        histogram = go.Figure(go.Histogram(x=bundle.master[STUDENTS], nbinsx=30, marker_color="#6AAE75"))
        histogram.update_layout(
            title="학교별 학생 수 분포",
            xaxis_title="학생 수(명)",
            yaxis_title="학교 수",
            showlegend=False,
            height=350,
            margin=dict(l=20, r=20, t=55, b=20),
        )
        st.plotly_chart(histogram, width="stretch")

    district = bundle.district_summary.sort_values("소규모학교수", ascending=False)
    district_figure = go.Figure(
        [
            go.Bar(x=district[DISTRICT], y=district["소규모학교수"], name="소규모학교", marker_color="#E76F51"),
            go.Bar(
                x=district[DISTRICT],
                y=district["전체학교수"] - district["소규모학교수"],
                name="그 외 공립학교",
                marker_color="#B9C7D8",
            ),
        ]
    )
    district_figure.update_layout(
        title="행정구별 공립초등학교",
        yaxis_title="학교 수",
        barmode="stack",
        height=380,
        margin=dict(l=20, r=20, t=55, b=20),
    )
    st.plotly_chart(district_figure, width="stretch")
    st.caption("학년별 학생 수는 2025년 4월 1일 반별 자료, 학교별 규모와 소규모 분류는 2025년 10월 1일 자료를 사용합니다.")

st.divider()
st.subheader("1. 소규모학교와 후보학교 선택")
small_options = bundle.candidate_counts.sort_values(["후보존재", "소규모학교명"], ascending=[False, True]).copy()
small_label = {
    row[PAIR_A_CODE]: f"{row['소규모학교명']} · {row['소규모학교_행정구']} · 후보 {row['후보학교수_3km내']}개"
    for _, row in small_options.iterrows()
}
preferred_a = "213021106"
default_a = list(small_label).index(preferred_a) if preferred_a in small_label else 0
a_code = st.selectbox("통합 전 소규모학교 A", options=list(small_label), index=default_a, format_func=small_label.get)
a_school = bundle.master.set_index(KEDI).loc[a_code]
a_cols = st.columns(4)
a_cols[0].metric("A 학생 수", f"{int(a_school[STUDENTS]):,}명")
a_cols[1].metric("행정구", str(a_school[DISTRICT]))
a_cols[2].metric("현재 학급당 학생 수", format_number(a_school["학급당학생수_20251001_계산"], 1, "명"))
a_cols[3].metric("현재 학생/교원", format_number(a_school["교원1인당학생수_20251001_계산"], 1, "명"))

a_pairs = bundle.candidate_pairs.loc[bundle.candidate_pairs[PAIR_A_CODE].eq(a_code)].copy()
if a_pairs.empty:
    st.warning("이 학교는 학교점 직선거리 3km 이내 후보학교가 없습니다. 후보 반경을 임의로 넓히지 않고 빈 결과로 표시합니다.")
    st.stop()

b_label = {
    row[PAIR_B_CODE]: f"{row['후보학교명']} · {row['학교간직선거리_km']:.2f}km · 현재 {row['후보학교_학급당학생수_20251001']:.1f}명/학급"
    for _, row in a_pairs.sort_values("학교간직선거리_km").iterrows()
}
b_code = st.selectbox("통합 후 수용학교 B", options=list(b_label), format_func=b_label.get)

scenario, grid = run_scenario(
    bundle.master,
    bundle.candidate_pairs,
    bundle.school_points,
    bundle.catchments,
    a_code,
    b_code,
)
resource = scenario["resource"]
access = scenario["accessibility"]

st.subheader(f"2. {resource['a_name']} → {resource['b_name']} 시뮬레이션")
hero = st.columns(4)
hero[0].metric("이동 학생", f"{resource['moving_students']:,}명")
hero[1].metric("학교간 직선거리", f"{scenario['pair']['distance_km']:.2f}km")
hero[2].metric("평균 추가 접근거리", f"{access['added_mean_km']:+.2f}km")
hero[3].metric("접근성 악화 격자", f"{access['worsened_pct']:.1f}%")

resource_tab, access_tab, compare_tab, data_tab = st.tabs(
    ["학교 안 교육자원", "학교 밖 교육접근성", "후보 비교", "데이터·가정"]
)
with resource_tab:
    left, right = st.columns([1.25, 1])
    with left:
        st.dataframe(resource_comparison_table(resource), hide_index=True, width="stretch")
        st.caption("양수는 학생 밀도·부담 증가를, 학생 1인당 교지면적의 음수는 이용 가능 면적 감소를 뜻합니다.")
    with right:
        st.plotly_chart(
            class_size_figure(resource["class_size_before"], resource["class_size_after"]),
            width="stretch",
        )
    if resource["overcrowded_28_after"] and not resource["overcrowded_28_before"]:
        st.warning("이 시나리오에서는 통합 후 학급당 학생 수가 28명 참고선을 새로 넘습니다.")

with access_tab:
    access_cols = st.columns(4)
    access_cols[0].metric("현재 평균", f"{access['current_mean_km']:.2f}km")
    access_cols[1].metric("통합 후 평균", f"{access['after_mean_km']:.2f}km")
    access_cols[2].metric("추가거리 중앙값", f"{access['added_median_km']:+.2f}km")
    access_cols[3].metric("추가거리 최댓값", f"{access['added_max_km']:+.2f}km")
    st.plotly_chart(accessibility_figure(bundle, grid, a_code, b_code), width="stretch")
    st.caption(f"격자 {access['grid_point_count']}개 · {access['assumption']}")

with compare_tab:
    st.dataframe(scenario_comparison(bundle, a_code, a_pairs), hide_index=True, width="stretch")
    st.caption("후보 비교표는 단일 종합점수나 순위를 만들지 않습니다. 접근성 상세치는 위에서 선택한 한 시나리오에 대해 계산합니다.")

with data_tab:
    st.markdown(
        """
        **분석 기준**

        - 학교 모집단: 2025년 10월 1일 부산 공립 초등학교 운영 본교
        - 학생·학급·교원: 2025년 10월 1일
        - 학년별 학생·교실·교지: 2025년 4월 1일
        - 통학구역·학교위치: 2026년 3월 20일 공개자료를 2025년 학교 마스터에 조인
        - 후보: GIS 사용 가능한 소규모학교에서 학교점 직선거리 3km 이하

        **해석 한계**

        이 도구는 실제 통폐합 여부를 결정하거나 추천하지 않습니다. 현재 학급·교원·시설이 유지된다는 고정자원 가정과 직선거리 기반 접근성 변화를 보여주는 What-if 시뮬레이션입니다. 실제 결정에는 통학수단, 도로망, 통학버스, 교원 재배치, 시설 확충, 학생·학부모·지역사회 의견이 추가로 필요합니다.
        """
    )
    st.dataframe(bundle.quality_report, hide_index=True, width="stretch")
    if not bundle.excluded_schools.empty:
        st.caption("GIS 제외 학교")
        st.dataframe(bundle.excluded_schools, hide_index=True, width="stretch")

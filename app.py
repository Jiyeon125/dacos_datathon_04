from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.candidate_generator import PAIR_A_CODE, PAIR_B_CODE
from src.config import PROCESSED_DIR
from src.data_loader import load_bundle
from src.resource_benchmark import build_resource_scenario_table, comparative_resource_profile
from src.resource_simulator import resource_comparison_table
from src.scenario_engine import run_scenario
from src.schema import CLASS_SIZE, DISTRICT, KEDI, SCHOOL_NAME, SMALL_FLAG, STUDENTS


st.set_page_config(page_title="학교 통합 교육여건 시뮬레이터", page_icon="🏫", layout="wide")


@st.cache_resource(show_spinner=False)
def get_bundle(data_dir: str):
    return load_bundle(Path(data_dir))


@st.cache_data(show_spinner=False)
def get_resource_scenarios(master: pd.DataFrame, candidate_pairs: pd.DataFrame) -> pd.DataFrame:
    return build_resource_scenario_table(master, candidate_pairs)


def format_number(value, digits: int = 1, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "자료 없음"
    return f"{float(value):,.{digits}f}{suffix}"


def reset_receiver_selection() -> None:
    st.session_state.pop("b_select", None)
    st.session_state.pop("pending_b", None)


def resource_change_figure(resource: dict) -> go.Figure:
    metrics = [
        ("학생 수", "students_before", "students_after", "명"),
        ("학급당 학생 수", "class_size_before", "class_size_after", "명"),
        ("교원 1인당 학생 수", "students_per_teacher_before", "students_per_teacher_after", "명"),
        ("학생/교실", "students_per_classroom_before", "students_per_classroom_after", "명"),
        ("학생 1인당 교지면적", "land_per_student_before", "land_per_student_after", "㎡"),
    ]
    figure = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[metric[0] for metric in metrics],
        horizontal_spacing=0.12,
        vertical_spacing=0.20,
    )
    for index, (label, before_key, after_key, unit) in enumerate(metrics):
        row, column = divmod(index, 2)
        row, column = row + 1, column + 1
        before = resource[before_key]
        after = resource[after_key]
        if before is None or after is None or pd.isna(before) or pd.isna(after):
            continue
        before, after = float(before), float(after)
        range_values = [before, after]
        if label == "학급당 학생 수":
            range_values.append(28.0)
        span = max(range_values) - min(range_values)
        padding = max(span * 0.18, max(abs(value) for value in range_values) * 0.06, 0.8)
        axis_low = max(0, min(range_values) - padding)
        axis_high = max(range_values) + padding

        figure.add_trace(
            go.Scatter(
                x=[before, after],
                y=[0, 0],
                mode="lines",
                line=dict(color="#8A8F98", width=4),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=[before],
                y=[0],
                mode="markers+text",
                marker=dict(color="#4878CF", size=14),
                text=[f"전 {before:,.1f}{unit}"],
                textposition="top center",
                hovertemplate=f"통합 전: %{{x:,.2f}}{unit}<extra>{label}</extra>",
                name="통합 전",
                legendgroup="before",
                showlegend=index == 0,
            ),
            row=row,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=[after],
                y=[0],
                mode="markers+text",
                marker=dict(color="#F28E2B", size=14),
                text=[f"후 {after:,.1f}{unit}"],
                textposition="bottom center",
                hovertemplate=f"통합 후: %{{x:,.2f}}{unit}<extra>{label}</extra>",
                name="통합 후",
                legendgroup="after",
                showlegend=index == 0,
            ),
            row=row,
            col=column,
        )
        figure.update_xaxes(
            range=[axis_low, axis_high],
            title_text=unit,
            showgrid=False,
            zeroline=False,
            row=row,
            col=column,
        )
        figure.update_yaxes(visible=False, range=[-0.5, 0.5], row=row, col=column)
        if label == "학급당 학생 수":
            figure.add_vline(
                x=28,
                line_dash="dash",
                line_color="#C43C39",
                annotation_text="28명 참고선",
                annotation_position="top right",
                row=row,
                col=column,
            )

    figure.update_layout(
        height=650,
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=25, r=25, t=75, b=25),
        hovermode="closest",
    )
    return figure


def resource_radar_figure(profile: pd.DataFrame, title: str, scope_size: int, selected_name: str) -> go.Figure:
    plot = profile.dropna(subset=["percentile"]).copy()
    theta = plot["axis"].tolist()
    percentiles = plot["percentile"].tolist()
    custom = plot[["raw_label", "raw_value", "unit", "valid_n"]].values.tolist()
    if theta:
        theta.append(theta[0])
        percentiles.append(percentiles[0])
        custom.append(custom[0])

    figure = go.Figure()
    figure.add_trace(
        go.Scatterpolar(
            r=[50] * len(theta),
            theta=theta,
            mode="lines",
            line=dict(color="#8A8F98", width=2, dash="dot"),
            hoverinfo="skip",
            name="비교집단 중앙(50)",
        )
    )
    figure.add_trace(
        go.Scatterpolar(
            r=percentiles,
            theta=theta,
            customdata=custom,
            mode="lines+markers",
            fill="toself",
            fillcolor="rgba(242, 142, 43, 0.22)",
            line=dict(color="#F28E2B", width=3),
            marker=dict(color="#F28E2B", size=8),
            hovertemplate=(
                "<b>%{theta}</b><br>"
                "%{customdata[0]}: %{customdata[1]:.1f}%{customdata[2]}<br>"
                "유리한 방향 백분위: %{r:.0f}<br>"
                "유효 시나리오: %{customdata[3]}개<extra></extra>"
            ),
            name=selected_name,
        )
    )
    figure.update_layout(
        title=dict(text=f"{title}<br><sup>{scope_size:,}개 시나리오 기준</sup>", x=0.02),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75, 100], ticksuffix="", angle=90),
        ),
        legend=dict(orientation="h", y=-0.12, x=0),
        margin=dict(l=35, r=35, t=75, b=55),
        height=430,
    )
    return figure


def _candidate_hover(frame: pd.DataFrame) -> list[str]:
    return [
        (
            f"<b>{row['후보학교명']}</b><br>"
            f"{row['후보학교_행정구']} · {row['학교간직선거리_km']:.2f}km<br>"
            f"학생 {int(row['후보학교_학생수_20251001']):,}명 · "
            f"학급당 {row['후보학교_학급당학생수_20251001']:.1f}명"
        )
        for _, row in frame.iterrows()
    ]


def _map_zoom(distance_km: float | None) -> float:
    if distance_km is None:
        return 12.2
    if distance_km < 0.5:
        return 14.2
    if distance_km < 1.0:
        return 13.7
    if distance_km < 2.0:
        return 13.1
    return 12.6


def candidate_map_figure(bundle, a_code: str, a_pairs: pd.DataFrame, b_code: str | None) -> go.Figure:
    points_metric = bundle.school_points.set_index(KEDI)
    points_wgs84 = bundle.school_points.to_crs(4326).set_index(KEDI)
    a_metric = points_metric.loc[a_code].geometry
    a_point = points_wgs84.loc[a_code].geometry
    if isinstance(a_metric, pd.Series):
        a_metric = a_metric.iloc[0]
    if isinstance(a_point, pd.Series):
        a_point = a_point.iloc[0]

    circle = gpd.GeoSeries([a_metric.buffer(3000)], crs=bundle.school_points.crs).to_crs(4326).iloc[0]
    circle_lon, circle_lat = circle.exterior.xy
    figure = go.Figure()
    figure.add_trace(
        go.Scattermap(
            lon=list(circle_lon),
            lat=list(circle_lat),
            mode="lines",
            fill="toself",
            fillcolor="rgba(214, 39, 40, 0.06)",
            line=dict(color="rgba(214, 39, 40, 0.75)", width=2),
            name="3km 탐색범위",
            hoverinfo="skip",
        )
    )

    candidate_points = points_wgs84.reset_index()[[KEDI, "geometry"]]
    plotted = a_pairs.merge(candidate_points, left_on=PAIR_B_CODE, right_on=KEDI, how="left", validate="many_to_one")
    plotted["후보유형"] = plotted["후보학교_소규모여부_정책2026"].map({True: "소규모 후보", False: "일반 후보"})
    not_selected = plotted.loc[plotted[PAIR_B_CODE].ne(b_code)].copy()
    styles = {
        "일반 후보": ("#2A6FBB", 12),
        "소규모 후보": ("#F28E2B", 12),
    }
    for candidate_type, (color, size) in styles.items():
        subset = not_selected.loc[not_selected["후보유형"].eq(candidate_type)]
        if subset.empty:
            continue
        figure.add_trace(
            go.Scattermap(
                lon=subset["geometry"].map(lambda geometry: geometry.x),
                lat=subset["geometry"].map(lambda geometry: geometry.y),
                mode="markers",
                marker=dict(size=size, color=color, opacity=0.88),
                text=_candidate_hover(subset),
                customdata=[[str(code)] for code in subset[PAIR_B_CODE]],
                hovertemplate="%{text}<extra>%{fullData.name}</extra>",
                name=candidate_type,
            )
        )

    selected_distance = None
    center_lon, center_lat = a_point.x, a_point.y
    if b_code is not None:
        selected = plotted.loc[plotted[PAIR_B_CODE].eq(b_code)]
        if not selected.empty:
            b_row = selected.iloc[0]
            b_point = b_row.geometry
            selected_distance = float(b_row["학교간직선거리_km"])
            center_lon = (a_point.x + b_point.x) / 2
            center_lat = (a_point.y + b_point.y) / 2
            figure.add_trace(
                go.Scattermap(
                    lon=[b_point.x],
                    lat=[b_point.y],
                    mode="markers+text",
                    marker=dict(size=23, color="#1B4965", opacity=1),
                    text=[str(b_row["후보학교명"])],
                    textposition="top right",
                    customdata=[[str(b_code)]],
                    hovertext=_candidate_hover(selected),
                    hovertemplate="%{hovertext}<extra>선택한 수용학교</extra>",
                    name="선택한 수용학교",
                )
            )

    a_name = bundle.master.set_index(KEDI).loc[a_code, SCHOOL_NAME]
    figure.add_trace(
        go.Scattermap(
            lon=[a_point.x],
            lat=[a_point.y],
            mode="markers+text",
            marker=dict(size=25, color="#D62728", opacity=1),
            text=[str(a_name)],
            textposition="top left",
            customdata=[[str(a_code)]],
            hovertext=[f"<b>{a_name}</b><br>통합 대상으로 가정한 학교"],
            hovertemplate="%{hovertext}<extra>통합 대상학교</extra>",
            name="통합 대상학교",
        )
    )
    figure.update_layout(
        map=dict(style="carto-positron", center=dict(lon=center_lon, lat=center_lat), zoom=_map_zoom(selected_distance)),
        height=640,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.01, x=0),
        clickmode="event+select",
        uirevision=f"{a_code}-{b_code or 'none'}",
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
                name="통합 대상학교 통학구역",
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
    names = bundle.master.set_index(KEDI)[SCHOOL_NAME]
    a_name, b_name = names.loc[a_code], names.loc[b_code]
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
            marker=dict(size=15, color=["#D62728", "#1B4965"], symbol=["x", "star"]),
            text=[f"현재 {a_name}", f"수용 {b_name}"],
            textposition=["top left", "top right"],
            name="학교",
        )
    )
    figure.update_layout(
        title="통합 대상학교 통학구역의 접근거리 변화",
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=10, r=10, t=70, b=10),
        height=500,
    )
    return figure


def scenario_comparison(resource_scenarios: pd.DataFrame, a_code: str, b_code: str | None) -> pd.DataFrame:
    comparison = resource_scenarios.loc[resource_scenarios[PAIR_A_CODE].eq(a_code)].copy()
    comparison["선택"] = comparison[PAIR_B_CODE].map(lambda code: "●" if code == b_code else "")
    comparison["28명 이상"] = comparison["class_size_after"].ge(28).map({True: "예", False: "아니오"})
    comparison["선택순서"] = comparison[PAIR_B_CODE].ne(b_code).astype(int) if b_code is not None else 0
    comparison = comparison.sort_values(["선택순서", "학교간직선거리_km", "후보학교명"])
    comparison = comparison.rename(
        columns={
            "후보학교명": "후보 학교",
            "학교간직선거리_km": "거리(km)",
            "class_size_after": "통합 후 학급당 학생",
            "students_per_teacher_after": "통합 후 학생/교원",
            "students_per_classroom_after": "통합 후 학생/교실",
            "land_per_student_after": "통합 후 교지㎡/학생",
        }
    )
    columns = [
        "선택",
        "후보 학교",
        "거리(km)",
        "통합 후 학급당 학생",
        "통합 후 학생/교원",
        "통합 후 학생/교실",
        "통합 후 교지㎡/학생",
        "28명 이상",
    ]
    for column in columns[2:-1]:
        comparison[column] = comparison[column].round(2)
    return comparison[columns].reset_index(drop=True)


def render_busan_eda(bundle) -> None:
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
        grade_figure.update_layout(title="학년별 학생 수", yaxis_title="명", showlegend=False, height=350)
        st.plotly_chart(grade_figure, width="stretch")
    with chart_right:
        histogram = go.Figure(go.Histogram(x=bundle.master[STUDENTS], nbinsx=30, marker_color="#6AAE75"))
        histogram.update_layout(title="학교별 학생 수 분포", xaxis_title="학생 수(명)", yaxis_title="학교 수", showlegend=False, height=350)
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
    district_figure.update_layout(title="행정구별 공립초등학교", yaxis_title="학교 수", barmode="stack", height=380)
    st.plotly_chart(district_figure, width="stretch")
    st.caption("학년별 학생 수는 2025년 4월 1일, 학교규모와 소규모 분류는 2025년 10월 1일 기준입니다.")


data_dir = os.getenv("DACOS_DATA_DIR", str(PROCESSED_DIR))
try:
    bundle = get_bundle(data_dir)
except Exception as error:
    st.error("배포용 데이터가 없습니다. `python -m scripts.build_assets`를 먼저 실행하세요.")
    st.exception(error)
    st.stop()

resource_scenarios = get_resource_scenarios(bundle.master, bundle.candidate_pairs)

counts = bundle.manifest["row_counts"]
st.title("학교 통합, 교육여건은 어떻게 달라질까?")
st.caption(
    f"부산 공립 본교 {counts['busan_public_elementary_main']}개교 · "
    f"소규모 분석대상 {counts['small_public_schools']}개교 · "
    f"3km 후보 시나리오 {counts['candidate_pairs_3km']:,}건"
)

gis_codes = set(bundle.school_points[KEDI])
small_options = bundle.small_schools[[KEDI, SCHOOL_NAME, DISTRICT, STUDENTS]].copy()
small_options = small_options.merge(
    bundle.candidate_counts[[PAIR_A_CODE, "후보학교수_3km내"]],
    left_on=KEDI,
    right_on=PAIR_A_CODE,
    how="left",
)
small_options["후보학교수_3km내"] = small_options["후보학교수_3km내"].fillna(0).astype(int)
small_options["GIS사용가능"] = small_options[KEDI].isin(gis_codes)
small_options = small_options.sort_values(SCHOOL_NAME)
small_label = {
    row[KEDI]: (
        f"{row[SCHOOL_NAME]} · {row[DISTRICT]} · 학생 {int(row[STUDENTS])}명"
        + (f" · 후보 {row['후보학교수_3km내']}개" if row["GIS사용가능"] else " · GIS 제외")
    )
    for _, row in small_options.iterrows()
}

preferred_a = "213021106"
default_a = list(small_label).index(preferred_a) if preferred_a in small_label else 0
control_a, control_b = st.columns(2)
with control_a:
    a_code = st.selectbox(
        "통합 대상으로 가정할 소규모학교",
        options=list(small_label),
        index=default_a,
        format_func=small_label.get,
        key="a_select",
        on_change=reset_receiver_selection,
    )

a_pairs = bundle.candidate_pairs.loc[bundle.candidate_pairs[PAIR_A_CODE].eq(a_code)].sort_values("학교간직선거리_km").copy()
candidate_codes = a_pairs[PAIR_B_CODE].astype(str).tolist()
pending_b = st.session_state.pop("pending_b", None)
if pending_b in candidate_codes:
    st.session_state["b_select"] = pending_b
if st.session_state.get("b_select") not in candidate_codes:
    st.session_state["b_select"] = None

b_label = {
    row[PAIR_B_CODE]: (
        f"{row['후보학교명']} · {row['학교간직선거리_km']:.2f}km · "
        f"현재 {row['후보학교_학급당학생수_20251001']:.1f}명/학급"
    )
    for _, row in a_pairs.iterrows()
}
with control_b:
    if candidate_codes:
        b_code = st.selectbox(
            "통합 후 학생을 받을 수용학교",
            options=[None, *candidate_codes],
            format_func=lambda code: "지도 또는 목록에서 후보를 선택하세요" if code is None else b_label[code],
            key="b_select",
        )
    else:
        st.selectbox("통합 후 학생을 받을 수용학교", options=["선택 가능한 후보가 없습니다"], disabled=True)
        b_code = None

a_school = bundle.master.set_index(KEDI).loc[a_code]
st.caption(
    f"통합 대상학교 현재 학생 {int(a_school[STUDENTS]):,}명 · "
    f"학급당 {a_school[CLASS_SIZE]:.1f}명 · "
    f"교원 1인당 {a_school['교원1인당학생수_20251001_계산']:.1f}명 · "
    f"3km 후보 {len(a_pairs)}개"
)

if a_code not in gis_codes:
    excluded = bundle.excluded_schools.loc[bundle.excluded_schools[KEDI].eq(a_code)]
    reason = excluded["제외사유"].iloc[0] if not excluded.empty else "학교 위치 또는 통학구역 조인 불가"
    st.warning(f"{a_school[SCHOOL_NAME]}은(는) 2025년 마스터에는 있으나 현재 GIS 분석에서는 제외됩니다. {reason}")
elif a_pairs.empty:
    map_event = st.plotly_chart(
        candidate_map_figure(bundle, a_code, a_pairs, None),
        width="stretch",
        key=f"candidate_map_{a_code}_none",
        on_select="rerun",
        selection_mode="points",
        config={"scrollZoom": True, "displaylogo": False},
    )
    st.info("학교점 직선거리 3km 이내에 선택 가능한 수용학교가 없습니다.")
else:
    map_event = st.plotly_chart(
        candidate_map_figure(bundle, a_code, a_pairs, b_code),
        width="stretch",
        key=f"candidate_map_{a_code}_{b_code or 'none'}",
        on_select="rerun",
        selection_mode="points",
        config={"scrollZoom": True, "displaylogo": False},
    )
    selected_points = getattr(getattr(map_event, "selection", None), "points", [])
    if selected_points:
        custom_data = selected_points[-1].get("customdata")
        clicked_code = str(custom_data[0] if isinstance(custom_data, (list, tuple)) else custom_data)
        if clicked_code in candidate_codes and clicked_code != b_code:
            st.session_state["pending_b"] = clicked_code
            st.rerun()
    st.caption("지도 점을 클릭하거나 위 수용학교 목록에서 후보를 선택하세요. 주황색은 소규모 후보, 파란색은 그 외 후보입니다.")

if b_code is None:
    st.info("학생을 받을 수용학교를 선택하면 교육자원과 교육접근성 변화가 아래에 표시됩니다.")
else:
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
    st.divider()
    st.subheader(f"{resource['a_name']} → {resource['b_name']}")
    hero = st.columns(4)
    hero[0].metric("이동 학생", f"{resource['moving_students']:,}명")
    hero[1].metric("학교간 직선거리", f"{scenario['pair']['distance_km']:.2f}km")
    hero[2].metric("평균 추가 접근거리", f"{access['added_mean_km']:+.2f}km")
    hero[3].metric("접근성 악화 격자", f"{access['worsened_pct']:.1f}%")

    resource_tab, access_tab = st.tabs(["학교 안 교육자원", "학교 밖 교육접근성"])
    with resource_tab:
        st.markdown("#### 선택 시나리오의 교육자원 변화")
        st.dataframe(resource_comparison_table(resource), hide_index=True, width="stretch")
        st.plotly_chart(resource_change_figure(resource), width="stretch")
        st.caption(
            "각 패널은 서로 다른 단위와 범위를 유지한 통합 전→후 원수치입니다. "
            "고정자원 가정에 따라 통합 대상학교 학생만 선택한 수용학교로 이동하고, "
            "수용학교의 학급·교원·교실·교지는 유지합니다."
        )
        if resource["overcrowded_28_after"] and not resource["overcrowded_28_before"]:
            st.warning("통합 후 학급당 학생 수가 28명 참고선을 새로 넘습니다.")

        st.markdown("#### 선택한 수용학교의 교육자원 여유는 어느 정도일까?")
        same_a_profile, same_a_size = comparative_resource_profile(
            resource_scenarios,
            a_code,
            b_code,
            same_a_only=True,
        )
        all_profile, all_size = comparative_resource_profile(
            resource_scenarios,
            a_code,
            b_code,
            same_a_only=False,
        )
        radar_left, radar_right = st.columns(2)
        with radar_left:
            st.plotly_chart(
                resource_radar_figure(
                    same_a_profile,
                    "통합 대상학교 주변 3km 수용 후보와 비교",
                    same_a_size,
                    resource["b_name"],
                ),
                width="stretch",
            )
        with radar_right:
            st.plotly_chart(
                resource_radar_figure(
                    all_profile,
                    "부산 전체 통합 시나리오와 비교",
                    all_size,
                    resource["b_name"],
                ),
                width="stretch",
            )
        st.caption(
            "읽는 법: 주황색 선이 바깥쪽일수록 선택한 수용학교가 해당 교육자원에서 비교 대상보다 상대적으로 "
            "여유가 있습니다. 점선 50은 비교 대상의 중간 위치입니다. 네 축을 합산한 종합점수나 추천 순위는 아닙니다."
        )
    with access_tab:
        access_cols = st.columns(4)
        access_cols[0].metric("현재 평균", f"{access['current_mean_km']:.2f}km")
        access_cols[1].metric("통합 후 평균", f"{access['after_mean_km']:.2f}km")
        access_cols[2].metric("추가거리 중앙값", f"{access['added_median_km']:+.2f}km")
        access_cols[3].metric("추가거리 최댓값", f"{access['added_max_km']:+.2f}km")
        st.plotly_chart(accessibility_figure(bundle, grid, a_code, b_code), width="stretch")
        st.caption(f"격자 {access['grid_point_count']}개 · {access['assumption']}")

if not a_pairs.empty:
    with st.expander(f"{a_school[SCHOOL_NAME]}의 모든 후보 수치 비교", expanded=False):
        st.dataframe(scenario_comparison(resource_scenarios, a_code, b_code), hide_index=True, width="stretch")
        st.caption("● 표시는 현재 선택한 수용학교입니다. 모든 값은 수용학교의 현재 자원을 고정한 통합 후 수치입니다.")

with st.expander("부산 전체 현황 EDA", expanded=False):
    render_busan_eda(bundle)

with st.expander("데이터 기준과 해석 한계", expanded=False):
    st.markdown(
        """
        - 학교 모집단: 2025년 10월 1일 부산 공립 초등학교 운영 본교
        - 학생·학급·교원: 2025년 10월 1일
        - 학년별 학생·교실·교지: 2025년 4월 1일
        - 통학구역·학교위치: 2026년 3월 20일 공개자료를 2025년 학교 마스터에 조인
        - 3km는 정책 판정기준이 아니라 POC의 후보 탐색범위

        이 도구는 실제 통폐합 여부를 결정하거나 추천하지 않습니다. 접근성은 학생 거주분포·도로망·통학수단을 반영하지 않은 250m 공간격자 기반 직선거리이며, 실제 결정에는 통학버스, 교원 재배치, 시설 확충, 학생·학부모·지역사회 의견을 추가로 검토해야 합니다.
        """
    )

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import geopandas as gpd
import pandas as pd

from src.schema import DISTRICT, KEDI, SCHOOL_NAME, normalize_kedi


@dataclass
class GisBuildResult:
    school_points: gpd.GeoDataFrame
    catchments: gpd.GeoDataFrame
    excluded_schools: pd.DataFrame
    point_zone_qa: pd.DataFrame


def normalize_school_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFC", str(value))
    return re.sub(r"[\s·ㆍ]", "", text)


def catchment_name_to_school(value: object) -> str:
    text = normalize_school_name(value)
    text = re.sub(r"(공동)?(통학구역|학구)$", "", text)
    if text.endswith("초"):
        text = f"{text}등학교"
    return text


def school_short_name(value: object) -> str:
    return normalize_school_name(value).replace("등학교", "")


def _read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _extract_district(address: pd.Series) -> pd.Series:
    return address.astype("string").str.extract(r"부산광역시\s+([^\s]+)", expand=False)


def _choose_nearest_school(zone_row, candidate_points: gpd.GeoDataFrame) -> str | None:
    if candidate_points.empty:
        return None
    distances = candidate_points.geometry.distance(zone_row.geometry.representative_point())
    return str(candidate_points.loc[distances.idxmin(), KEDI])


def build_gis_assets(
    master: pd.DataFrame,
    catchment_shapes: gpd.GeoDataFrame,
    catchment_meta: pd.DataFrame,
    school_locations: pd.DataFrame,
) -> GisBuildResult:
    """2025년 학교 마스터를 기준으로 위치와 공식 통학구역을 KEDI 단위로 결합한다."""
    shapes = catchment_shapes.copy()
    if shapes.crs is None:
        raise ValueError("통학구역 SHP에 좌표계가 없습니다. .prj 파일을 확인하세요.")
    if not shapes.crs.is_projected:
        shapes = shapes.to_crs(5186)

    meta = catchment_meta.copy()
    busan_meta = meta.loc[meta["시도교육청명"].eq("부산광역시교육청")].copy()
    busan_sd_codes = set(busan_meta["시도코드"].astype("string").str.replace(r"\.0$", "", regex=True))
    shapes["SD_CD"] = shapes["SD_CD"].astype("string").str.replace(r"\.0$", "", regex=True)
    shapes = shapes.loc[shapes["SD_CD"].isin(busan_sd_codes)].copy()

    locations = school_locations.loc[
        school_locations["시도교육청명"].eq("부산광역시교육청")
        & school_locations["학교급구분"].eq("초등학교")
    ].copy()
    locations["학교명키"] = locations["학교명"].map(normalize_school_name)
    locations["행정구키"] = _extract_district(locations["소재지도로명주소"])
    locations["위도"] = pd.to_numeric(locations["위도"], errors="coerce")
    locations["경도"] = pd.to_numeric(locations["경도"], errors="coerce")
    locations = locations.dropna(subset=["위도", "경도"])

    master_key = master[[KEDI, SCHOOL_NAME, DISTRICT]].copy()
    master_key[KEDI] = normalize_kedi(master_key[KEDI])
    master_key["학교명키"] = master_key[SCHOOL_NAME].map(normalize_school_name)
    master_key["행정구키"] = master_key[DISTRICT].astype("string")
    matched = master_key.merge(
        locations[["학교명키", "행정구키", "위도", "경도", "소재지도로명주소"]],
        on=["학교명키", "행정구키"],
        how="left",
        validate="one_to_one",
    )
    matched_rows = matched.dropna(subset=["위도", "경도"]).copy()
    school_points = gpd.GeoDataFrame(
        matched_rows,
        geometry=gpd.points_from_xy(matched_rows["경도"], matched_rows["위도"]),
        crs=4326,
    ).to_crs(shapes.crs)
    school_points = school_points.merge(
        master.drop(columns=[SCHOOL_NAME, DISTRICT], errors="ignore"),
        on=KEDI,
        how="left",
        validate="one_to_one",
    )

    excluded = matched.loc[matched["위도"].isna(), [KEDI, SCHOOL_NAME, DISTRICT]].copy()
    excluded["제외사유"] = "2026 위치표준데이터에서 학교명+행정구 조인 실패"
    excluded = excluded.reset_index(drop=True)

    shapes["통학구역명"] = shapes["HAKGUDO_NM"].astype("string")
    shapes["통학구역핵심"] = shapes["통학구역명"].map(catchment_name_to_school)
    shapes["통학구역짧은키"] = shapes["통학구역핵심"].map(school_short_name)
    school_points["학교짧은키"] = school_points[SCHOOL_NAME].map(school_short_name)

    assignments: list[dict] = []
    for zone_index, zone in shapes.iterrows():
        zone_key = zone["통학구역짧은키"]
        is_shared = "공동" in normalize_school_name(zone["통학구역명"])
        if is_shared:
            candidate_keys = school_points.loc[
                school_points["학교짧은키"].map(lambda key: bool(key) and key in zone_key),
                "학교짧은키",
            ].unique()
        else:
            candidate_keys = [zone_key]
        for key in candidate_keys:
            candidates = school_points.loc[school_points["학교짧은키"].eq(key)]
            school_code = _choose_nearest_school(zone, candidates)
            if school_code:
                assignments.append({"zone_index": zone_index, KEDI: school_code})

    assignment_frame = pd.DataFrame(assignments).drop_duplicates(["zone_index", KEDI])
    if assignment_frame.empty:
        raise ValueError("통학구역과 학교를 한 건도 연결하지 못했습니다.")
    zone_parts = shapes.merge(assignment_frame, left_index=True, right_on="zone_index", how="inner")
    catchments = zone_parts[[KEDI, "geometry"]].dissolve(by=KEDI, as_index=False)
    catchments[KEDI] = normalize_kedi(catchments[KEDI])
    catchments = catchments.merge(
        school_points[[KEDI, SCHOOL_NAME, DISTRICT]],
        on=KEDI,
        how="left",
        validate="one_to_one",
    )

    usable_codes = set(school_points[KEDI]) & set(catchments[KEDI])
    school_points = school_points.loc[school_points[KEDI].isin(usable_codes)].copy().reset_index(drop=True)
    catchments = catchments.loc[catchments[KEDI].isin(usable_codes)].copy().reset_index(drop=True)
    point_lookup = school_points.set_index(KEDI).geometry
    qa_rows = []
    for _, row in catchments.iterrows():
        code = row[KEDI]
        point = point_lookup.loc[code]
        distance = float(row.geometry.distance(point))
        qa_rows.append(
            {
                KEDI: code,
                SCHOOL_NAME: row[SCHOOL_NAME],
                "학교점_통학구역포함": bool(row.geometry.covers(point)),
                "학교점_통학구역거리_m": distance,
            }
        )
    qa = pd.DataFrame(qa_rows)
    return GisBuildResult(school_points, catchments, excluded, qa)

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from src.schema import CLASS_SIZE, DISTRICT, KEDI, SCHOOL_NAME, SMALL_FLAG, STUDENTS, normalize_kedi


PAIR_A_CODE = "소규모학교_KEDI"
PAIR_B_CODE = "후보학교_KEDI"


def generate_candidate_pairs(
    school_points: gpd.GeoDataFrame,
    max_distance_km: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GIS 사용 가능한 소규모학교 A마다 직선 3km 이내 모든 B를 후보로 만든다."""
    if school_points.crs is None or not school_points.crs.is_projected:
        raise ValueError("후보 생성에는 미터 단위 투영좌표계가 필요합니다.")
    points = school_points.copy()
    points[KEDI] = normalize_kedi(points[KEDI])
    a_rows = points.loc[points[SMALL_FLAG].astype(bool)].copy()
    pairs: list[dict] = []
    for _, a in a_rows.iterrows():
        a_code = a[KEDI]
        distances_km = points.geometry.distance(a.geometry) / 1000
        nearby = points.loc[(points[KEDI].ne(a_code)) & distances_km.le(max_distance_km)].copy()
        nearby["학교간직선거리_km"] = distances_km.loc[nearby.index]
        nearby = nearby.sort_values(["학교간직선거리_km", SCHOOL_NAME])
        for rank, (_, b) in enumerate(nearby.iterrows(), start=1):
            pairs.append(
                {
                    PAIR_A_CODE: a_code,
                    "소규모학교명": a[SCHOOL_NAME],
                    "소규모학교_행정구": a[DISTRICT],
                    "소규모학교_학생수_20251001": a[STUDENTS],
                    PAIR_B_CODE: b[KEDI],
                    "후보학교명": b[SCHOOL_NAME],
                    "후보학교_행정구": b[DISTRICT],
                    "후보학교_학생수_20251001": b[STUDENTS],
                    "후보학교_학급당학생수_20251001": b[CLASS_SIZE],
                    "후보학교_소규모여부_정책2026": bool(b[SMALL_FLAG]),
                    "학교간직선거리_km": round(float(b["학교간직선거리_km"]), 6),
                    "후보생성기준": f"학교점 간 직선거리 {max_distance_km:g}km 이하",
                    "거리순위_3km내": rank,
                }
            )
    pair_frame = pd.DataFrame(pairs)
    all_small = a_rows[[KEDI, SCHOOL_NAME, DISTRICT]].rename(
        columns={KEDI: PAIR_A_CODE, SCHOOL_NAME: "소규모학교명", DISTRICT: "소규모학교_행정구"}
    )
    counts = pair_frame.groupby(PAIR_A_CODE).size().rename("후보학교수_3km내") if not pair_frame.empty else pd.Series(dtype=int)
    summary = all_small.merge(counts, left_on=PAIR_A_CODE, right_index=True, how="left")
    summary["후보학교수_3km내"] = summary["후보학교수_3km내"].fillna(0).astype(int)
    summary["후보존재"] = summary["후보학교수_3km내"].gt(0)
    return pair_frame, summary.sort_values(["후보존재", "후보학교수_3km내", "소규모학교명"], ascending=[False, False, True])

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BUSAN_SMALL_SCHOOL_THRESHOLD
from src.schema import (
    ADDRESS,
    CLASSES,
    CLASSROOMS,
    CLASS_SIZE,
    DISTRICT,
    FOUNDATION,
    KEDI,
    LAND_AREA,
    POLICY_THRESHOLD,
    REGION_SIZE,
    SCHOOL_NAME,
    SMALL_FLAG,
    STUDENTS,
    STUDENTS_PER_TEACHER,
    TEACHERS,
    normalize_kedi,
)


@dataclass
class EducationBuildResult:
    master: pd.DataFrame
    small_schools: pd.DataFrame
    school_grade: pd.DataFrame
    quality_report: pd.DataFrame
    district_summary: pd.DataFrame
    region_summary: pd.DataFrame
    grade_summary: pd.DataFrame
    operating_count: int


def read_clean_csv(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if KEDI in frame.columns:
        frame[KEDI] = normalize_kedi(frame[KEDI])
    return frame


def _to_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def filter_busan_operating_elementary(data2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """303개 운영 본교를 만든 뒤 공립만 남겨 296개 분석 모집단을 만든다."""
    operating_states = {"기존(원)교", "신설(원)교"}
    base = data2.loc[
        data2["시도"].eq("부산")
        & data2["학교급"].eq("초등학교")
        & data2["본분교"].eq("본교")
        & data2["상태"].isin(operating_states)
    ].copy()
    public = base.loc[base["설립"].eq("공립")].copy()
    return base, public


def _build_data1_grade(data1: pd.DataFrame, school_codes: set[str]) -> tuple[pd.DataFrame, int]:
    rows = data1.loc[
        data1["시도"].eq("부산")
        & data1["학제 대분류"].eq("초등학교")
        & data1["본분교"].eq("본교")
        & data1[KEDI].isin(school_codes)
    ].copy()
    rows = _to_numeric(rows, ["학년", "학급수", "학생수_계"])
    grain = [KEDI, "학과코드", "학과 주/야간", "단식/복식", "일반/특수/순회", "학년", "반"]
    duplicate_count = int(rows.duplicated(grain, keep=False).sum())
    grade_rows = rows.loc[rows["학년"].between(1, 6)].copy()
    grade = grade_rows.groupby([KEDI, "학년"], as_index=False)[["학생수_계", "학급수"]].sum()
    for category, prefix in (("일반", "일반"), ("특수", "특수")):
        category_grade = (
            grade_rows.loc[grade_rows["일반/특수/순회"].eq(category)]
            .groupby([KEDI, "학년"], as_index=False)[["학생수_계", "학급수"]]
            .sum()
            .rename(columns={"학생수_계": f"{prefix}학생수", "학급수": f"{prefix}학급수"})
        )
        grade = grade.merge(category_grade, on=[KEDI, "학년"], how="left", validate="one_to_one")
        grade[[f"{prefix}학생수", f"{prefix}학급수"]] = grade[
            [f"{prefix}학생수", f"{prefix}학급수"]
        ].fillna(0)
    return grade, duplicate_count


def _pivot_grade(grade: pd.DataFrame) -> pd.DataFrame:
    pivots = {
        value: grade.pivot(index=KEDI, columns="학년", values=value)
        for value in ["학생수_계", "학급수", "일반학생수", "일반학급수", "특수학생수", "특수학급수"]
    }
    output = pd.DataFrame(index=pivots["학생수_계"].index)
    output.index.name = KEDI
    for grade_no in range(1, 7):
        output[f"학생수_20250401_{grade_no}학년"] = pivots["학생수_계"].get(grade_no).fillna(0)
        output[f"학급수_20250401_{grade_no}학년"] = pivots["학급수"].get(grade_no).fillna(0)
        for prefix in ("일반", "특수"):
            output[f"{prefix}학생수_20250401_{grade_no}학년"] = pivots[f"{prefix}학생수"].get(grade_no).fillna(0)
            output[f"{prefix}학급수_20250401_{grade_no}학년"] = pivots[f"{prefix}학급수"].get(grade_no).fillna(0)
    for prefix, student_base, class_base in (
        ("", "학생수", "학급수"),
        ("일반", "일반학생수", "일반학급수"),
        ("특수", "특수학생수", "특수학급수"),
    ):
        student_cols = [f"{student_base}_20250401_{grade_no}학년" for grade_no in range(1, 7)]
        class_cols = [f"{class_base}_20250401_{grade_no}학년" for grade_no in range(1, 7)]
        output[f"{student_base}_20250401_반합계"] = output[student_cols].sum(axis=1, min_count=1)
        output[f"{class_base}_20250401_반합계"] = output[class_cols].sum(axis=1, min_count=1)
    return output.reset_index()


def _build_facilities(data3: pd.DataFrame, school_codes: set[str]) -> pd.DataFrame:
    rows = data3.loc[
        data3["시도"].eq("부산")
        & data3["학교급"].eq("초등학교")
        & data3["본분교"].eq("본교")
        & data3[KEDI].isin(school_codes)
    ].copy()
    facility_columns = ["일반 교실", "교과 교실", "특별 교실", "수준별교실", "기타 교실"]
    numeric = ["학생수_총계_계", "편성학급수_계", "교지면적", "학생 1인당 교지면적", *facility_columns]
    rows = _to_numeric(rows, numeric)
    selected = rows[[KEDI, "조사기준일", *numeric]].drop_duplicates(KEDI).copy()
    rename = {
        "조사기준일": "조사기준일_데이터3",
        "학생수_총계_계": "학생수_20250401_학교총계",
        "편성학급수_계": "학급수_20250401_학교총계",
        "일반 교실": "일반교실수_20250401",
        "교과 교실": "교과교실수_20250401",
        "특별 교실": "특별교실수_20250401",
        "수준별교실": "수준별교실수_20250401",
        "기타 교실": "기타교실수_20250401",
        "교지면적": LAND_AREA,
        "학생 1인당 교지면적": "학생1인당교지면적_20250401_공시",
    }
    selected = selected.rename(columns=rename)
    classroom_components = [rename[column] for column in facility_columns]
    selected[CLASSROOMS] = selected[classroom_components].sum(axis=1, min_count=1)
    selected["학생수_일반교실_20250401"] = _safe_ratio(
        selected["학생수_20250401_학교총계"], selected["일반교실수_20250401"]
    )
    selected["학생수_전체교실_20250401"] = _safe_ratio(
        selected["학생수_20250401_학교총계"], selected[CLASSROOMS]
    )
    return selected


def _quality_row(check: str, passed: bool, observed: object, expected: object, note: str = "") -> dict:
    return {
        "검증항목": check,
        "통과여부": bool(passed),
        "관측값": observed,
        "기대값": expected,
        "설명": note,
    }


def build_education_assets(data1: pd.DataFrame, data2: pd.DataFrame, data3: pd.DataFrame) -> EducationBuildResult:
    data1 = data1.copy()
    data2 = data2.copy()
    data3 = data3.copy()
    for frame in (data1, data2, data3):
        frame[KEDI] = normalize_kedi(frame[KEDI])

    operating, public = filter_busan_operating_elementary(data2)
    public = _to_numeric(
        public,
        ["학생수_총계_계", "편성학급수_계", "교원수_총계_계", "학급당 학생수", "교원1인당 학생수"],
    )
    public[POLICY_THRESHOLD] = public["지역규모"].map(BUSAN_SMALL_SCHOOL_THRESHOLD)
    public["학생수기준_충족_정책2026"] = public["학생수_총계_계"].le(public[POLICY_THRESHOLD])
    public["정책적용대상_공립_정책2026"] = public["설립"].eq("공립")
    public[SMALL_FLAG] = public["학생수기준_충족_정책2026"] & public["정책적용대상_공립_정책2026"]

    master_columns = [
        KEDI,
        "학교명",
        "행정구",
        "설립",
        "지역규모",
        "주소",
        "학생수_총계_계",
        "편성학급수_계",
        "교원수_총계_계",
        "학급당 학생수",
        "교원1인당 학생수",
        POLICY_THRESHOLD,
        "학생수기준_충족_정책2026",
        "정책적용대상_공립_정책2026",
        SMALL_FLAG,
    ]
    master = public[master_columns].rename(
        columns={
            "학교명": SCHOOL_NAME,
            "행정구": DISTRICT,
            "설립": FOUNDATION,
            "지역규모": REGION_SIZE,
            "주소": ADDRESS,
            "학생수_총계_계": STUDENTS,
            "편성학급수_계": CLASSES,
            "교원수_총계_계": TEACHERS,
            "학급당 학생수": "학급당학생수_20251001_공시",
            "교원1인당 학생수": "교원1인당학생수_20251001_공시",
        }
    )

    school_codes = set(master[KEDI])
    school_grade, duplicate_count = _build_data1_grade(data1, school_codes)
    master = master.merge(_pivot_grade(school_grade), on=KEDI, how="left", validate="one_to_one")
    facilities = _build_facilities(data3, school_codes)
    master = master.merge(facilities, on=KEDI, how="left", validate="one_to_one")
    master[CLASS_SIZE] = _safe_ratio(master[STUDENTS], master[CLASSES])
    master[STUDENTS_PER_TEACHER] = _safe_ratio(master[STUDENTS], master[TEACHERS])
    master = master.sort_values([DISTRICT, SCHOOL_NAME]).reset_index(drop=True)
    small = master.loc[master[SMALL_FLAG]].copy().reset_index(drop=True)

    matching_totals = (
        master["학생수_20250401_반합계"].fillna(-1).eq(master["학생수_20250401_학교총계"].fillna(-2))
    )
    matching_student_types = master["학생수_20250401_반합계"].eq(
        master["일반학생수_20250401_반합계"] + master["특수학생수_20250401_반합계"]
    )
    matching_class_types = master["학급수_20250401_반합계"].eq(
        master["일반학급수_20250401_반합계"] + master["특수학급수_20250401_반합계"]
    )
    class_formation_columns = [
        f"{prefix}{kind}_20250401_{grade}학년"
        for prefix in ("일반", "특수")
        for kind in ("학생수", "학급수")
        for grade in range(1, 7)
    ]
    ratio_diff = (master[CLASS_SIZE] - master["학급당학생수_20251001_공시"]).abs()
    quality_rows = [
        _quality_row("부산 운영 본교 수", len(operating) == 303, len(operating), 303),
        _quality_row("부산 공립 운영 본교 수", len(master) == 296, len(master), 296),
        _quality_row("공립 마스터 KEDI 유일성", master[KEDI].is_unique, master[KEDI].nunique(), len(master)),
        _quality_row("지역규모 기준 매핑", master[POLICY_THRESHOLD].notna().all(), int(master[POLICY_THRESHOLD].notna().sum()), len(master)),
        _quality_row("소규모학교 수", len(small) == 92, len(small), 92),
        _quality_row("데이터1 원시 grain 중복", duplicate_count == 0, duplicate_count, 0),
        _quality_row("데이터1 학교 조인", master["학생수_20250401_반합계"].notna().all(), int(master["학생수_20250401_반합계"].notna().sum()), len(master)),
        _quality_row("데이터3 학교 조인", master[LAND_AREA].notna().all(), int(master[LAND_AREA].notna().sum()), len(master)),
        _quality_row("4월 학생총계 일치", bool(matching_totals.all()), int(matching_totals.sum()), len(master)),
        _quality_row("4월 일반+특수 학생 합계 일치", bool(matching_student_types.all()), int(matching_student_types.sum()), len(master)),
        _quality_row("4월 일반+특수 학급 합계 일치", bool(matching_class_types.all()), int(matching_class_types.sum()), len(master)),
        _quality_row(
            "학년별 학급 재편성 입력값 완전성",
            bool(master[class_formation_columns].notna().all().all()),
            int(master[class_formation_columns].notna().all(axis=1).sum()),
            len(master),
            "해당 학년 행이 없으면 학생·학급 0으로 처리",
        ),
        _quality_row("10월 학급당학생수 공시 반올림 일치", bool(ratio_diff.le(0.051).all()), float(ratio_diff.max()), "<=0.051"),
    ]
    quality = pd.DataFrame(quality_rows)

    district = (
        master.groupby(DISTRICT, as_index=False)
        .agg(전체학교수=(KEDI, "size"), 소규모학교수=(SMALL_FLAG, "sum"), 학생수합계=(STUDENTS, "sum"))
    )
    district["소규모학교비율_pct"] = district["소규모학교수"] / district["전체학교수"] * 100
    region = (
        master.groupby(REGION_SIZE, as_index=False)
        .agg(전체학교수=(KEDI, "size"), 소규모학교수=(SMALL_FLAG, "sum"), 학생수중앙값=(STUDENTS, "median"))
    )
    region["소규모학교비율_pct"] = region["소규모학교수"] / region["전체학교수"] * 100
    grade_summary = (
        school_grade.groupby("학년", as_index=False)[["학생수_계", "학급수"]]
        .sum()
        .rename(columns={"학생수_계": "학생수_20250401", "학급수": "학급수_20250401"})
    )
    return EducationBuildResult(master, small, school_grade, quality, district, region, grade_summary, len(operating))

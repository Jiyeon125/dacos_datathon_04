from __future__ import annotations

import pandas as pd


KEDI = "학교코드(KEDI)"
SCHOOL_NAME = "학교명_20251001"
DISTRICT = "행정구_20251001"
FOUNDATION = "설립_20251001"
REGION_SIZE = "지역규모_20251001"
ADDRESS = "주소_20251001"
STUDENTS = "학생수_20251001"
CLASSES = "학급수_20251001"
TEACHERS = "교원수_20251001"
CLASS_SIZE = "학급당학생수_20251001_계산"
STUDENTS_PER_TEACHER = "교원1인당학생수_20251001_계산"
CLASSROOMS = "전체교실수_20250401"
GENERAL_CLASSROOMS = "일반교실수_20250401"
LAND_AREA = "교지면적_20250401"
SMALL_FLAG = "주분석대상_소규모공립_정책2026"
POLICY_THRESHOLD = "부산교육청_기준학생수_정책2026"


ALIASES = {
    SCHOOL_NAME: ["학교명", "학교명_20251001"],
    DISTRICT: ["행정구", "행정구_20251001"],
    FOUNDATION: ["설립", "설립_20251001"],
    REGION_SIZE: ["지역규모", "지역규모_20251001"],
    ADDRESS: ["주소", "주소_20251001"],
    STUDENTS: ["학생수_20251001_계", "학생수_20251001"],
    CLASSES: ["편성학급수_20251001", "학급수_20251001"],
    TEACHERS: ["교원수_20251001_계", "교원수_20251001"],
    CLASSROOMS: ["교실수합계_20250401", "전체교실수_20250401"],
}


REQUIRED_MASTER_COLUMNS = [
    KEDI,
    SCHOOL_NAME,
    DISTRICT,
    STUDENTS,
    CLASSES,
    TEACHERS,
    CLASSROOMS,
    GENERAL_CLASSROOMS,
    LAND_AREA,
    SMALL_FLAG,
]


def normalize_kedi(series: pd.Series) -> pd.Series:
    """CSV 숫자 추론으로 생긴 `.0`을 제거하고 KEDI를 문자열로 통일한다."""
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def normalize_master_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """이전 Colab 산출물과 새 Python 산출물의 컬럼명을 공통 스키마로 맞춘다."""
    result = frame.copy()
    for canonical, candidates in ALIASES.items():
        if canonical in result.columns:
            continue
        for candidate in candidates:
            if candidate in result.columns:
                result = result.rename(columns={candidate: canonical})
                break
    if KEDI in result.columns:
        result[KEDI] = normalize_kedi(result[KEDI])
    missing = [column for column in REQUIRED_MASTER_COLUMNS if column not in result.columns]
    if missing:
        raise ValueError(f"학교 마스터 필수 컬럼 누락: {missing}")
    return result

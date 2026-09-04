from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
EDUCATION_RAW_DIR = RAW_DIR / "education"
GIS_RAW_DIR = RAW_DIR / "gis"
PROCESSED_DIR = ROOT / "data" / "processed"

DATA1_PATH = EDUCATION_RAW_DIR / "data1_students_by_class.csv"
DATA2_PATH = EDUCATION_RAW_DIR / "data2_school_resources_20251001.csv"
DATA3_PATH = EDUCATION_RAW_DIR / "data3_school_facilities_20250401.csv"

CATCHMENT_SHP_PATH = GIS_RAW_DIR / "초등학교통학구역.shp"
CATCHMENT_META_PATH = GIS_RAW_DIR / "전국초등학교통학구역표준데이터.csv"
SCHOOL_LOCATION_PATH = GIS_RAW_DIR / "전국초중등학교위치표준데이터.csv"

BUSAN_SMALL_SCHOOL_THRESHOLD = {
    "특별/광역시": 240,
    "시지역": 240,
    "읍지역": 120,
    "면지역": 120,
}

BUSAN_ELEMENTARY_CLASS_FORMATION_2025 = {
    "year": 2025,
    "students_per_class": 25,
    "status": "official_document_derived",
    "label": "2025 부산 초등학교 학생배치지표 25명",
    "source_urls": [
        "https://www.pen.go.kr/upload/dep17/na/bbs_2577/ntt_1094796/doc_03aav3a86=d1vd6=41vf5=b5v39=a91av4f9bvb82d_v7360.pdf",
        "https://www.pen.go.kr/main/na/ntt/selectNttInfo.do?mi=31012&bbsId=2518&nttSn=1123657",
        "https://www.pen.go.kr/main/na/ntt/selectNttInfo.do?mi=31012&bbsId=2518&nttSn=1165624",
    ],
}

EXPECTED_COUNTS = {
    "busan_operating_elementary_main": 303,
    "busan_public_elementary_main": 296,
    "small_public_schools": 92,
    "gis_usable_schools": 294,
    "gis_usable_small_schools": 90,
    "candidate_pairs_3km": 1481,
    "small_schools_with_candidate": 83,
    "small_schools_without_candidate": 7,
}

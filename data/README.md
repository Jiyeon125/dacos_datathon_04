# 데이터 배치

## 원자료: `data/raw/` (Git 제외)

```text
data/raw/
├── education/
│   ├── data1_students_by_class.csv
│   ├── data2_school_resources_20251001.csv
│   └── data3_school_facilities_20250401.csv
└── gis/
    ├── 초등학교통학구역.shp
    ├── 초등학교통학구역.shx
    ├── 초등학교통학구역.dbf
    ├── 초등학교통학구역.prj
    ├── 전국초등학교통학구역표준데이터.csv
    └── 전국초중등학교위치표준데이터.csv
```

## 배포 데이터: `data/processed/` (Git 포함)

`python -m scripts.build_assets`가 생성한다. Streamlit 앱은 이 폴더만 읽으므로 배포 환경에 원자료가 없어도 실행된다.

- `school_master_2025.csv`: 부산 공립 본교 296개교 마스터
- `small_schools_2025.csv`: 부산교육청 2026년 학생수 기준 적용 92개교
- `candidate_pairs_within_3km_2025.csv`: GIS 사용 가능한 소규모학교의 3km 후보쌍
- `school_points_2025.geojson`: 학교 위치 294개교
- `school_catchments_2025.geojson`: KEDI별 통학구역 294개교(배포용 단순화)
- `gis_excluded_schools_2025.csv`: 2025 마스터에는 있으나 GIS 조인이 안 된 학교
- `data_quality_report.csv`: 핵심 품질검증 결과
- `dataset_manifest.json`: 기준일·행수·파일 크기·생성 시각

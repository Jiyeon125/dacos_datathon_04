from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.candidate_generator import generate_candidate_pairs
from src.config import (
    CATCHMENT_META_PATH,
    CATCHMENT_SHP_PATH,
    DATA1_PATH,
    DATA2_PATH,
    DATA3_PATH,
    EXPECTED_COUNTS,
    PROCESSED_DIR,
    SCHOOL_LOCATION_PATH,
)
from src.gis_preprocessing import build_gis_assets
from src.preprocessing import build_education_assets, read_clean_csv
from src.schema import KEDI, SMALL_FLAG


def read_public_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"CSV 인코딩 확인 실패: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def build_manifest(output_dir: Path, row_counts: dict[str, int]) -> dict:
    files = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "dataset_manifest.json":
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return {
        "project": "부산 소규모 초등학교 통합 교육여건 변화 시뮬레이터",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dates": {"data1": "2025-04-01", "data2": "2025-10-01", "data3": "2025-04-01", "gis": "2026-03-20"},
        "analysis_universe": "2025-10-01 부산 공립 초등학교 운영 본교",
        "candidate_rule": "GIS 조인 가능한 소규모학교 A에서 학교점 직선거리 3km 이하의 다른 초등학교 B",
        "accessibility_rule": "A 통학구역 250m 균일격자에서 A/B 학교점까지 직선거리 비교",
        "row_counts": row_counts,
        "files": files,
    }


def assert_expected(row_counts: dict[str, int]) -> None:
    failures = {
        key: (row_counts.get(key), expected)
        for key, expected in EXPECTED_COUNTS.items()
        if key in row_counts and row_counts.get(key) != expected
    }
    if failures:
        formatted = ", ".join(f"{key}: {actual} (기대 {expected})" for key, (actual, expected) in failures.items())
        raise AssertionError(f"기준 산출값과 불일치: {formatted}")


def main() -> None:
    parser = argparse.ArgumentParser(description="원자료에서 Streamlit 배포용 가공 데이터를 생성합니다.")
    parser.add_argument("--data1", type=Path, default=DATA1_PATH)
    parser.add_argument("--data2", type=Path, default=DATA2_PATH)
    parser.add_argument("--data3", type=Path, default=DATA3_PATH)
    parser.add_argument("--catchment-shp", type=Path, default=CATCHMENT_SHP_PATH)
    parser.add_argument("--catchment-meta", type=Path, default=CATCHMENT_META_PATH)
    parser.add_argument("--school-location", type=Path, default=SCHOOL_LOCATION_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--skip-known-count-check", action="store_true")
    args = parser.parse_args()

    required = [args.data1, args.data2, args.data3, args.catchment_shp, args.catchment_meta, args.school_location]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"원자료 누락: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    education = build_education_assets(
        read_clean_csv(args.data1),
        read_clean_csv(args.data2),
        read_clean_csv(args.data3),
    )
    gis = build_gis_assets(
        education.master,
        gpd.read_file(args.catchment_shp, encoding="cp949"),
        read_public_csv(args.catchment_meta),
        read_public_csv(args.school_location),
    )
    pairs, candidate_counts = generate_candidate_pairs(gis.school_points, max_distance_km=3.0)

    save_csv(education.master, args.output_dir / "school_master_2025.csv")
    save_csv(education.small_schools, args.output_dir / "small_schools_2025.csv")
    save_csv(education.quality_report, args.output_dir / "data_quality_report.csv")
    save_csv(education.district_summary, args.output_dir / "eda_district_summary.csv")
    save_csv(education.region_summary, args.output_dir / "eda_region_summary.csv")
    save_csv(education.grade_summary, args.output_dir / "eda_grade_summary.csv")
    save_csv(pairs, args.output_dir / "candidate_pairs_within_3km_2025.csv")
    save_csv(candidate_counts, args.output_dir / "small_school_candidate_counts_2025.csv")
    save_csv(gis.excluded_schools, args.output_dir / "gis_excluded_schools_2025.csv")
    save_csv(gis.point_zone_qa, args.output_dir / "gis_point_zone_qa_2025.csv")

    point_export = gis.school_points.to_crs(4326)
    point_export.to_file(args.output_dir / "school_points_2025.geojson", driver="GeoJSON")
    catchment_export = gis.catchments.to_crs(5186).copy()
    catchment_export.geometry = catchment_export.geometry.simplify(10, preserve_topology=True)
    catchment_export.to_crs(4326).to_file(args.output_dir / "school_catchments_2025.geojson", driver="GeoJSON")

    row_counts = {
        "busan_operating_elementary_main": education.operating_count,
        "busan_public_elementary_main": len(education.master),
        "small_public_schools": int(education.master[SMALL_FLAG].sum()),
        "gis_usable_schools": len(gis.school_points),
        "gis_usable_small_schools": len(candidate_counts),
        "candidate_pairs_3km": len(pairs),
        "small_schools_with_candidate": int(candidate_counts["후보존재"].sum()),
        "small_schools_without_candidate": int((~candidate_counts["후보존재"]).sum()),
    }
    if not args.skip_known_count_check:
        assert_expected(row_counts)
    manifest = build_manifest(args.output_dir, row_counts)
    with (args.output_dir / "dataset_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print("배포 데이터 생성 완료")
    for key, value in row_counts.items():
        print(f"- {key}: {value:,}")
    print(f"- output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

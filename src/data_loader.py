from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.config import PROCESSED_DIR
from src.schema import KEDI, SMALL_FLAG, normalize_kedi, normalize_master_columns


@dataclass
class DataBundle:
    master: pd.DataFrame
    small_schools: pd.DataFrame
    candidate_pairs: pd.DataFrame
    candidate_counts: pd.DataFrame
    school_points: gpd.GeoDataFrame
    catchments: gpd.GeoDataFrame
    quality_report: pd.DataFrame
    excluded_schools: pd.DataFrame
    district_summary: pd.DataFrame
    region_summary: pd.DataFrame
    grade_summary: pd.DataFrame
    manifest: dict


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    for column in [KEDI, "소규모학교_KEDI", "후보학교_KEDI"]:
        if column in frame.columns:
            frame[column] = normalize_kedi(frame[column])
    return frame


def load_bundle(processed_dir: Path | str = PROCESSED_DIR) -> DataBundle:
    directory = Path(processed_dir)
    required = [
        "school_master_2025.csv",
        "small_schools_2025.csv",
        "candidate_pairs_within_3km_2025.csv",
        "small_school_candidate_counts_2025.csv",
        "school_points_2025.geojson",
        "school_catchments_2025.geojson",
        "data_quality_report.csv",
        "gis_excluded_schools_2025.csv",
        "eda_district_summary.csv",
        "eda_region_summary.csv",
        "eda_grade_summary.csv",
        "dataset_manifest.json",
    ]
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(f"배포 데이터 누락: {missing}. python -m scripts.build_assets를 먼저 실행하세요.")
    master = normalize_master_columns(_read_csv(directory / "school_master_2025.csv"))
    small = normalize_master_columns(_read_csv(directory / "small_schools_2025.csv"))
    for frame in (master, small):
        frame[SMALL_FLAG] = frame[SMALL_FLAG].astype("string").str.lower().map({"true": True, "false": False}).fillna(frame[SMALL_FLAG]).astype(bool)
    with (directory / "dataset_manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)
    return DataBundle(
        master=master,
        small_schools=small,
        candidate_pairs=_read_csv(directory / "candidate_pairs_within_3km_2025.csv"),
        candidate_counts=_read_csv(directory / "small_school_candidate_counts_2025.csv"),
        school_points=gpd.read_file(directory / "school_points_2025.geojson").to_crs(5186),
        catchments=gpd.read_file(directory / "school_catchments_2025.geojson").to_crs(5186),
        quality_report=_read_csv(directory / "data_quality_report.csv"),
        excluded_schools=_read_csv(directory / "gis_excluded_schools_2025.csv"),
        district_summary=_read_csv(directory / "eda_district_summary.csv"),
        region_summary=_read_csv(directory / "eda_region_summary.csv"),
        grade_summary=_read_csv(directory / "eda_grade_summary.csv"),
        manifest=manifest,
    )

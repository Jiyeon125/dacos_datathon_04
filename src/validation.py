from __future__ import annotations

import pandas as pd

from src.config import EXPECTED_COUNTS
from src.data_loader import DataBundle
from src.schema import KEDI, SMALL_FLAG


def validate_bundle(bundle: DataBundle) -> pd.DataFrame:
    pairs = bundle.candidate_pairs
    counts = bundle.candidate_counts
    observed = {
        "busan_public_elementary_main": len(bundle.master),
        "small_public_schools": int(bundle.master[SMALL_FLAG].sum()),
        "gis_usable_schools": len(bundle.school_points),
        "gis_usable_small_schools": len(counts),
        "candidate_pairs_3km": len(pairs),
        "small_schools_with_candidate": int(counts["후보존재"].astype(bool).sum()),
        "small_schools_without_candidate": int((~counts["후보존재"].astype(bool)).sum()),
    }
    rows = []
    for key, value in observed.items():
        expected = EXPECTED_COUNTS[key]
        rows.append({"검증항목": key, "관측값": value, "기대값": expected, "통과여부": value == expected})
    rows.extend(
        [
            {
                "검증항목": "master_kedi_unique",
                "관측값": bundle.master[KEDI].nunique(),
                "기대값": len(bundle.master),
                "통과여부": bundle.master[KEDI].is_unique,
            },
            {
                "검증항목": "candidate_distance_within_3km",
                "관측값": float(pairs["학교간직선거리_km"].max()),
                "기대값": "<=3.0",
                "통과여부": bool(pairs["학교간직선거리_km"].le(3.0).all()),
            },
            {
                "검증항목": "school_point_catchment_same_codes",
                "관측값": len(set(bundle.school_points[KEDI]) & set(bundle.catchments[KEDI])),
                "기대값": len(bundle.school_points),
                "통과여부": set(bundle.school_points[KEDI]) == set(bundle.catchments[KEDI]),
            },
            {
                "검증항목": "education_quality_report_all_passed",
                "관측값": int(bundle.quality_report["통과여부"].astype(bool).sum()),
                "기대값": len(bundle.quality_report),
                "통과여부": bool(bundle.quality_report["통과여부"].astype(bool).all()),
            },
            {
                "검증항목": "candidate_pairs_unique",
                "관측값": int(pairs.duplicated(["소규모학교_KEDI", "후보학교_KEDI"]).sum()),
                "기대값": 0,
                "통과여부": not pairs.duplicated(["소규모학교_KEDI", "후보학교_KEDI"]).any(),
            },
            {
                "검증항목": "candidate_has_no_self_pair",
                "관측값": int(pairs["소규모학교_KEDI"].eq(pairs["후보학교_KEDI"]).sum()),
                "기대값": 0,
                "통과여부": not pairs["소규모학교_KEDI"].eq(pairs["후보학교_KEDI"]).any(),
            },
            {
                "검증항목": "expected_gis_exclusions",
                "관측값": ", ".join(sorted(bundle.excluded_schools["학교명_20251001"].tolist())),
                "기대값": "괘법초등학교, 신선초등학교",
                "통과여부": set(bundle.excluded_schools["학교명_20251001"]) == {"괘법초등학교", "신선초등학교"},
            },
        ]
    )
    return pd.DataFrame(rows)


def assert_valid_bundle(bundle: DataBundle) -> pd.DataFrame:
    report = validate_bundle(bundle)
    failed = report.loc[~report["통과여부"]]
    if not failed.empty:
        raise AssertionError(f"배포 데이터 검증 실패:\n{failed.to_string(index=False)}")
    return report

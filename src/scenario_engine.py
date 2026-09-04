from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd

from src.accessibility_simulator import simulate_accessibility
from src.candidate_generator import PAIR_A_CODE, PAIR_B_CODE
from src.resource_simulator import simulate_resource_change
from src.schema import normalize_kedi


def run_scenario(
    master: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    school_points: gpd.GeoDataFrame,
    catchments: gpd.GeoDataFrame,
    a_code: str,
    b_code: str,
    spacing_m: float = 250.0,
) -> tuple[dict[str, Any], gpd.GeoDataFrame]:
    a_code, b_code = str(a_code), str(b_code)
    pairs = candidate_pairs.copy()
    pairs[PAIR_A_CODE] = normalize_kedi(pairs[PAIR_A_CODE])
    pairs[PAIR_B_CODE] = normalize_kedi(pairs[PAIR_B_CODE])
    selected = pairs.loc[pairs[PAIR_A_CODE].eq(a_code) & pairs[PAIR_B_CODE].eq(b_code)]
    if len(selected) != 1:
        raise ValueError("선택한 A→B 조합은 3km 후보 목록의 유일한 시나리오가 아닙니다.")
    pair = selected.iloc[0]
    resource = simulate_resource_change(master, a_code, b_code)
    accessibility, grid = simulate_accessibility(catchments, school_points, a_code, b_code, spacing_m)
    result = {
        "pair": {
            "distance_km": float(pair["학교간직선거리_km"]),
            "distance_rank": int(pair["거리순위_3km내"]),
            "rule": pair["후보생성기준"],
        },
        "resource": resource,
        "accessibility": accessibility,
    }
    return result, grid


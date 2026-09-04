from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.schema import KEDI, normalize_kedi


def build_accessibility_grid(
    catchments: gpd.GeoDataFrame,
    school_points: gpd.GeoDataFrame,
    a_code: str,
    b_code: str,
    spacing_m: float = 250.0,
) -> gpd.GeoDataFrame:
    """A 통학구역 안에 균일 격자를 만들고 A/B 학교점까지 직선거리를 비교한다."""
    if spacing_m <= 0:
        raise ValueError("격자 간격은 0보다 커야 합니다.")
    if catchments.crs is None or school_points.crs is None:
        raise ValueError("GIS 레이어에 좌표계가 필요합니다.")
    if catchments.crs != school_points.crs:
        school_points = school_points.to_crs(catchments.crs)
    if not catchments.crs.is_projected:
        raise ValueError("접근거리 계산에는 미터 단위 투영좌표계가 필요합니다.")

    zones = catchments.assign(**{KEDI: normalize_kedi(catchments[KEDI])}).set_index(KEDI)
    points = school_points.assign(**{KEDI: normalize_kedi(school_points[KEDI])}).set_index(KEDI)
    a_code, b_code = str(a_code), str(b_code)
    if a_code not in zones.index:
        raise KeyError(f"A 통학구역 없음: {a_code}")
    if a_code not in points.index or b_code not in points.index:
        raise KeyError("A 또는 B 학교점이 없습니다.")
    polygon = zones.loc[a_code].geometry
    if isinstance(polygon, pd.Series):
        polygon = polygon.iloc[0]
    a_point = points.loc[a_code].geometry
    b_point = points.loc[b_code].geometry
    if isinstance(a_point, pd.Series):
        a_point = a_point.iloc[0]
    if isinstance(b_point, pd.Series):
        b_point = b_point.iloc[0]

    min_x, min_y, max_x, max_y = polygon.bounds
    x_values = np.arange(min_x + spacing_m / 2, max_x, spacing_m)
    y_values = np.arange(min_y + spacing_m / 2, max_y, spacing_m)
    grid_points = [Point(x, y) for x in x_values for y in y_values if polygon.covers(Point(x, y))]
    if not grid_points:
        grid_points = [polygon.representative_point()]
    grid = gpd.GeoDataFrame({"격자번호": range(1, len(grid_points) + 1)}, geometry=grid_points, crs=catchments.crs)
    grid["현재거리_km"] = grid.geometry.distance(a_point) / 1000
    grid["통합후거리_km"] = grid.geometry.distance(b_point) / 1000
    grid["추가접근거리_km"] = grid["통합후거리_km"] - grid["현재거리_km"]
    grid["접근성악화"] = grid["추가접근거리_km"].gt(0)
    return grid


def summarize_accessibility(grid: gpd.GeoDataFrame, spacing_m: float = 250.0) -> dict[str, Any]:
    return {
        "current_mean_km": float(grid["현재거리_km"].mean()),
        "after_mean_km": float(grid["통합후거리_km"].mean()),
        "added_mean_km": float(grid["추가접근거리_km"].mean()),
        "added_median_km": float(grid["추가접근거리_km"].median()),
        "added_max_km": float(grid["추가접근거리_km"].max()),
        "worsened_pct": float(grid["접근성악화"].mean() * 100),
        "grid_point_count": int(len(grid)),
        "spacing_m": float(spacing_m),
        "assumption": "A 통학구역 250m 균일격자에서 학교까지 직선거리 비교; 학생·도로망 가중 없음",
    }


def simulate_accessibility(
    catchments: gpd.GeoDataFrame,
    school_points: gpd.GeoDataFrame,
    a_code: str,
    b_code: str,
    spacing_m: float = 250.0,
) -> tuple[dict[str, Any], gpd.GeoDataFrame]:
    grid = build_accessibility_grid(catchments, school_points, a_code, b_code, spacing_m)
    return summarize_accessibility(grid, spacing_m), grid


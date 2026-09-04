from __future__ import annotations

import zlib
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point

from src.schema import KEDI, normalize_kedi


MONTE_CARLO_MIN_ITER = 20
MONTE_CARLO_MAX_ITER = 300
MONTE_CARLO_CHECK_EVERY = 20
MONTE_CARLO_TOLERANCE_KM = 0.01


def _single_geometry(frame: gpd.GeoDataFrame, code: str):
    geometry = frame.loc[code].geometry
    return geometry.iloc[0] if isinstance(geometry, pd.Series) else geometry


def _uniform_points_in_polygon(
    polygon,
    point_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """폴리곤 내부에 균일 무작위 점을 rejection sampling으로 생성한다."""
    min_x, min_y, max_x, max_y = polygon.bounds
    chunks: list[np.ndarray] = []
    accepted_count = 0
    while accepted_count < point_count:
        batch_size = max(point_count * 2, 200)
        x_values = rng.uniform(min_x, max_x, batch_size)
        y_values = rng.uniform(min_y, max_y, batch_size)
        inside = shapely.contains_xy(polygon, x_values, y_values)
        accepted = np.column_stack([x_values[inside], y_values[inside]])
        if len(accepted):
            chunks.append(accepted)
            accepted_count += len(accepted)
    return np.concatenate(chunks, axis=0)[:point_count]


def simulate_accessibility_for_candidates(
    catchments: gpd.GeoDataFrame,
    school_points: gpd.GeoDataFrame,
    a_code: str,
    candidate_codes: tuple[str, ...],
    student_count: int,
    min_iter: int = MONTE_CARLO_MIN_ITER,
    max_iter: int = MONTE_CARLO_MAX_ITER,
    check_every: int = MONTE_CARLO_CHECK_EVERY,
    tolerance_km: float = MONTE_CARLO_TOLERANCE_KM,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """한 학교의 모든 후보를 동일한 무작위 표본으로 반복 계산한다."""
    if student_count <= 0:
        raise ValueError("가상점 생성에는 1명 이상의 학생 수가 필요합니다.")
    if not candidate_codes:
        raise ValueError("접근성을 계산할 수용학교 후보가 없습니다.")
    if catchments.crs is None or school_points.crs is None:
        raise ValueError("GIS 레이어에 좌표계가 필요합니다.")
    if catchments.crs != school_points.crs:
        school_points = school_points.to_crs(catchments.crs)
    if not catchments.crs.is_projected:
        raise ValueError("접근거리 계산에는 미터 단위 투영좌표계가 필요합니다.")

    zones = catchments.assign(**{KEDI: normalize_kedi(catchments[KEDI])}).set_index(KEDI)
    points = school_points.assign(**{KEDI: normalize_kedi(school_points[KEDI])}).set_index(KEDI)
    a_code = str(a_code)
    candidate_codes = tuple(str(code) for code in candidate_codes if str(code) in points.index)
    if a_code not in zones.index or a_code not in points.index:
        raise KeyError(f"통합 대상학교 GIS 자료 없음: {a_code}")
    if not candidate_codes:
        raise KeyError("GIS 자료가 있는 수용학교 후보가 없습니다.")

    polygon = _single_geometry(zones, a_code)
    a_point = _single_geometry(points, a_code)
    candidate_xy = np.array(
        [[_single_geometry(points, code).x, _single_geometry(points, code).y] for code in candidate_codes]
    )
    seed = zlib.crc32(f"accessibility-v1:{a_code}".encode("utf-8"))
    rng = np.random.default_rng(seed)

    running_current = 0.0
    running_after = np.zeros(len(candidate_codes), dtype=float)
    running_worsened = np.zeros(len(candidate_codes), dtype=float)
    added_samples: list[list[np.ndarray]] = [[] for _ in candidate_codes]
    previous_after_means: np.ndarray | None = None
    representative_points: np.ndarray | None = None
    representative_current: np.ndarray | None = None
    converged = False

    for iteration in range(1, max_iter + 1):
        sampled_xy = _uniform_points_in_polygon(polygon, student_count, rng)
        current_km = np.hypot(sampled_xy[:, 0] - a_point.x, sampled_xy[:, 1] - a_point.y) / 1000
        after_km = np.hypot(
            sampled_xy[:, None, 0] - candidate_xy[None, :, 0],
            sampled_xy[:, None, 1] - candidate_xy[None, :, 1],
        ) / 1000
        added_km = after_km - current_km[:, None]

        if representative_points is None:
            representative_points = sampled_xy.copy()
            representative_current = current_km.copy()
        running_current += float(current_km.mean())
        running_after += after_km.mean(axis=0)
        running_worsened += (added_km > 0).mean(axis=0) * 100
        for index in range(len(candidate_codes)):
            added_samples[index].append(added_km[:, index].copy())

        if iteration >= min_iter and iteration % check_every == 0:
            after_means = running_after / iteration
            if previous_after_means is not None:
                if float(np.max(np.abs(after_means - previous_after_means))) < tolerance_km:
                    converged = True
                    break
            previous_after_means = after_means.copy()

    current_mean = running_current / iteration
    records = []
    for index, code in enumerate(candidate_codes):
        pooled_added = np.concatenate(added_samples[index])
        after_mean = running_after[index] / iteration
        records.append(
            {
                "후보학교_KEDI": code,
                "current_mean_km": current_mean,
                "after_mean_km": after_mean,
                "added_mean_km": after_mean - current_mean,
                "added_median_km": float(np.median(pooled_added)),
                "added_max_km": float(np.max(pooled_added)),
                "worsened_pct": running_worsened[index] / iteration,
                "sample_point_count": student_count,
                "total_draw_count": student_count * iteration,
                "iterations": iteration,
                "converged": converged,
                "tolerance_km": tolerance_km,
                "assumption": (
                    "재학생 수와 같은 개수의 통학구역 내 균일 무작위 표본을 반복한 직선거리 평균; "
                    "실제 거주지·도로망 가중 없음"
                ),
            }
        )

    sample = gpd.GeoDataFrame(
        {
            "표본번호": range(1, student_count + 1),
            "현재거리_km": representative_current,
        },
        geometry=gpd.points_from_xy(representative_points[:, 0], representative_points[:, 1]),
        crs=catchments.crs,
    )
    return pd.DataFrame(records), sample


def build_candidate_sample(
    sample: gpd.GeoDataFrame,
    school_points: gpd.GeoDataFrame,
    b_code: str,
) -> gpd.GeoDataFrame:
    """대표 표본 한 세트에 선택한 수용학교까지의 거리를 붙인다."""
    if sample.crs != school_points.crs:
        school_points = school_points.to_crs(sample.crs)
    points = school_points.assign(**{KEDI: normalize_kedi(school_points[KEDI])}).set_index(KEDI)
    b_point = _single_geometry(points, str(b_code))
    result = sample.copy()
    result["통합후거리_km"] = result.geometry.distance(b_point) / 1000
    result["추가접근거리_km"] = result["통합후거리_km"] - result["현재거리_km"]
    result["접근성악화"] = result["추가접근거리_km"].gt(0)
    return result


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

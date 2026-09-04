import geopandas as gpd
from shapely.geometry import Point

from src.candidate_generator import generate_candidate_pairs
from src.schema import CLASS_SIZE, DISTRICT, KEDI, SCHOOL_NAME, SMALL_FLAG, STUDENTS


def test_candidates_include_only_other_schools_within_radius():
    points = gpd.GeoDataFrame(
        [
            {KEDI: "A", SCHOOL_NAME: "A초", DISTRICT: "갑구", STUDENTS: 50, CLASS_SIZE: 10, SMALL_FLAG: True, "geometry": Point(0, 0)},
            {KEDI: "B", SCHOOL_NAME: "B초", DISTRICT: "갑구", STUDENTS: 100, CLASS_SIZE: 20, SMALL_FLAG: False, "geometry": Point(2900, 0)},
            {KEDI: "C", SCHOOL_NAME: "C초", DISTRICT: "갑구", STUDENTS: 100, CLASS_SIZE: 20, SMALL_FLAG: False, "geometry": Point(3100, 0)},
        ],
        crs=5186,
    )
    pairs, counts = generate_candidate_pairs(points, 3.0)
    assert pairs["후보학교_KEDI"].tolist() == ["B"]
    assert pairs.iloc[0]["학교간직선거리_km"] == 2.9
    assert counts.iloc[0]["후보학교수_3km내"] == 1


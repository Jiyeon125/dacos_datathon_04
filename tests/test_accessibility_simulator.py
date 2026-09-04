import geopandas as gpd
from shapely.geometry import Point, Polygon

from src.accessibility_simulator import simulate_accessibility
from src.schema import KEDI


def test_accessibility_grid_reports_added_straight_distance():
    zones = gpd.GeoDataFrame([{KEDI: "A", "geometry": Polygon([(0, 0), (500, 0), (500, 500), (0, 500)])}], crs=5186)
    points = gpd.GeoDataFrame(
        [{KEDI: "A", "geometry": Point(250, 250)}, {KEDI: "B", "geometry": Point(1250, 250)}],
        crs=5186,
    )
    summary, grid = simulate_accessibility(zones, points, "A", "B", spacing_m=250)
    assert len(grid) == 4
    assert summary["after_mean_km"] > summary["current_mean_km"]
    assert summary["worsened_pct"] == 100.0


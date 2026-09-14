"""Tests for bbox-scoped manifest loading (Registry(bbox=...))."""

from pathlib import Path

import pandas as pd
import pytest

from geotessera.registry import Registry


def make_manifest(path: Path, rows):
    """Write a minimal file-scan-inventory-schema manifest.parquet."""
    df = pd.DataFrame(
        rows, columns=["year", "lon", "lat", "grid_size", "scales_size"]
    )
    df.to_parquet(path)


def make_empty_landmasks(path: Path):
    """Write a minimal landmasks.parquet so Registry() makes no network call."""
    pd.DataFrame(
        {"lon": [], "lat": [], "file_size": []}
    ).astype({"lon": "float64", "lat": "float64", "file_size": "int64"}).to_parquet(
        path
    )


def test_bbox_scopes_loaded_rows(tmp_path: Path):
    """Only rows within bbox (+ margin) should end up in the registry."""
    manifest_path = tmp_path / "manifest.parquet"
    inside = [(2024, 1.5, 42.5, 100, 10), (2024, 1.6, 42.6, 100, 10)]
    outside = [(2024, 50.0, -10.0, 100, 10), (2024, -120.0, 35.0, 100, 10)]
    make_manifest(manifest_path, inside + outside)
    landmasks_path = tmp_path / "landmasks.parquet"
    make_empty_landmasks(landmasks_path)

    bbox = (1.4, 42.4, 1.7, 42.7)
    registry = Registry(
        version="v1",
        registry_path=manifest_path,
        landmasks_registry_path=landmasks_path,
        bbox=bbox,
        bbox_margin_deg=0.15,
    )

    gdf = registry._registry_gdf
    assert len(gdf) == len(inside)
    lons = sorted(gdf["lon"].tolist())
    assert lons == [1.5, 1.6]


def test_bbox_margin_keeps_boundary_tiles(tmp_path: Path):
    """A tile just outside bbox, but within the margin, must be kept."""
    manifest_path = tmp_path / "manifest.parquet"
    rows = [
        (2024, 1.40, 42.50, 100, 10),  # 0.1 deg outside min_lon, within 0.15 margin
        (2024, 5.00, 42.50, 100, 10),  # far outside, must be dropped
    ]
    make_manifest(manifest_path, rows)
    landmasks_path = tmp_path / "landmasks.parquet"
    make_empty_landmasks(landmasks_path)

    bbox = (1.5, 42.4, 1.7, 42.7)
    registry = Registry(
        version="v1",
        registry_path=manifest_path,
        landmasks_registry_path=landmasks_path,
        bbox=bbox,
        bbox_margin_deg=0.15,
    )

    gdf = registry._registry_gdf
    assert len(gdf) == 1
    assert gdf["lon"].iloc[0] == pytest.approx(1.40)


def test_no_bbox_loads_everything(tmp_path: Path):
    """Without bbox, behaviour is unchanged: every row is loaded."""
    manifest_path = tmp_path / "manifest.parquet"
    rows = [
        (2024, 1.5, 42.5, 100, 10),
        (2024, 50.0, -10.0, 100, 10),
        (2024, -120.0, 35.0, 100, 10),
    ]
    make_manifest(manifest_path, rows)
    landmasks_path = tmp_path / "landmasks.parquet"
    make_empty_landmasks(landmasks_path)

    registry = Registry(
        version="v1",
        registry_path=manifest_path,
        landmasks_registry_path=landmasks_path,
    )

    assert len(registry._registry_gdf) == len(rows)

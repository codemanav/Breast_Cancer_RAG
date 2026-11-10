import datetime as dt

import openpyxl
import pytest

from src.data_loader import load_residences
from src.exposure import (
    exposure_text,
    haversine_km,
    load_pfas_sites,
    participant_exposures,
    residence_exposure,
    to_number,
    to_year,
)

HEADER = [
    "Site name", "State", "Other site name(s)", "Industry", "Sample Matrix", "Latittude", "Longitude",
    "Sample Date", "PFOA (ppt)", "PFOS (ppt)", "PFOA+PFOS from one sample (ppt)",
    "Total PFAS from one sample (ppt)",
]

# Two sites ~1-3 km from a home at (40.20, -75.09) and one ~150 km away.
NEAR_A = {"name": "Base A", "state": "PA", "industry": "Military", "matrix": "Groundwater",
          "lat": 40.21, "lon": -75.07, "sample_year": 2019, "pfoa_pfos_ppt": 3000.0, "total_pfas_ppt": None}
NEAR_B = {"name": "Well B", "state": "PA", "industry": None, "matrix": "Drinking Water",
          "lat": 40.19, "lon": -75.10, "sample_year": None, "pfoa_pfos_ppt": None, "total_pfas_ppt": None}
FAR = {"name": "Far C", "state": "NJ", "industry": "Landfill", "matrix": "Groundwater",
       "lat": 41.50, "lon": -75.09, "sample_year": 2020, "pfoa_pfos_ppt": 99999.0, "total_pfas_ppt": None}
HOME = {"participant_id": "P1", "place": "Home", "lat": 40.20, "lon": -75.09,
        "start_year": 1960, "end_year": 1980, "current": False}


@pytest.mark.parametrize(
    "raw, expected",
    [(3770, 3770.0), ("1,200", 1200.0), ("<4", 4.0), ("12.5 ppt", 12.5), ("ND", None), (None, None),
     (float("nan"), None)],
)
def test_to_number(raw, expected):
    assert to_number(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [(2019, 2019), ("2018-2019", 2018), (dt.datetime(2017, 5, 1), 2017), ("unknown", None), (None, None)],
)
def test_to_year(raw, expected):
    assert to_year(raw) == expected


def test_haversine_known_distance():
    # One degree of latitude along a meridian is 2*pi*R/360 km.
    assert haversine_km(40.0, -75.0, 41.0, -75.0) == pytest.approx(2 * 3.141592653589793 * 6371.0088 / 360)
    assert haversine_km(40.2, -75.1, 39.9, -75.2) == pytest.approx(haversine_km(39.9, -75.2, 40.2, -75.1))


def test_residence_finds_only_sites_within_radius():
    result = residence_exposure(HOME, [FAR, NEAR_A, NEAR_B], radius_km=5)

    assert result["n_sites"] == 2
    assert [s["name"] for s in result["sites"]] == ["Well B", "Base A"]  # nearest first
    assert result["drinking_water_sites"] == 1
    assert result["max_pfoa_pfos_ppt"] == 3000.0  # the far site's value is excluded
    assert result["years"] == 20


def test_no_sites_nearby():
    result = residence_exposure(HOME, [FAR], radius_km=5)

    assert result["n_sites"] == 0 and result["nearest_km"] is None


def test_durations_distinguish_unknown_from_present():
    unknown_end = {**HOME, "end_year": None, "current": False}
    current = {**HOME, "end_year": None, "current": True}

    assert residence_exposure(unknown_end, [NEAR_A], 5)["years"] is None
    assert residence_exposure(current, [NEAR_A], 5, current_year=2026)["years"] == 66


def test_participant_summary_and_text():
    homes = [HOME, {**HOME, "place": "Second", "start_year": None, "end_year": 1990}]

    summary = participant_exposures(homes, [NEAR_A, NEAR_B], radius_km=5)["P1"]["summary"]
    text = exposure_text(participant_exposures(homes, [NEAR_A, NEAR_B], radius_km=5)["P1"])

    assert summary["residences_near_sites"] == 2
    assert summary["known_years_near_sites"] == 20
    assert summary["has_unknown_durations"] is True
    assert "None" not in text
    assert "Base A" in text and "PFOA+PFOS 3000 ppt" in text
    assert "1960-1980" in text and "?-1990" in text


def test_exposure_text_without_history():
    assert exposure_text(None) == "No residence history available."


def test_load_pfas_sites_from_tracker_layout(tmp_path):
    path = tmp_path / "pfas.xlsx"
    wb = openpyxl.Workbook()
    wb.active.title = "READ ME"
    ws = wb.create_sheet("USA Contamination Sites")
    ws.append(HEADER)
    ws.append(["Base\nA", "PA", None, "Military", "Groundwater", 40.21, -75.07, 2019, 600, 2400, "3,000", None])
    ws.append(["No coords", "PA", None, None, None, None, None, None, None, None, None, None])
    wb.save(path)

    sites = load_pfas_sites(path)

    assert len(sites) == 1
    assert sites[0]["name"] == "Base A"  # whitespace normalised
    assert sites[0]["pfoa_pfos_ppt"] == 3000.0
    assert sites[0]["sample_year"] == 2019


def test_bundled_tracker_covers_pilot_sites():
    """The committed tracker loads and contains the sites near the pilot homes."""
    from src.config import PFAS_FILE

    sites = load_pfas_sites(PFAS_FILE)
    warminster = {"place": "Warminster, PA", "lat": 40.2007, "lon": -75.0899,
                  "start_year": 1976, "end_year": None, "current": True}

    nearby = residence_exposure(warminster, sites, radius_km=5)

    assert len(sites) > 1500
    assert {"Naval Air Warfare Center Warminster", "Horsham Air Guard Station"} <= {
        s["name"] for s in nearby["sites"]
    }


def test_repository_residences_parse():
    rows = load_residences()

    assert {r["participant_id"] for r in rows} == {"P001", "P002"}
    assert any(r["current"] for r in rows)
    assert any(r["end_year"] is None and not r["current"] for r in rows)

"""
Estimate PFAS exposure by matching each participant's residence history to
known contamination sites.

Data: PFAS Contamination Site Tracker, PFAS Project Lab, Social Science
Environmental Health Research Institute, Northeastern University (included
with the lab's permission; see data/external/README.md for terms of use).

Limitation: a site's sample date is when contamination was *detected*, not
when it began, so proximity is a screening signal, not a dose estimate.
"""

import datetime as dt
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.config import EXPOSURE_FILE, PFAS_RADIUS_KM

EARTH_RADIUS_KM = 6371.0088

# Column names in the tracker's "USA Contamination Sites" sheet ("Latittude" is
# spelled that way in the source).
COLUMNS = {
    "name": "Site name",
    "state": "State",
    "industry": "Industry",
    "matrix": "Sample Matrix",
    "lat": "Latittude",
    "lon": "Longitude",
    "sample_date": "Sample Date",
    "pfoa_pfos": "PFOA+PFOS from one sample (ppt)",
    "total_pfas": "Total PFAS from one sample (ppt)",
}


def to_number(value):
    """Parse tracker values like 3770, '1,200', '<4', 'ND' -> float or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None if isinstance(value, float) and math.isnan(value) else float(value)
    match = re.search(r"\d[\d,]*\.?\d*", str(value))
    return float(match.group(0).replace(",", "")) if match else None


def to_year(value):
    """Extract a year from ints, datetimes, or strings like '2019-2020'."""
    if isinstance(value, (dt.date, dt.datetime)):
        return value.year
    match = re.search(r"(19|20)\d{2}", str(value or ""))
    return int(match.group(0)) if match else None


def load_pfas_sites(path, sheet=None):
    """
    Load contamination sites with coordinates from the tracker workbook.
    Returns a list of dicts; rows without coordinates are dropped.
    """
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet] if sheet else next(s for s in wb.worksheets if "Contamination Sites" in s.title)
    rows = ws.iter_rows(values_only=True)
    header = [str(h).strip() if h else "" for h in next(rows)]
    index = {key: header.index(col) for key, col in COLUMNS.items()}

    sites = []
    for row in rows:
        lat, lon = to_number(row[index["lat"]]), to_number(row[index["lon"]])
        if lat is None or lon is None:
            continue
        sites.append(
            {
                "name": " ".join(str(row[index["name"]] or "").split()),
                "state": row[index["state"]],
                "industry": row[index["industry"]],
                "matrix": row[index["matrix"]],
                "lat": lat,
                "lon": lon,
                "sample_year": to_year(row[index["sample_date"]]),
                "pfoa_pfos_ppt": to_number(row[index["pfoa_pfos"]]),
                "total_pfas_ppt": to_number(row[index["total_pfas"]]),
            }
        )
    return sites


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; accepts scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _years(res, current_year):
    """Years lived at a residence, or None when the start or end is unknown."""
    start, end = res["start_year"], res["end_year"]
    if end is None and res.get("current"):
        end = current_year
    if start is None or end is None:
        return None
    return max(end - start, 0)


def _max(values):
    values = [v for v in values if v is not None]
    return max(values) if values else None


def residence_exposure(residence, sites, radius_km=PFAS_RADIUS_KM, current_year=None):
    """Sites within `radius_km` of one residence, with summary statistics."""
    current_year = current_year or dt.date.today().year
    if sites:
        lats = np.array([s["lat"] for s in sites])
        lons = np.array([s["lon"] for s in sites])
        dist = haversine_km(residence["lat"], residence["lon"], lats, lons)
        near = [(float(d), s) for d, s in zip(dist, sites) if d <= radius_km]
    else:
        near = []
    near.sort(key=lambda x: x[0])

    return {
        "place": residence["place"],
        "start_year": residence["start_year"],
        "end_year": residence["end_year"],
        "current": residence.get("current", False),
        "years": _years(residence, current_year),
        "n_sites": len(near),
        "drinking_water_sites": sum(1 for _, s in near if s["matrix"] == "Drinking Water"),
        "nearest_km": round(near[0][0], 2) if near else None,
        "nearest_site": near[0][1]["name"] if near else None,
        "max_pfoa_pfos_ppt": _max(s["pfoa_pfos_ppt"] for _, s in near),
        "max_total_pfas_ppt": _max(s["total_pfas_ppt"] for _, s in near),
        "sites": [
            {"name": s["name"], "km": round(d, 2), "industry": s["industry"], "matrix": s["matrix"],
             "sample_year": s["sample_year"], "pfoa_pfos_ppt": s["pfoa_pfos_ppt"]}
            for d, s in near
        ],
    }


def participant_exposures(residences, sites, radius_km=PFAS_RADIUS_KM, current_year=None):
    """Exposure summary for every participant with a residence history."""
    by_person = defaultdict(list)
    for r in residences:
        by_person[r["participant_id"]].append(r)

    result = {}
    for pid, rows in by_person.items():
        homes = [residence_exposure(r, sites, radius_km, current_year) for r in rows]
        exposed = [h for h in homes if h["n_sites"]]
        known_years = [h["years"] for h in exposed if h["years"] is not None]
        result[pid] = {
            "participant_id": pid,
            "radius_km": radius_km,
            "residences": homes,
            "summary": {
                "residences_near_sites": len(exposed),
                "known_years_near_sites": sum(known_years),
                "has_unknown_durations": any(h["years"] is None for h in exposed),
                "nearest_km": min((h["nearest_km"] for h in exposed), default=None),
                "max_pfoa_pfos_ppt": _max(h["max_pfoa_pfos_ppt"] for h in exposed),
            },
        }
    return result


def _site_text(site):
    parts = [f"{site['name']} {site['km']} km"]
    if site["matrix"]:
        parts.append(site["matrix"])
    if site["sample_year"]:
        parts.append(f"sampled {site['sample_year']}")
    if site["pfoa_pfos_ppt"] is not None:
        parts.append(f"PFOA+PFOS {site['pfoa_pfos_ppt']:g} ppt")
    return ", ".join(parts)


def exposure_text(exposure):
    """Compact, model-readable description of one participant's PFAS exposure."""
    if not exposure:
        return "No residence history available."
    lines = [f"PFAS contamination sites within {exposure['radius_km']:g} km of each residence "
             "(sample year = when detected, not when contamination began):"]
    for h in exposure["residences"]:
        end = "present" if h.get("current") else (h["end_year"] or "?")
        span = f"{h['start_year'] or '?'}-{end}"
        if not h["n_sites"]:
            lines.append(f"- {h['place']} ({span}): no known sites")
            continue
        sites = "; ".join(_site_text(s) for s in h["sites"][:3])
        lines.append(f"- {h['place']} ({span}): {h['n_sites']} site(s): {sites}")
    return "\n".join(lines)


def save_exposures(exposures, path=EXPOSURE_FILE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(exposures, indent=2, default=str), encoding="utf-8")


def load_exposures(path=EXPOSURE_FILE):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

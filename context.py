"""Spatial context for a report: OpenStreetMap features around the location,
turned into a suggested Population_Exposure term with a small fuzzy rule base
(zero-order Takagi-Sugeno: rule strength = min of memberships, output = weighted
average of the rule consequents).

Data source: OpenStreetMap via the public Overpass API (ODbL licence).
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

from fcm_model import nearest_term

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass-api.de/api/interpreter",  # retry once
    "https://overpass.kumi.systems/api/interpreter",
]
RADIUS_M = 200          # transit stops
VULNERABLE_RADIUS_M = 120
VULNERABLE = ("school", "kindergarten", "hospital", "nursing_home")

# How busy a street type is (0 = quiet, 1 = very busy), team judgement.
ROAD_BUSY = {
    "motorway": 1.0, "trunk": 1.0, "primary": 0.95, "secondary": 0.8, "tertiary": 0.65,
    "pedestrian": 0.8, "living_street": 0.4, "residential": 0.35, "unclassified": 0.3,
    "service": 0.2, "footway": 0.45, "cycleway": 0.4, "path": 0.2, "track": 0.1,
}


def fetch_osm(lat: float, lon: float, radius: int = RADIUS_M, timeout: int = 40) -> dict:
    """Count sensitive facilities and transit stops nearby and find the nearest street type."""
    q = f"""
[out:json][timeout:{timeout}];
(
  nwr(around:{VULNERABLE_RADIUS_M},{lat},{lon})[amenity~"^({'|'.join(VULNERABLE)})$"];
)->.vul;
(
  node(around:{radius},{lat},{lon})[highway=bus_stop];
  node(around:{radius},{lat},{lon})[railway=tram_stop];
  node(around:{radius},{lat},{lon})[public_transport=stop_position];
)->.pt;
way(around:30,{lat},{lon})[highway]->.roads;
.vul out tags center;
.pt out tags;
.roads out tags;
"""
    data = urllib.parse.urlencode({"data": q}).encode()
    last_err = None
    for url in OVERPASS_ENDPOINTS:  # public servers are often busy: try the next one
        try:
            req = urllib.request.Request(url, data=data,
                                         headers={"User-Agent": "urban-problem-fcm-student-project"})
            with urllib.request.urlopen(req, timeout=timeout + 5) as r:
                elements = json.load(r)["elements"]
            break
        except Exception as exc:
            last_err = exc
            time.sleep(1)
    else:
        raise RuntimeError(f"all Overpass servers failed, last error: {last_err}")

    vulnerable, stops, roads = [], set(), []
    for e in elements:
        t = e.get("tags", {})
        if t.get("amenity") in VULNERABLE:
            vulnerable.append(f"{t['amenity']}: {t.get('name', 'unnamed')}")
        elif t.get("highway") == "bus_stop" or t.get("railway") == "tram_stop" or t.get("public_transport"):
            stops.add(t.get("name", str(e["id"])))  # dedupe platforms of the same stop by name
        elif "highway" in t and e["type"] == "way":
            roads.append(t["highway"])
    return {"vulnerable": sorted(set(vulnerable)), "n_stops": len(stops), "roads": roads}


def ramp(x: float, lo: float, hi: float) -> float:
    """Membership rising linearly from 0 at lo to 1 at hi."""
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def exposure_from_context(ctx: dict) -> dict:
    n_vul = len(ctx["vulnerable"])
    busy = max((ROAD_BUSY.get(r, 0.3) for r in ctx["roads"]), default=0.3)
    mu = {
        "vulnerable_near": ramp(n_vul, 0, 2),        # 2+ schools/hospitals within 120 m -> fully "near"
        "transit_dense": ramp(ctx["n_stops"], 0, 4),  # 4+ distinct stops -> fully "dense"
        "street_busy": ramp(busy, 0.3, 0.9),
    }
    mu["street_quiet"] = 1 - mu["street_busy"]
    rules = [
        ("IF vulnerable facilities near THEN exposure very high", mu["vulnerable_near"], 0.92),
        ("IF transit stops dense THEN exposure high", mu["transit_dense"], 0.75),
        ("IF street busy THEN exposure high", mu["street_busy"], 0.75),
        ("IF street quiet AND transit sparse AND no vulnerable facilities THEN exposure low",
         min(mu["street_quiet"], 1 - mu["transit_dense"], 1 - mu["vulnerable_near"]), 0.25),
    ]
    num = sum(w * z for _, w, z in rules)
    den = sum(w for _, w, _ in rules)
    value = num / den if den > 0 else 0.5
    return {"value": value, "term": nearest_term(value), "memberships": mu,
            "rules": [(text, round(w, 2)) for text, w, _ in rules], "busiest_road_score": busy}

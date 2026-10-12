"""Reproduce five OSM functional maps within synthetic Sioux Falls OD areas.

Five broad functions follow the non-residential DisasterMobi classification.
The areas are nearest-node constructs, not official traffic-analysis zones;
mapped OSM objects are place proxies, not employment or trip totals.

Run with ``python -m src.environment.poi_function_maps``. This is an independent
environment study and does not alter the RL environment or demand model.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/siouxfalls_toy/poi"
OUT = ROOT / "outputs/studies/human_behavior/poi_function_maps"
from shapely.geometry import MultiPoint, Point, mapping, shape
from shapely.ops import transform, unary_union, voronoi_diagram

SERVERS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
)
CATEGORIES = ("commercial", "leisure", "industrial", "health", "public")
LABELS = {
    "commercial": "Commercial places",
    "leisure": "Leisure, food and lodging places",
    "industrial": "Industrial facilities and warehouses",
    "health": "Health and social-care places",
    "public": "Education and public-service places",
}
AMENITY = {
    "commercial": {"bank", "car_rental", "car_wash", "marketplace"},
    "leisure": {"restaurant", "cafe", "fast_food", "bar", "pub", "food_court", "ice_cream", "cinema", "theatre", "arts_centre", "nightclub"},
    "health": {"hospital", "clinic", "doctors", "dentist", "pharmacy", "nursing_home", "social_facility"},
    "public": {"school", "university", "college", "kindergarten", "library", "townhall", "police", "fire_station", "courthouse", "post_office"},
}
LEISURE = {"fitness_centre", "sports_centre", "golf_course", "bowling_alley", "stadium", "amusement_arcade", "escape_game", "water_park", "dance"}
TOURISM = {"hotel", "motel", "guest_house", "hostel", "museum", "attraction", "gallery", "apartment"}


def categories(tags: dict) -> list[str]:
    """Deterministic precedence avoids counting one OSM object twice."""
    amenity = tags.get("amenity", "")
    office = tags.get("office", "")
    shop = tags.get("shop", "")
    if amenity in AMENITY["health"] or tags.get("healthcare") not in (None, "no") or office == "healthcare":
        return ["health"]
    if amenity in AMENITY["public"] or office in {"government", "educational_institution"}:
        return ["public"]
    if tags.get("industrial") not in (None, "no") or tags.get("man_made") == "works" or tags.get("building") in {"industrial", "warehouse"}:
        return ["industrial"]
    if shop not in ("", "vacant", "no"):
        return ["commercial"]
    if amenity in AMENITY["leisure"] or tags.get("leisure") in LEISURE or tags.get("tourism") in TOURISM:
        return ["leisure"]
    if (shop not in ("", "vacant", "no") or office not in ("", "no") or
            tags.get("craft") not in (None, "", "no") or amenity in AMENITY["commercial"]):
        return ["commercial"]
    return []


def osm_query(bounds: tuple[float, float, float, float]) -> str:
    west, south, east, north = bounds
    bbox = f"({south-0.001:.6f},{west-0.001:.6f},{north+0.001:.6f},{east+0.001:.6f})"
    clauses = [
        '["shop"]', '["office"]', '["craft"]', '["healthcare"]', '["industrial"]',
        '["man_made"="works"]',
        '["building"~"^(industrial|warehouse)$"]', '["leisure"]', '["tourism"]',
        '["amenity"~"^(bank|car_rental|car_wash|marketplace|restaurant|cafe|fast_food|bar|pub|food_court|ice_cream|cinema|theatre|arts_centre|nightclub|hospital|clinic|doctors|dentist|pharmacy|nursing_home|social_facility|school|university|college|kindergarten|library|townhall|police|fire_station|courthouse|post_office)$"]',
    ]
    return "[out:json][timeout:110];(" + "".join(f"nwr{clause}{bbox};" for clause in clauses) + ");out center tags;"


def fetch(query: str) -> dict:
    cache = SOURCE / "osm_functional_features_source.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    errors = []
    for server in SERVERS:
        for attempt in range(2):
            try:
                response = requests.post(server, data={"data": query}, headers={"User-Agent": "RoadRestorationAcademicExploration/1.0"}, timeout=135)
                response.raise_for_status()
                result = response.json()
                if "elements" not in result:
                    raise ValueError("No Overpass elements in response")
                cache.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                return result
            except (requests.RequestException, ValueError) as exc:
                errors.append(f"{server} attempt {attempt+1}: {exc}")
                time.sleep(2)
    raise RuntimeError("All Overpass downloads failed: " + " | ".join(errors))


def polygon_patches(geometry) -> list[MplPolygon]:
    if geometry.geom_type == "Polygon":
        polygons = [geometry]
    elif geometry.geom_type == "MultiPolygon":
        polygons = list(geometry.geoms)
    else:
        raise ValueError(geometry.geom_type)
    return [MplPolygon(np.asarray(p.exterior.coords), closed=True) for p in polygons]


def draw_geography(ax, zones, values, cmap, norm, study, nodes, edges,
                   node_lookup, subset, *, compact=False):
    """Draw the same geographic data in full-study and uniform slide frames."""
    patches, patch_values = [], []
    for (_, geom), value in zip(zones, values):
        current = polygon_patches(geom)
        patches.extend(current)
        patch_values.extend([value] * len(current))
    collection = PatchCollection(
        patches, cmap=cmap, norm=norm, edgecolor="#ffffff",
        linewidth=0.9 if compact else 1.4, zorder=2,
    )
    collection.set_array(np.asarray(patch_values))
    ax.add_collection(collection)
    for geom in ([study] if study.geom_type == "Polygon" else study.geoms):
        bx, by = geom.exterior.xy
        ax.plot(bx, by, color="#555e6e", lw=1.5 if compact else 2.0, zorder=4)
    for edge in edges.itertuples():
        u, v = node_lookup.loc[edge.u], node_lookup.loc[edge.v]
        ax.plot([u.x, v.x], [u.y, v.y], color="#8e879a",
                lw=0.5 if compact else 0.8, alpha=0.45, zorder=3)
    ax.scatter(nodes.x, nodes.y, s=30 if compact else 68, facecolor="white",
               edgecolor="#352956", lw=1.1 if compact else 1.5, zorder=5)
    ax.scatter(subset.longitude, subset.latitude, s=30 if compact else 86,
               facecolor="#ee8738", edgecolor="white",
               linewidth=0.6 if compact else 0.9, alpha=0.90, zorder=6)
    return collection


def synthetic_zones(nodes: pd.DataFrame):
    """Clip 24 nearest-node cells to a buffered node hull and city limits."""
    if len(nodes) != 24 or nodes.node_id.nunique() != 24:
        raise ValueError("Expected the 24 distinct Sioux Falls OD nodes")
    lon0, lat0 = float(nodes.x.mean()), float(nodes.y.mean())
    km_per_lon = 111.320 * math.cos(math.radians(lat0))
    km_per_lat = 111.132

    def forward(lon, lat, z=None):
        return ((lon - lon0) * km_per_lon, (lat - lat0) * km_per_lat)

    def inverse(x, y, z=None):
        return (x / km_per_lon + lon0, y / km_per_lat + lat0)

    xy = np.column_stack(forward(nodes.x.to_numpy(), nodes.y.to_numpy()))
    separation = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(separation, np.inf)
    median_nearest_km = float(np.median(separation.min(axis=1)))
    buffer_km = float(np.clip(0.75 * median_nearest_km, 1.25, 3.0))
    city_source = json.loads((SOURCE / "city_limits_source.geojson").read_text(encoding="utf-8"))
    city = transform(forward, unary_union([shape(f["geometry"]) for f in city_source["features"]]))
    sites = [Point(float(x), float(y)) for x, y in xy]
    study = MultiPoint(sites).convex_hull.buffer(buffer_km).intersection(city)
    if any(not study.covers(site) for site in sites):
        raise ValueError("A benchmark OD node lies outside the synthetic study boundary")
    cells = list(voronoi_diagram(MultiPoint(sites), envelope=study.envelope.buffer(5)).geoms)
    zones = []
    for node_id, site in zip(nodes.node_id.astype(int), sites):
        owners = [cell for cell in cells if cell.buffer(1e-8).covers(site)]
        if len(owners) != 1:
            raise ValueError(f"Nearest-node cell ownership failed for OD node {node_id}")
        clipped = owners[0].intersection(study)
        if clipped.is_empty or not clipped.covers(site):
            raise ValueError(f"Empty synthetic area for OD node {node_id}")
        zones.append((node_id, transform(inverse, clipped)))
    union = unary_union([polygon for _, polygon in zones])
    if study.symmetric_difference(transform(forward, union)).area > 1e-6:
        raise ValueError("Synthetic OD areas do not partition the study boundary")
    return zones, {"buffer_km": buffer_km, "median_nearest_node_km": median_nearest_km,
                   "area_km2": study.area, "municipal_clip": "Sioux Falls city limits"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(ROOT / "data/siouxfalls_toy/network/nodes.csv")
    edges = pd.read_csv(ROOT / "data/siouxfalls_toy/network/edges.csv")
    zones, boundary = synthetic_zones(nodes)
    (OUT / "synthetic_od_zones.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "metadata": {"definition": "Nearest benchmark OD node within buffered node hull clipped to city limits", **boundary},
        "features": [{"type": "Feature", "properties": {"od_node": node_id},
                      "geometry": mapping(geom)} for node_id, geom in zones],
    }), encoding="utf-8")
    study = unary_union([geom for _, geom in zones])
    raw = fetch(osm_query(study.bounds))
    rows = []
    candidates = Counter()
    for element in raw["elements"]:
        loc = element if element["type"] == "node" else element.get("center", {})
        if "lon" not in loc or "lat" not in loc:
            continue
        p = Point(float(loc["lon"]), float(loc["lat"]))
        if not study.covers(p):
            continue
        tags = element.get("tags", {})
        matched = categories(tags)
        for category in matched:
            candidates[category] += 1
            owners = [node_id for node_id, geometry in zones if geometry.covers(p)]
            if len(owners) > 1:  # On a shared boundary, choose the nearest node.
                subset = nodes[nodes.node_id.isin(owners)]
                owner = int(subset.iloc[np.argmin((subset.x - p.x) ** 2 + (subset.y - p.y) ** 2)].node_id)
            elif len(owners) == 1:
                owner = owners[0]
            else:
                raise AssertionError("A point inside the study area has no zone")
            rows.append({"category": category, "osm_type": element["type"], "osm_id": int(element["id"]),
                         "name": tags.get("name", ""), "latitude": p.y, "longitude": p.x,
                         "od_node": owner, "primary_tag": next((f"{key}={tags[key]}" for key in ("amenity", "healthcare", "shop", "office", "craft", "leisure", "tourism", "industrial", "man_made", "landuse", "building") if key in tags), "")})
    data = pd.DataFrame(rows)
    if data.empty:
        raise RuntimeError("No mapped features inside the study boundary")
    if data.duplicated(["osm_type", "osm_id"]).any():
        raise AssertionError("An OSM object was assigned to multiple categories")
    data.to_csv(OUT / "osm_functional_features_in_synthetic_area.csv", index=False)
    counts = data.groupby(["category", "od_node"]).size()
    max_count = int(counts.max())
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 18, "axes.labelsize": 19, "xtick.labelsize": 16,
        "ytick.labelsize": 16, "legend.fontsize": 16,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    summary = {}
    node_lookup = nodes.set_index("node_id")
    west, south, east, north = study.bounds
    for category in CATEGORIES:
        subset = data.loc[data.category == category]
        values = [int(counts.get((category, node_id), 0)) for node_id, _ in zones]
        zone_max = max(values)
        # Common sequential family; discrete colour bins adapt to each category's range.
        colors = plt.get_cmap("Purples")(np.linspace(0.10, 0.86, zone_max + 1))
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(-0.5, zone_max + 1.5), cmap.N)
        fig, ax = plt.subplots(figsize=(11.8, 8.0), constrained_layout=True)
        collection = draw_geography(ax, zones, values, cmap, norm, study, nodes,
                                    edges, node_lookup, subset)
        # Use a continuous guide for dense categories and integer ticks for sparse ones.
        ticks = np.arange(zone_max + 1) if zone_max <= 12 else np.linspace(0, zone_max, 6).round().astype(int)
        cb = fig.colorbar(collection, ax=ax, shrink=0.84, pad=0.025, ticks=np.unique(ticks))
        cb.set_label("Mapped features per synthetic OD area", labelpad=12)
        ax.legend(handles=[
            Line2D([], [], marker="o", linestyle="None", markersize=11,
                   markerfacecolor="#ee8738", markeredgecolor="white", label=LABELS[category]),
            Line2D([], [], marker="o", linestyle="None", markersize=9,
                   markerfacecolor="white", markeredgecolor="#352956", label="OD node"),
            Line2D([], [], color="#555e6e", lw=2.0, label="Synthetic boundary"),
        ], loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=3,
            frameon=False, columnspacing=0.7, handletextpad=0.35,
            handlelength=1.25, labelspacing=0.25, borderaxespad=0.0)
        ax.set_xlim(west - 0.008, east + 0.008)
        ax.set_ylim(south - 0.006, north + 0.006)
        ax.set_aspect(1 / np.cos(np.deg2rad(float(nodes.y.mean()))))
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°N)")
        ax.grid(color="#eaedf2", lw=0.8)
        ax.set_axisbelow(True)
        # Fixed map aspect leaves unused space in the colourbar's original
        # layout slot. Place the bar relative to the rendered map edge.
        fig.canvas.draw()
        map_box = ax.get_position()
        bar_box = cb.ax.get_position()
        fig.set_layout_engine(None)
        cb.ax.set_position([map_box.x1 + 0.018, bar_box.y0,
                            bar_box.width, bar_box.height])
        output = OUT / f"sioux_falls_{category}_synthetic_od_areas.png"
        fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        # A second, label-free rendering uses an identical geographic viewport
        # and canvas for all five PowerPoint panels. Full maps remain above.
        slide_fig, slide_ax = plt.subplots(figsize=(4.1, 4.8), dpi=300)
        draw_geography(slide_ax, zones, values, cmap, norm, study, nodes,
                       edges, node_lookup, subset, compact=True)
        slide_ax.set_xlim(west - 0.008, east + 0.008)
        slide_ax.set_ylim(south - 0.006, north + 0.006)
        slide_ax.set_aspect(1 / np.cos(np.deg2rad(float(nodes.y.mean()))))
        slide_ax.axis("off")
        slide_fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.015)
        slide_output = OUT / f"sioux_falls_{category}_slide_map.png"
        slide_fig.savefig(slide_output, dpi=300, facecolor="white")
        plt.close(slide_fig)
        summary[category] = {"mapped_features": len(subset), "areas_with_features": sum(x > 0 for x in values),
                             "maximum_in_one_area": zone_max, "by_osm_element_type": subset.osm_type.value_counts().to_dict(),
                             "main_tags": subset.primary_tag.value_counts().head(12).to_dict(),
                             "figure": output.name, "slide_map": slide_output.name}
    (OUT / "functional_map_summary.json").write_text(json.dumps({
        "boundary": boundary,
        "source": "Frozen OpenStreetMap Overpass response in data/siouxfalls_toy/poi; city-limit GeoJSON from Sioux Falls GIS",
        "category_assignment": "One category per OSM element, in health/public/industrial/shop/leisure/remaining-commercial precedence",
        "categories": summary,
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

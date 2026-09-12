"""
gee_sar_export.py
------------------
Pulls Sentinel-1 GRD (VV, VH) imagery from Google Earth Engine for the same
field geometries/dates as the provided Sentinel-2 optical patches, and
exports them as GeoTIFFs to Google Drive.

Prereqs:
    pip install earthengine-api --break-system-packages
    earthengine authenticate      # one-time, opens a browser login

EDIT ME: `fields` should come from your actual field geometry/date table
(e.g. parsed from the optical dataset's metadata). This script is a
starting template, not a finished pipeline.
"""

import ee

ee.Authenize = None  # placeholder to avoid accidental import-time side effects


def init():
    ee.Initialize()


def get_sentinel1_patch(geometry, start_date, end_date, orbit_pass="DESCENDING"):
    """
    geometry: ee.Geometry (field boundary, e.g. from a bounding box or polygon)
    start_date, end_date: 'YYYY-MM-DD' strings bracketing the optical
        acquisition date (e.g. +/- 5-6 days, since S1 revisit is ~6-12 days)

    Returns an ee.Image with bands ['VV', 'VH'], median-composited over the
    date range to reduce speckle a little before pixel-level filtering.
    """
    coll = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
    )
    return coll.median().clip(geometry)


def export_field_patch(image, geometry, description, folder="SAR_exports", scale=10):
    """Kick off an export task to Google Drive. Check the GEE Tasks tab to monitor."""
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=geometry,
        scale=scale,
        crs="EPSG:4326",
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


def export_all_fields(fields):
    """
    fields: list of dicts, each like
        {"field_id": "...", "geometry": ee.Geometry(...), "date": "YYYY-MM-DD"}
    EDIT: build this list from your parsed optical-patch metadata.
    """
    tasks = []
    for f in fields:
        geom = f["geometry"]
        date = f["date"]
        # +/- a few days window around the optical acquisition date
        start = ee.Date(date).advance(-6, "day")
        end = ee.Date(date).advance(6, "day")
        img = get_sentinel1_patch(geom, start, end)
        task = export_field_patch(img, geom, description=f"S1_{f['field_id']}")
        tasks.append(task)
        print(f"Started export for field {f['field_id']}")
    return tasks


if __name__ == "__main__":
    init()
    # EDIT: replace with your real field list
    example_fields = []
    export_all_fields(example_fields)

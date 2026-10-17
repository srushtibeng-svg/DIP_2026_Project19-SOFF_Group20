"""
gee_sar_export.py

Export Sentinel-1 VV and VH imagery from Google Earth Engine.

Requires:
    earthengine-api

A real field geometry and acquisition date are required.
"""

import ee


def init():
    """Initialize Google Earth Engine."""

    ee.Initialize()


def get_sentinel1_patch(
    geometry,
    start_date,
    end_date,
    orbit_pass=None
):
    """Retrieve Sentinel-1 VV/VH imagery for a geometry and date range."""

    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(
            ee.Filter.eq(
                "instrumentMode",
                "IW"
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation",
                "VV"
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation",
                "VH"
            )
        )
        .select(["VV", "VH"])
    )

    if orbit_pass is not None:
        collection = collection.filter(
            ee.Filter.eq(
                "orbitProperties_pass",
                orbit_pass
            )
        )

    count = collection.size().getInfo()

    if count == 0:
        raise ValueError(
            "No Sentinel-1 images found for this geometry/date range."
        )

    return collection.median().clip(geometry)


def export_field_patch(
    image,
    geometry,
    description,
    folder="SAR_exports",
    scale=10
):
    """Export a Sentinel-1 image to Google Drive."""

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=geometry,
        scale=scale,
        fileFormat="GeoTIFF",
        maxPixels=1e9
    )

    task.start()

    return task


def export_all_fields(fields):
    """
    Expected field format:
        {
            "field_id": "field_001",
            "geometry": ee.Geometry(...),
            "date": "2017-10-29"
        }
    """

    tasks = []

    for field in fields:
        required = [
            "field_id",
            "geometry",
            "date"
        ]

        for key in required:
            if key not in field:
                raise ValueError(
                    f"Missing '{key}' in field information."
                )

        field_id = str(field["field_id"])
        geometry = field["geometry"]
        date = field["date"]

        start_date = (
            ee.Date(date)
            .advance(-6, "day")
        )

        end_date = (
            ee.Date(date)
            .advance(6, "day")
        )

        image = get_sentinel1_patch(
            geometry,
            start_date,
            end_date
        )

        task = export_field_patch(
            image=image,
            geometry=geometry,
            description=f"S1_{field_id}"
        )

        tasks.append(task)

        print(
            f"Export started for field: {field_id}"
        )

    return tasks


if __name__ == "__main__":
    init()

    # Add verified field geometries and dates here.
    # Do not use dummy coordinates.
    fields = []

    if not fields:
        print(
            "No verified field geometries supplied. "
            "No SAR exports were started."
        )
    else:
        export_all_fields(fields)

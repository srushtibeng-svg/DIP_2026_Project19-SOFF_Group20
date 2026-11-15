
import ee
import pandas as pd
import rasterio
import requests
import json
import time
import math

from pathlib import Path
from pyproj import Transformer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT = "dip-project19-sar-fusion"

METADATA_DIR = Path("SAR_selected_orbit_metadata")

METADATA_FILE = (
    METADATA_DIR / "selected_orbit_SAR_metadata_all_fields.csv"
)

OUTPUT_DIR = Path("S1_selected_orbit_downloads")
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 10
PIXEL_SIZE = 10
ROI_SIZE = 210

MAX_RETRIES = 3


# ============================================================
# FIELD INFORMATION
# ============================================================

FIELDS = {
    "Germany_DUP3_farm1_field55_rapeseed_2019": {
        "longitude": 13.6830215808288,
        "latitude": 53.53984552831187,
        "epsg": 32633,
        "orbit": 168
    },

    "Germany_DUP3_farm1_field58_wheat_2019": {
        "longitude": 13.643199348127297,
        "latitude": 53.50771368193889,
        "epsg": 32633,
        "orbit": 168
    },

    "Uruguay_DUP4_farm2_field29_soybean_2021": {
        "longitude": -58.11317531,
        "latitude": -34.05093412,
        "epsg": 32721,
        "orbit": 68
    },

    "Uruguay_DUP4_farm7_field178_soybean_2020": {
        "longitude": -57.75139448,
        "latitude": -32.49499952,
        "epsg": 32721,
        "orbit": 68
    }
}


# ============================================================
# CONNECT TO GOOGLE EARTH ENGINE
# ============================================================

def initialize_gee():

    print("\nConnecting to Google Earth Engine...")

    ee.Initialize(project=PROJECT)

    print("GEE connected successfully.")


# ============================================================
# CREATE ROI AND 10m GRID
# ============================================================

def create_roi(field_info):

    epsg = field_info["epsg"]

    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg}",
        always_xy=True
    )

    center_x, center_y = transformer.transform(
        field_info["longitude"],
        field_info["latitude"]
    )

    left = (
        math.floor(center_x / PIXEL_SIZE) * PIXEL_SIZE - 100
    )

    top = (
        math.ceil(center_y / PIXEL_SIZE) * PIXEL_SIZE + 100
    )

    right = left + ROI_SIZE
    bottom = top - ROI_SIZE

    region = ee.Geometry.Rectangle(
        [left, bottom, right, top],
        proj=f"EPSG:{epsg}",
        geodesic=False
    )

    transform = [
        PIXEL_SIZE, 0, left,
        0, -PIXEL_SIZE, top
    ]

    return region, transform, left, top


# ============================================================
# CREATE SELECTED ORBIT SAR COLLECTION
# ============================================================

def create_collection(field_info, field_metadata):

    first_row = field_metadata.iloc[0]

    start_date = first_row["sowing_date"]

    end_date = (
        pd.to_datetime(first_row["harvest_date"])
        + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    point = ee.Geometry.Point([
        field_info["longitude"],
        field_info["latitude"]
    ])

    region = point.buffer(150)

    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(
            ee.Filter.eq("instrumentMode", "IW")
        )
        .filter(
            ee.Filter.eq(
                "orbitProperties_pass",
                "DESCENDING"
            )
        )
        .filter(
            ee.Filter.eq(
                "relativeOrbitNumber_start",
                field_info["orbit"]
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
        .sort("system:time_start")
    )

    return collection


# ============================================================
# SAVE BATCH METADATA
# ============================================================

def save_batch_metadata(
    field_name,
    field_info,
    field_metadata,
    batch_number,
    offset,
    batch_count,
    field_output,
    left,
    top
):

    batch_metadata = []

    for local_index in range(batch_count):

        global_index = offset + local_index

        row = field_metadata.iloc[global_index]

        record = {
            "field_name": field_name,
            "batch_number": batch_number,
            "batch_local_order": local_index + 1,
            "global_acquisition_order": global_index + 1,
            "acquisition_date": row["acquisition_date"],
            "image_id": row["image_id"],
            "orbit_pass": row["orbit_pass"],
            "relative_orbit": int(row["relative_orbit"]),
            "platform": row["platform"],
            "VV_band": local_index * 2 + 1,
            "VH_band": local_index * 2 + 2,
            "EPSG": field_info["epsg"],
            "pixel_size_m": PIXEL_SIZE,
            "ROI_width_m": ROI_SIZE,
            "ROI_height_m": ROI_SIZE,
            "ROI_left": left,
            "ROI_top": top
        }

        batch_metadata.append(record)

    batch_df = pd.DataFrame(batch_metadata)

    output_csv = (
        field_output /
        f"{field_name}_batch_{batch_number:03d}_metadata.csv"
    )

    output_json = (
        field_output /
        f"{field_name}_batch_{batch_number:03d}_metadata.json"
    )

    batch_df.to_csv(
        output_csv,
        index=False
    )

    with open(output_json, "w", encoding="utf-8") as file:

        json.dump(
            batch_metadata,
            file,
            indent=4
        )

    print("Batch CSV metadata saved.")
    print("Batch JSON metadata saved.")


# ============================================================
# VERIFY DOWNLOADED TIFF
# ============================================================

def verify_tiff(
    tiff_path,
    field_info,
    batch_count
):

    with rasterio.open(tiff_path) as src:

        expected_bands = batch_count * 2

        print("\nVerifying TIFF...")
        print("Bands:", src.count)
        print("Dimensions:", src.width, "x", src.height)
        print("CRS:", src.crs)
        print("Resolution:", src.res)

        if src.count != expected_bands:
            raise ValueError(
                f"Expected {expected_bands} bands, got {src.count}"
            )

        if src.width != 21 or src.height != 21:
            raise ValueError(
                "Unexpected TIFF dimensions."
            )

        if src.crs.to_epsg() != field_info["epsg"]:
            raise ValueError(
                "Unexpected TIFF CRS."
            )

        if not math.isclose(
            abs(src.res[0]),
            PIXEL_SIZE,
            abs_tol=0.001
        ):
            raise ValueError(
                "Unexpected horizontal pixel resolution."
            )

        if not math.isclose(
            abs(src.res[1]),
            PIXEL_SIZE,
            abs_tol=0.001
        ):
            raise ValueError(
                "Unexpected vertical pixel resolution."
            )

    print("TIFF verification passed.")

    return True


# ============================================================
# DOWNLOAD ONE BATCH
# ============================================================

def download_batch(
    field_name,
    collection,
    field_info,
    field_metadata,
    batch_number,
    offset,
    batch_count,
    field_output
):

    print("\n" + "-" * 60)
    print(f"Downloading batch {batch_number:03d}")
    print("Acquisitions:", batch_count)
    print("-" * 60)

    download_name = (
        f"{field_name}_batch_{batch_number:03d}"
    )

    output_tiff = (
        field_output / f"{download_name}.tif"
    )

    region, transform, left, top = create_roi(
        field_info
    )

    # --------------------------------------------------------
    # CHECK EXISTING TIFF
    # --------------------------------------------------------

    if output_tiff.exists():

        print("Existing TIFF found:", output_tiff)

        try:

            verify_tiff(
                output_tiff,
                field_info,
                batch_count
            )

        except Exception as error:

            print("Existing TIFF verification failed:", error)
            print("Please check this TIFF before continuing.")
            return False

    else:

        # ----------------------------------------------------
        # CREATE IMAGE COLLECTION FOR THIS BATCH
        # ----------------------------------------------------

        image_list = collection.toList(
            batch_count,
            offset
        )

        batch_collection = ee.ImageCollection.fromImages(
            image_list
        )

        stacked_image = batch_collection.toBands()

        download_params = {
            "name": download_name,
            "format": "GEO_TIFF",
            "region": region,
            "crs": f"EPSG:{field_info['epsg']}",
            "crs_transform": transform,
            "filePerBand": False
        }

        # ----------------------------------------------------
        # DOWNLOAD WITH RETRIES
        # ----------------------------------------------------

        success = False

        temporary_file = (
            field_output /
            f"{download_name}.temporary"
        )

        for attempt in range(1, MAX_RETRIES + 1):

            try:

                print(
                    f"GEE download attempt "
                    f"{attempt}/{MAX_RETRIES}"
                )

                url = stacked_image.getDownloadURL(
                    download_params
                )

                response = requests.get(
                    url,
                    timeout=180
                )

                response.raise_for_status()

                if len(response.content) < 100:
                    raise ValueError(
                        "Downloaded file is too small."
                    )

                temporary_file.write_bytes(
                    response.content
                )

                # Verify temporary TIFF before saving final file.
                verify_tiff(
                    temporary_file,
                    field_info,
                    batch_count
                )

                temporary_file.replace(output_tiff)

                print("Download successful:", output_tiff)

                success = True
                break

            except Exception as error:

                print("Download error:", error)

                if temporary_file.exists():
                    temporary_file.unlink()

                if attempt < MAX_RETRIES:

                    print("Retrying after 10 seconds...")
                    time.sleep(10)

        if not success:

            print("Batch download failed.")
            return False

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    save_batch_metadata(
        field_name,
        field_info,
        field_metadata,
        batch_number,
        offset,
        batch_count,
        field_output,
        left,
        top
    )

    print(f"Batch {batch_number:03d} completed successfully.")

    return True


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    initialize_gee()

    if not METADATA_FILE.exists():

        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}"
        )

    metadata = pd.read_csv(METADATA_FILE)

    required_columns = [
        "field_name",
        "acquisition_order",
        "acquisition_date",
        "image_id",
        "orbit_pass",
        "relative_orbit",
        "platform",
        "sowing_date",
        "harvest_date"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in metadata.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing metadata columns: {missing_columns}"
        )

    print("\n" + "=" * 65)
    print("SELECTED ORBIT SAR BATCH DOWNLOAD")
    print("=" * 65)

    total_successful = 0
    total_failed = 0

    # --------------------------------------------------------
    # PROCESS FOUR FIELDS
    # --------------------------------------------------------

    for field_name, field_info in FIELDS.items():

        print("\n" + "=" * 65)
        print("FIELD:", field_name)
        print("EPSG:", field_info["epsg"])
        print("Relative orbit:", field_info["orbit"])
        print("Pass: DESCENDING")
        print("=" * 65)

        field_metadata = metadata[
            metadata["field_name"] == field_name
        ].copy()

        field_metadata = (
            field_metadata.sort_values(
                "acquisition_order"
            )
            .reset_index(drop=True)
        )

        if field_metadata.empty:

            print("No metadata found. Skipping field.")
            total_failed += 1
            continue

        # Verify orbit selection.
        actual_orbits = set(
            field_metadata["relative_orbit"].astype(int)
        )

        if actual_orbits != {field_info["orbit"]}:

            raise ValueError(
                f"Orbit mismatch for {field_name}: "
                f"{actual_orbits}"
            )

        # Create GEE collection.
        collection = create_collection(
            field_info,
            field_metadata
        )

        gee_count = collection.size().getInfo()

        metadata_count = len(field_metadata)

        if gee_count != metadata_count:

            raise ValueError(
                f"Metadata/GEE count mismatch for {field_name}. "
                f"Metadata={metadata_count}, GEE={gee_count}"
            )

        field_output = OUTPUT_DIR / field_name
        field_output.mkdir(exist_ok=True)

        total_images = metadata_count

        total_batches = math.ceil(
            total_images / BATCH_SIZE
        )

        print("Total acquisitions:", total_images)
        print("Total batches:", total_batches)

        # ----------------------------------------------------
        # PROCESS BATCHES
        # ----------------------------------------------------

        for batch_index in range(total_batches):

            offset = batch_index * BATCH_SIZE

            batch_count = min(
                BATCH_SIZE,
                total_images - offset
            )

            try:

                success = download_batch(
                    field_name,
                    collection,
                    field_info,
                    field_metadata,
                    batch_index + 1,
                    offset,
                    batch_count,
                    field_output
                )

                if success:
                    total_successful += 1
                else:
                    total_failed += 1

            except Exception as error:

                print(
                    f"Unexpected error in batch "
                    f"{batch_index + 1}: {error}"
                )

                total_failed += 1

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL DOWNLOAD SUMMARY")
    print("=" * 65)

    print("Successful batches:", total_successful)
    print("Failed batches:", total_failed)

    print("Output folder:", OUTPUT_DIR.resolve())

    if total_failed == 0:

        print("\nAll batches completed successfully.")

    else:

        print(
            "\nSome batches failed. "
            "Check the errors and rerun after fixing them."
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

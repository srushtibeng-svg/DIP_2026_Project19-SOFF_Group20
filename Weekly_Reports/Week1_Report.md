# Weekly Progress Report — Week 1

**Project Title:** SAR–Optical Feature Fusion for Crop-Type / Yield Classification
**Repo:** <GitHub repo link>
**Team:** <names / roll numbers>
**Date:** <date>

## Work Done This Week
- Created GitHub repository `DIP_2026_ProjectNo_GroupNumber` with the required
  directory structure (`Mid_Sem_Report/`, `End_Sem_Report/`, `Results/`, `Codes/`).
- Drafted overall project plan and 10-week timeline.
- Reviewed the provided Sentinel-2 optical dataset for assigned countries
  (<country 1>, <country 2>): patch size, number of bands, crop-type label
  format, per-pixel yield mask format.
- Set up Google Earth Engine account/access for later Sentinel-1 acquisition.
- Read introductory literature on SAR–optical fusion for agricultural
  classification (see below).

## Literature Review
- <Paper 1 — one-line takeaway relevant to feature choice or fusion strategy>
- <Paper 2 — one-line takeaway>
- <Paper 3 — one-line takeaway>

## Work Planned for Next Week
- Build a data loader that pairs optical patches with labels and identifies
  field geometries/dates needed to pull matching Sentinel-1 SAR data.
- Check class balance and dataset size per country.
- Begin scripting the GEE export for Sentinel-1 GRD (VV/VH) for the same fields.

## Challenges Faced
- <e.g., understanding the exact yield-mask format / GEE quota limits / etc.>

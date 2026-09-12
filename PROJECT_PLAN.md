# 10-Week Project Plan — SAR–Optical Feature Fusion

Assumes Week 1 = the week this brief was issued. Adjust dates to your actual
academic calendar; keep the **Saturday 5 PM commit** cadence regardless.
Insert your institute's midterm week wherever it actually falls — the plan
below places it at Week 6 as an example; move the whole block if yours
differs.

| Week | Focus | Concrete deliverables this week | Commit by Sat 5 PM |
|---|---|---|---|
| **1** | Kickoff, repo, literature | Create GitHub repo with required folder structure; skim 4–5 papers on SAR/optical fusion for crop classification; set up GEE account & Sentinel-1 access; inspect the provided Sentinel-2 dataset (bands, patch size, label format, yield mask format) | README + repo skeleton + Week1 report |
| **2** | Data understanding & pairing | Write loader for optical patches + labels; identify field IDs/geometries so SAR can be pulled for the *same* fields; sanity-check label distribution, class imbalance | `Codes/utils.py` (data loading) + Week2 report |
| **3** | SAR acquisition | Pull Sentinel-1 GRD (VV, VH) for matched fields/dates via GEE, export as GeoTIFF; basic preprocessing (calibration, speckle filtering, co-registration check against optical) | SAR patches saved + `Codes/gee_sar_export.py` + Week3 report |
| **4** | Optical feature extraction | Implement spectral indices (NDVI, NDWI, EVI, SAVI) + texture (GLCM: contrast, homogeneity, energy, correlation) + basic color/band statistics per patch | `Codes/optical_features.py` + first feature CSV + Week4 report |
| **5** | SAR feature extraction | Implement backscatter statistics (mean/std/median of VV, VH), VV/VH ratio, GLCM texture on SAR intensity, optionally speckle-robust features | `Codes/sar_features.py` + SAR feature CSV + Week5 report |
| **6** | *(Midterm exam week — no weekly commit required)* Mid-sem report writing | Consolidate Weeks 1–5 into `Mid_Sem_Report/` (problem statement, data, methodology so far, preliminary optical-only results if ready) | **Mid-sem report submitted** (separate deadline, not the weekly commit) |
| **7** | Fusion + optical-only model | Implement fusion strategies (simple concatenation; optionally PCA / feature selection); train Random Forest & SVM optical-only baseline; record accuracy/F1/confusion matrix | `Codes/fusion.py`, `Codes/train_classifier.py` (optical-only mode) + Week7 report |
| **8** | SAR-only model | Train same classifiers SAR-only; compare against optical-only; start error analysis (which crop types confuse which) | Results/tables (SAR-only metrics) + Week8 report |
| **9** | Fused model + comparison | Train fused-feature classifier; build comparison table/plots (optical vs SAR vs fused) across accuracy, F1, confusion matrices; ablate fusion choices if time permits | Results/graphs (comparison chart) + Week9 report |
| **10** | Finalize & report | Clean code, finalize `Results/`, write `End_Sem_Report/` (full methodology, results, discussion, limitations, future work), final commit | **End-sem report submitted** + Week10 report |

## Notes
- From Week 2 onward, every weekly commit should include **both** a report and a code update — an empty/cosmetic commit the day before the deadline is exactly what's being watched for.
- Literature review is expected explicitly in Weeks 1–2 reports; after that it's optional per report but worth a line if something relevant comes up.
- Keep raw large imagery out of git (see `.gitignore`) — commit code, small feature CSVs, plots, and reports, not multi-GB rasters.

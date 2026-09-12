# Codes

## Setup
```bash
pip install -r requirements.txt --break-system-packages
earthengine authenticate     # one-time, for gee_sar_export.py
```

## Files
| File | Purpose |
|---|---|
| `utils.py` | Load optical/SAR raster patches + label table. **Edit first** to match your actual dataset format. |
| `gee_sar_export.py` | Pull matching Sentinel-1 (VV/VH) patches from GEE for the same fields as the optical data. |
| `optical_features.py` | Spectral indices (NDVI, NDWI, EVI, SAVI), band stats, GLCM texture from Sentinel-2 patches. |
| `sar_features.py` | Backscatter stats, VV/VH ratio, GLCM texture from Sentinel-1 patches. |
| `fusion.py` | Combine optical + SAR feature dicts into a fused feature table; scaling/PCA helpers. |
| `train_classifier.py` | Train RF/SVM for optical-only, SAR-only, fused settings; save accuracy/F1/confusion matrices to `../Results/`. |

## Suggested run order
1. Fix up `utils.py` for your real file/label format.
2. `gee_sar_export.py` — export SAR patches for your assigned fields (one-time, or per new batch of fields).
3. Loop over patches, call `extract_optical_features()` / `extract_sar_features()`, collect into records, then `fusion.build_feature_dataframe(records)` → save as `features_wide.csv`.
4. In `train_classifier.py`, load that CSV and call:
   ```python
   df = pd.read_csv("features_wide.csv")
   run_comparison(df, model_name="rf")
   run_comparison(df, model_name="svm")
   ```
5. Check `Results/graphs/`, `Results/tables/` for the comparison plots and confusion matrices.

## Notes
- Raw imagery is not committed (see `.gitignore`) — only code, small feature CSVs, and result plots/tables go into git.
- These scripts are a **starting scaffold**: the exact dataset paths, band ordering, and label format need to be filled in based on what the course staff actually provided.

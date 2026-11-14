# SAR-Optical Feature Fusion for Crop-Type Classification

## Project Overview
This project uses Sentinel-1 SAR and Sentinel-2 optical data for crop-type analysis.

## Optical Feature Extraction
- 120 Germany fields: 60 Rapeseed and 60 Wheat.
- 12 optical features from NDVI, NDRE, NDMI and EVI.
- Dataset: germany_optical_features_120.csv

## Machine Learning Results
Leave-One-Farm-Out cross-validation across six farms.

| Model | Pooled OOF Accuracy | Macro F1 |
|---|---:|---:|
| Random Forest | 92.50% | 0.9249 |
| SVM | 90.83% | 0.9083 |

## SAR Feature Extraction
Four pilot fields were processed: Germany Rapeseed 2019, Germany Wheat 2019, Uruguay Soybean 2020 and Uruguay Soybean 2021.

A total of 214 SAR acquisition records were processed. VV/VH statistical and GLCM texture features were extracted.

## SAR-Optical Analysis
- Optical observations: 134
- Matched observations within three days: 126
- Unmatched observations: 8

## Limitations
- Classification experiment covers only Germany, Rapeseed and Wheat.
- SAR analysis is an exploratory four-field pilot.
- No separately validated SAR-only or fused-feature classifier was produced.
- SAR patches are 21 x 21 pixels at 10 m around field centroids, not verified whole-field boundaries.
- Optical indices use available raster footprints; clipping to verified field polygons was not established.

## Main Output Folders
- Optical_ML_Results
- SAR_Feature_Extraction
- SAR_Optical_Fusion
- Final_Master_Results
- SAR_selected_orbit_metadata

## Requirements
Install dependencies using:
pip install -r requirements.txt

Google Earth Engine authentication is required for satellite download scripts.

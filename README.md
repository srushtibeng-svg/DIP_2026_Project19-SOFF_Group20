# SAR–Optical Feature Fusion for Crop-Type Classification

## Project Overview

This project investigates the use of Sentinel-1 SAR and Sentinel-2 optical satellite data for crop-type classification.

The main objective is to evaluate whether combining optical and SAR features improves crop classification compared with using either modality independently.

The project focuses on two crop classes in Germany:

- Rapeseed
- Wheat

Three experiments were conducted:

1. Optical-only classification using 120 German fields.
2. Optical, SAR, and fused-feature classification using 30 German fields.
3. SAR–optical temporal matching and feature analysis using four pilot fields from Germany and Uruguay.

The project evaluates crop-type classification. It does not include a separately validated crop-yield prediction model.

---

## 1. Optical Feature Extraction

### Dataset

- 120 German fields.
- 60 Rapeseed fields.
- 60 Wheat fields.
- Six farms.
- 12 optical predictors.

The optical predictors include vegetation indices and temporal statistics derived from Sentinel-2 observations.

The vegetation indices include:

- NDVI
- NDRE
- NDMI
- EVI

### Machine Learning

Two classifiers were evaluated:

- Random Forest
- Support Vector Machine

Evaluation used Leave-One-Farm-Out cross-validation, with farm identifiers used as groups.

### Optical Classification Results

| Model | Pooled OOF Accuracy | Macro F1 |
|---|---:|---:|
| Random Forest | 92.50% | 0.9249 |
| SVM | 90.83% | 0.9083 |

The Random Forest model achieved the highest pooled out-of-fold accuracy in the 120-field optical experiment.

One farm contributed 69 of the 120 fields, which represents 57.5% of the dataset. Therefore, the pooled results are influenced by the field distribution across farms.

### Dataset File

`germany_optical_features_120.csv`

---

## 2. Optical, SAR, and Fusion Classification

A controlled modality comparison was conducted using 30 German fields.

### Dataset

- 30 German fields.
- 15 Rapeseed fields.
- 15 Wheat fields.
- Six farms.

The same fields and farm-wise cross-validation folds were used for all three modalities.

### Feature Sets

| Modality | Predictors |
|---|---:|
| Optical | 12 |
| SAR | 16 |
| Fusion | 28 |

Optical features were derived from Sentinel-2 observations.

SAR features were derived from temporal statistics of Sentinel-1 VV and VH backscatter.

Fusion was performed by concatenating the optical and SAR predictors.

### Classification Models

**Random Forest**

- 300 estimators.
- Balanced class weights.
- Minimum leaf size of 2.

**Support Vector Machine**

- RBF kernel.
- C = 1.
- Gamma = scale.
- StandardScaler.
- Balanced class weights.

### Evaluation

Leave-One-Group-Out cross-validation was performed using `farm_identifier`.

Six farms were held out one at a time.

Pooled out-of-fold predictions were used to calculate:

- Accuracy
- Balanced accuracy
- Macro precision
- Macro recall
- Macro F1-score

### Pooled Out-of-Fold Results

| Modality | Model | Accuracy | Macro F1 |
|---|---|---:|---:|
| Optical | Random Forest | 83.33% | 0.8331 |
| Optical | SVM | 93.33% | 0.9330 |
| SAR | Random Forest | 83.33% | 0.8331 |
| SAR | SVM | 93.33% | 0.9330 |
| Fusion | Random Forest | 96.67% | 0.9666 |
| Fusion | SVM | 90.00% | 0.8999 |

The fused Random Forest model achieved the highest pooled out-of-fold accuracy, correctly classifying 29 out of 30 fields.

Feature fusion improved the Random Forest result in this experiment.

However, fusion did not improve the SVM result. Therefore, the performance benefit of feature fusion depends on the classifier.

### Confusion Matrices

The pooled confusion matrices were:

**Optical Random Forest**

`[[12, 3], [2, 13]]`

**SAR Random Forest**

`[[12, 3], [2, 13]]`

**Fusion Random Forest**

`[[14, 1], [0, 15]]`

The fused Random Forest correctly classified 14 Rapeseed fields and all 15 Wheat fields.

### Main Result Files

`SAR_30_features/SAR_Field_Level_Features_30.csv`

`SAR_30_features/Optical_30_Fields.csv`

`SAR_30_features/SAR_30_Fields.csv`

`SAR_30_features/Optical_SAR_Fusion_30_Fields.csv`

`SAR_30_features/ML_Results_30/summary_metrics.csv`

`SAR_30_features/ML_Results_30/fold_metrics.csv`

`SAR_30_features/ML_Results_30/oof_predictions.csv`

`SAR_30_features/ML_Results_30/modality_comparison_30.png`

---

## 3. SAR Feature Extraction

Four pilot fields were processed:

1. Germany Rapeseed 2019.
2. Germany Wheat 2019.
3. Uruguay Soybean 2020.
4. Uruguay Soybean 2021.

Sentinel-1 GRD observations were retrieved using VV and VH backscatter channels.

### SAR Acquisition

- Total SAR acquisition records: 214.
- SAR channels: VV and VH.
- Patch size: 21 × 21 pixels.
- Nominal pixel spacing: 10 m.

The SAR patches were extracted around field centroids.

These patches represent point-centred samples and are not verified whole-field polygon averages.

### SAR Features

Temporal statistics were extracted from VV and VH backscatter.

For the four-field pilot, additional GLCM texture features were analyzed:

- Contrast
- Homogeneity
- Energy
- Correlation

The pilot was used to investigate SAR acquisition, feature extraction, and SAR–optical temporal matching.

---

## 4. SAR–Optical Temporal Matching

Optical observations were matched with SAR observations using a three-day temporal matching window.

### Matching Results

| Pilot Field | Optical Observations | Matched | Match Rate |
|---|---:|---:|---:|
| Germany Rapeseed 2019 | 42 | 40 | 95.24% |
| Germany Wheat 2019 | 42 | 39 | 92.86% |
| Uruguay Soybean 2020 | 24 | 22 | 91.67% |
| Uruguay Soybean 2021 | 26 | 25 | 96.15% |
| **Total** | **134** | **126** | **94.03%** |

Eight optical observations did not have a SAR observation within the selected three-day matching window.

The pilot generated SAR–optical feature and texture summaries for the selected fields.

This experiment evaluates the feasibility of the data-processing workflow. It is not a separately validated SAR-only or fused-feature crop classifier.

---

## 5. Limitations

- The 120-field optical experiment covers Germany.
- The 30-field modality comparison covers six German farms.
- Classification classes are Rapeseed and Wheat.
- The four-field SAR pilot includes Uruguay Soybean fields.
- The pilot does not establish cross-country classification generalization.
- SAR patches are centroid-based samples.
- Complete SAR extraction using verified field boundaries was not established.
- Optical indices use available raster footprints.
- Clipping to verified field polygons was not established for every result.
- The 30-field classification dataset is relatively small.
- Farm-level sampling distribution affects pooled evaluation results.
- Fusion improved Random Forest but not SVM.
- No separately validated crop-yield prediction model was produced.

---

## 6. Main Output Folders

### Optical Machine Learning

`Optical_ML_Results`

Contains optical-only classification results for the 120-field experiment.

### SAR 30-Field Classification

`SAR_30_features`

Contains the 30-field optical, SAR, and fusion datasets and classification results.

### SAR Feature Extraction

`SAR_Feature_Extraction`

Contains SAR acquisition-level and field-level feature tables.

### SAR–Optical Fusion

`SAR_Optical_Fusion`

Contains four-field pilot fusion outputs, comparison graphs, correlation analysis, and validation outputs.

### Final Master Results

`Final_Master_Results`

Contains the four-field pilot master results and field-level comparison tables.

### SAR Time-Series Graphs

`SAR_timeseries_graphs`

Contains SAR time-series visualizations for the selected pilot fields.

### SAR Time-Series Statistics

`SAR_timeseries_statistics`

Contains pilot SAR statistical summaries.

### Selected Orbit Metadata

`SAR_selected_orbit_metadata`

Contains selected SAR acquisition metadata for the pilot fields.

### Spatial Validation

`Spatial_Validation`

Contains spatial validation outputs and supporting evidence.

---

## 7. Reproducibility

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Google Earth Engine authentication is required for satellite download scripts.

The project includes scripts for:

- Optical feature extraction.
- Sentinel-1 SAR acquisition.
- SAR feature extraction.
- SAR–optical fusion.
- Classification experiments.
- Confusion matrix generation.
- Result verification.

The 120-field optical experiment, 30-field modality comparison, and four-field SAR–optical pilot are separate experiments and should be interpreted independently.

---

## 8. Conclusion

The project evaluated optical-only classification, SAR-only classification, and feature-level SAR–optical fusion.

The 120-field optical experiment achieved 92.50% pooled out-of-fold accuracy using Random Forest.

In the controlled 30-field comparison, fused Random Forest achieved the highest pooled out-of-fold accuracy of 96.67%.

The four-field pilot demonstrated temporal matching between optical and SAR observations and generated SAR–optical feature and texture summaries.

The results suggest that SAR–optical feature fusion can improve crop-type classification for the tested Random Forest setting. However, larger datasets, verified field-boundary extraction, and independent geographic validation are required before drawing broader conclusions about generalization.

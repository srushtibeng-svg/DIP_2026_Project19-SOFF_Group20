# SAR–Optical Feature Fusion for Crop-Type / Yield Classification

**Course:** Digital Image Processing (DIP), 2026
**Project No.:** <fill in>   **Group No.:** <fill in>
**Team members:** <name (roll no.)>, <name (roll no.)>, <name (roll no.)>
**Assigned countries:** <Germany + Uruguay  /  Argentina + Brazil>
**Repo link:** <paste GitHub URL here once created>

---

## 1. Problem Statement
Given co-registered SAR (Sentinel-1) and optical (Sentinel-2) field crops with an
associated crop-type / yield-derived label, we:
1. Extract **classical, hand-crafted features** from each modality separately.
2. **Fuse** the optical and SAR feature vectors.
3. Train a **classical ML model** (Random Forest / SVM) in three settings —
   optical-only, SAR-only, fused — and compare performance.

## 2. Data
| Source | Content | Provided by |
|---|---|---|
| Sentinel-2 | Optical field-crop patches, crop-type labels, per-pixel yield masks | Course staff (YieldSAT-derived) |
| Sentinel-1 | SAR (VV/VH) patches for the *same* fields | Acquired by the group via Google Earth Engine |

Countries used: **<country 1>** and **<country 2>**.

## 3. Repository Structure
```
DIP_2026_ProjectNo_GroupNumber/
├── Mid_Sem_Report/       # Mid-semester report (pdf/docx + source)
├── End_Sem_Report/       # End-semester report (pdf/docx + source)
├── Results/
│   ├── graphs/           # accuracy/F1 bar charts, ROC curves, etc.
│   ├── images/           # example SAR/optical crops, feature maps
│   └── tables/           # confusion matrices, metric tables (csv/png)
├── Codes/                # all source code (see Codes/README.md)
├── Weekly_Reports/        # one .md file per week (progress log)
└── README.md
```

## 4. Pipeline Overview
```
Sentinel-2 optical patch ──► optical_features.py ──┐
                                                     ├──► fusion.py ──► train_classifier.py ──► metrics + confusion matrix
Sentinel-1 SAR patch (GEE) ──► sar_features.py ─────┘
```

## 5. How to Run
See `Codes/README.md` for setup and run instructions.

## 6. Weekly Progress
Progress is logged weekly in `Weekly_Reports/`. Each report follows the
standard template: *Work done this week / Work planned for next week /
Literature review (if applicable) / Challenges faced*. Commits are made
every **Saturday, 5:00 PM**, except during the midterm exam week.

## 7. Status
| Milestone | Status |
|---|---|
| Repo + plan set up | ✅ |
| Data exploration | ⬜ |
| GEE SAR acquisition | ⬜ |
| Optical feature extraction | ⬜ |
| SAR feature extraction | ⬜ |
| Feature fusion | ⬜ |
| Optical-only model | ⬜ |
| SAR-only model | ⬜ |
| Fused model | ⬜ |
| Mid-sem report | ⬜ |
| End-sem report | ⬜ |

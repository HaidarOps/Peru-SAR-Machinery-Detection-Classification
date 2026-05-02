# Peruvian Amazon — Illegal Activity Detection Pipeline
### SAR-based disturbance classification with RADD alert integration

**Author:** Haidar Chawki Kassem 
**Institution:** Queen Mary University of London   
**Platform:** Google Earth Engine (JavaScript)  

---

## Overview

This project implements a cloud-based remote sensing pipeline in Google Earth Engine that combines Sentinel-1 synthetic aperture radar (SAR) imagery with RADD forest disturbance alerts to detect and classify illegal activity inside protected areas of the Peruvian Amazon.

A Random Forest classifier trained on manually delineated training polygons distinguishes four land cover states. RADD alert sites are classified by majority vote of SAR pixels within a 400m buffer, then assigned an operational threat level for ranger prioritisation.

### Output classes

| Label | Class | Description |
|---|---|---|
| 0 | Intact forest | Undisturbed Amazon canopy |
| 1 | Historical disturbance | Old mining scars, stable ponds, weathered tailings |
| 2 | Active disturbance | Operating mines, turbid water, fresh bare soil, recent clearing |
| 3 | Coca cultivation | Planted coca plots with visible row structure |

### Threat levels

| Level | Trigger |
|---|---|
| CRITICAL | Active disturbance, active_proportion ≥ 0.7 |
| HIGH | Active disturbance (0.5–0.7), coca cultivation, or freshly cleared forest |
| MEDIUM | Historical disturbance — abandoned site |
| LOW | Intact forest or RADD false positive |

---

## Running the Script

The complete pipeline including all training polygons is available as a shared Google Earth Engine script:

**[Click here to open in Google Earth Engine](https://code.earthengine.google.com/YOUR_LINK_HERE)**

> A free Google Earth Engine account is required. Register at [earthengine.google.com](https://earthengine.google.com).

Training polygons are embedded as drawn geometry imports in the shared script — no separate upload or import is required. Simply open the link, click **Run**, and the pipeline will execute.

---

## Repository Structure

```
peru-sar-illegal-activity-detection/
│
├── README.md
│
├── src/
│   └── peru_sar_radd_disturbance_classifier.js       ← annotated GEE pipeline script
│
├── data/
│   ├── training_intact_forest.geojson          ← reference copy of training polygons
│   ├── training_historical_disturbance.geojson
│   ├── training_active_disturbance.geojson
│   ├── training_coca_cultivation.geojson
│   ├── training_freshly_cleared.geojson
│   └── coca_peru_2014.csv                      ← coca presence points (Dávalos et al. 2016)
│
├── outputs/
│   └── alert_sites_madre_de_dios_2026_04_20.csv   ← final classified alert site results
│
└── notebooks/
    └── results_analysis.ipynb            ← post-processing figures and summary tables
```

> Note: The GeoJSON files in `data/` are reference copies of the training polygons for documentation purposes. The authoritative version of the training polygons is embedded in the shared GEE script linked above.

---

## Model Performance

Evaluated using polygon-level train/test split (75% train / 25% test) to prevent spatial autocorrelation between training and test pixels — a known source of inflated accuracy in pixel-level splits.

| Metric | Value |
|---|---|
| Overall accuracy | 78.1% |
| Cohen's Kappa | 0.70 |
| Intact forest — producer accuracy | 100% |
| Historical disturbance — producer accuracy | 65% |
| Active disturbance — producer accuracy | 62% |
| Coca cultivation — producer accuracy | 76% |

---

## Output Description

### CSV columns (`alert_sites_madre_de_dios_2026_04_20.csv`)

| Column | Description |
|---|---|
| `alert_id` | Unique RADD alert site identifier |
| `classification` | Majority vote class (0=forest, 1=historical, 2=active, 3=coca) |
| `threat_level` | Operational priority: LOW / MEDIUM / HIGH / CRITICAL |
| `activity_label` | Human-readable classification description |
| `active_proportion` | Fraction of pixels voting active disturbance (0–1) — drives CRITICAL vs HIGH |
| `n_pixels` | Total SAR pixels sampled within 400m buffer |
| `n_forest` | Pixels classified as intact forest |
| `n_historical` | Pixels classified as historical disturbance |
| `n_active` | Pixels classified as active disturbance |
| `n_coca` | Pixels classified as coca cultivation |
| `region` | Geographic region of the alert |
| `.geo` | Alert centroid coordinates (GeoJSON point) |

### GeoTIFF bands (`disturbance_classification_madre_de_dios_2026_04_20.tif`)

| Band | Name | Values |
|---|---|---|
| 1 | classification | 0=intact forest, 1=historical disturbance, 2=active disturbance, 3=coca cultivation |
| 2 | threat_level | 0=none, 1=MEDIUM, 2=HIGH, 3=CRITICAL |
| 3 | protected_area | 0=outside protected area, 1=inside protected area |

---

## Results Summary

200 RADD alert sites analysed inside protected areas of the Madre de Dios focus region:

| Threat level | Sites | Interpretation |
|---|---|---|
| CRITICAL | 18 | High confidence active illegal activity |
| HIGH | 41 | Active disturbance or coca cultivation detected |
| MEDIUM | 85 | Historical disturbance — abandoned site |
| LOW | 56 | Intact forest or RADD false positive |

29.5% of all RADD alerts inside protected areas showed active or recent illegal activity signatures. Coca cultivation was detected inside protected area boundaries.

---

## Data Sources

| Dataset | Source | GEE Path / Reference |
|---|---|---|
| Sentinel-1 GRD | ESA / Copernicus | `COPERNICUS/S1_GRD` |
| Hansen Global Forest Change v1.12 | UMD / Google | `UMD/hansen/global_forest_change_2024_v1_12` |
| SRTM DEM | USGS | `USGS/SRTMGL1_003` |
| RADD Forest Disturbance Alerts | Wageningen University | `projects/radar-wur/raddalert/v1` |
| WDPA Protected Areas | IUCN / UNEP-WCMC | `WCMC/WDPA/current/polygons` |
| Coca presence points | Dávalos et al. 2016, Dryad | `doi:10.5061/dryad.1hb1f` |

---

## Key Methodological Notes

**Freshly cleared forest** was initially treated as a separate class but produced 0% producer accuracy. SAR backscatter from fresh felled debris is spectrally indistinguishable from active mining at 100m resolution. Freshly cleared polygons were merged into active disturbance for the final model. This is documented as a limitation in the dissertation methodology chapter.

**Coca cultivation** training polygons were delineated manually from high-resolution satellite imagery at locations identified using Dávalos et al. (2016) presence points as spatial reference. The Dávalos dataset is a 30km grid survey — points indicate the neighbourhood of cultivation, not precise field boundaries. Final polygon placement was confirmed visually from row structure visible in optical imagery.

**Polygon-level validation** splits on polygon ID rather than individual pixels, preventing spatial autocorrelation between training and test sets — a standard requirement for remote sensing classification validation.

**SAR feature stack** consists of 15 bands: gamma-nought VV and VH backscatter, VV/VH polarisation ratio, and 6 GLCM texture features each for VV and VH (contrast, dissimilarity, IDM, ASM, entropy, correlation). Computed from a May–July 2025 median composite of Sentinel-1 descending orbit acquisitions.

---

## Citation

If using this workflow or dataset, please cite:

> Haidar Chawki Kassem (2026). *Peruvian Amazon Illegal Activity Detection using SAR and RADD Alerts*. Queen Mary University of London, Final Year Dissertation.

> Dávalos, L.M., Sanchez, K.M. & Armenteras, D. (2016). Data from: Deforestation and coca cultivation rooted in twentieth-century development projects. *Dryad*. https://doi.org/10.5061/dryad.1hb1f

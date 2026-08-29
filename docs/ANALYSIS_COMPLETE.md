# fMRI GLM Analysis - Complete Summary

**Date**: August 18, 2026  
**Study**: Speech Perception in Schizophrenia (PLoS ONE reproduction)  
**Analysis Type**: First and Second-Level GLM Analysis

---

## ✅ ANALYSIS STATUS: COMPLETE

### First-Level Analysis
- **Status**: ✅ COMPLETE
- **Subjects**: 71/71 (100%)
- **Contrast Maps**: 213 total (71 subjects × 3 contrasts)
- **Location**: `/root/fMRI/analysis/first_level/contrast_maps/`

**Contrasts Analyzed**:
1. `words_vs_baseline`
2. `sentences_vs_baseline`
3. `reversed_vs_baseline`

### Second-Level Analysis
- **Status**: ✅ COMPLETE
- **Location**: `/root/fMRI/analysis/second_level/`

#### One-Sample T-Tests (Group-Level Activation)
Generated for each group (HC, AVH+, AVH-) × 3 contrasts = 9 analyses

| Group | Contrast | Z-Map | Thresholded Map | Figure |
|-------|----------|-------|-----------------|--------|
| HC | words_vs_baseline | ✅ | ✅ | ✅ |
| HC | sentences_vs_baseline | ✅ | ✅ | ✅ |
| HC | reversed_vs_baseline | ✅ | ✅ | ✅ |
| AVH+ | words_vs_baseline | ✅ | ✅ | ✅ |
| AVH+ | sentences_vs_baseline | ✅ | ✅ | ✅ |
| AVH+ | reversed_vs_baseline | ✅ | ✅ | ✅ |
| AVH- | words_vs_baseline | ✅ | ✅ | ✅ |
| AVH- | sentences_vs_baseline | ✅ | ✅ | ✅ |
| AVH- | reversed_vs_baseline | ✅ | ✅ | ✅ |

#### Two-Sample T-Tests (Group Comparisons)
AVH+ vs HC comparisons for all 3 contrasts

| Comparison | Contrast | Z-Map | Thresholded Map | Figure |
|------------|----------|-------|-----------------|--------|
| AVH+ > HC | words_vs_baseline | ✅ | ✅ | ✅ |
| AVH+ > HC | sentences_vs_baseline | ✅ | ✅ | ✅ |
| AVH+ > HC | reversed_vs_baseline | ✅ | ✅ | ✅ |

---

## Output Summary

### Statistical Maps (24 files)
**Location**: `/root/fMRI/analysis/second_level/stat_maps/`

**Unthresholded Z-Maps** (12 files, ~3.8-4.0 MB each):
- 9 one-sample t-test maps (3 groups × 3 contrasts)
- 3 two-sample t-test maps (AVH+ vs HC)

**Thresholded Maps** (12 files, 38-129 KB each):
- FDR-corrected cluster-thresholded maps
- Alpha = 0.05, cluster_threshold = 10 voxels

### Publication Figures (12 files)
**Location**: `/root/fMRI/analysis/second_level/figures/`

**Glass Brain Plots** (12 PNG files, ~600-700 KB each):
- Threshold: Z > 2.3
- Display mode: lyrz (left, right, axial, sagittal)
- Colormap: cold_hot (blue-white-red)

---

## Participant Demographics

| Group | N | Description |
|-------|---|-------------|
| HC | 25 | Healthy Controls |
| AVH+ | 23 | Schizophrenia with Auditory Hallucinations |
| AVH- | 23 | Schizophrenia without Auditory Hallucinations |
| **Total** | **71** | |

**Covariates**:
- Age (continuous)
- Sex (binary: male/female)
- IQ (continuous, 2 missing values filled with median)

---

## Analysis Parameters

### First-Level GLM
- **TR**: 2.0 seconds
- **Smoothing**: FWHM = 5.0 mm
- **Motion Parameters**: 6 (trans_x, trans_y, trans_z, rot_x, rot_y, rot_z)
- **HRF Model**: SPM
- **Drift Model**: Cosine
- **High-Pass Filter**: 0.008 Hz

### Second-Level GLM
- **Design Matrix**: intercept, age, sex, iq
- **Thresholding**: FDR correction (alpha = 0.05)
- **Cluster Threshold**: 10 voxels
- **Display Threshold**: Z > 2.3

---

## Technical Issues Resolved

1. ✅ **Design Matrix Error**: Excluded 'subject_id' column (object type)
2. ✅ **File Path Issues**: Added "sub-" prefix to file paths
3. ✅ **Group Labels**: Corrected from ['HC', 'SZ'] to ['HC', 'AVH+', 'AVH-']
4. ✅ **Missing Values**: Filled 2 missing IQ values with median
5. ✅ **Sex Encoding**: Converted string to numeric (male=1, female=0)
6. ✅ **Threshold API**: Changed `height_threshold` to `alpha` parameter

---

## Files Generated

### Statistical Maps (24 files)
```
/root/fMRI/analysis/second_level/stat_maps/
├── HC_words_vs_baseline_zmap.nii.gz (3.9M)
├── HC_words_vs_baseline_thresholded.nii.gz (87K)
├── HC_sentences_vs_baseline_zmap.nii.gz (3.9M)
├── HC_sentences_vs_baseline_thresholded.nii.gz (129K)
├── HC_reversed_vs_baseline_zmap.nii.gz (3.9M)
├── HC_reversed_vs_baseline_thresholded.nii.gz (103K)
├── AVH+_words_vs_baseline_zmap.nii.gz (3.8M)
├── AVH+_words_vs_baseline_thresholded.nii.gz (38K)
├── AVH+_sentences_vs_baseline_zmap.nii.gz (3.8M)
├── AVH+_sentences_vs_baseline_thresholded.nii.gz (38K)
├── AVH+_reversed_vs_baseline_zmap.nii.gz (3.8M)
├── AVH+_reversed_vs_baseline_thresholded.nii.gz (38K)
├── AVH-_words_vs_baseline_zmap.nii.gz (3.8M)
├── AVH-_words_vs_baseline_thresholded.nii.gz (78K)
├── AVH-_sentences_vs_baseline_zmap.nii.gz (3.8M)
├── AVH-_sentences_vs_baseline_thresholded.nii.gz (73K)
├── AVH-_reversed_vs_baseline_zmap.nii.gz (3.8M)
├── AVH-_reversed_vs_baseline_thresholded.nii.gz (68K)
├── AVH_gt_HC_words_vs_baseline_zmap.nii.gz (4.0M)
├── AVH_gt_HC_words_vs_baseline_thresholded.nii.gz (38K)
├── AVH_gt_HC_sentences_vs_baseline_zmap.nii.gz (3.9M)
├── AVH_gt_HC_sentences_vs_baseline_thresholded.nii.gz (38K)
├── AVH_gt_HC_reversed_vs_baseline_zmap.nii.gz (4.0M)
└── AVH_gt_HC_reversed_vs_baseline_thresholded.nii.gz (38K)
```

### Publication Figures (12 files)
```
/root/fMRI/analysis/second_level/figures/
├── HC_words_vs_baseline_glass_brain.png (652K)
├── HC_sentences_vs_baseline_glass_brain.png (683K)
├── HC_reversed_vs_baseline_glass_brain.png (668K)
├── AVH+_words_vs_baseline_glass_brain.png (670K)
├── AVH+_sentences_vs_baseline_glass_brain.png (659K)
├── AVH+_reversed_vs_baseline_glass_brain.png (697K)
├── AVH-_words_vs_baseline_glass_brain.png (697K)
├── AVH-_sentences_vs_baseline_glass_brain.png (700K)
├── AVH-_reversed_vs_baseline_glass_brain.png (677K)
├── AVH_gt_HC_words_vs_baseline_glass_brain.png (648K)
├── AVH_gt_HC_sentences_vs_baseline_glass_brain.png (598K)
└── AVH_gt_HC_reversed_vs_baseline_glass_brain.png (641K)
```

---

## Scripts Used

1. **`/root/fMRI/scripts/python/run_glm_and_visualizations.py`**
   - Main GLM analysis script (first-level)
   - Generated 213 contrast maps

2. **`/root/fMRI/scripts/python/run_second_level_only.py`**
   - Second-level GLM analysis script
   - Generated all second-level results and figures

---

## Next Steps

1. ✅ First-level GLM analysis - COMPLETE
2. ✅ Second-level GLM analysis - COMPLETE
3. ✅ Generate publication figures - COMPLETE
4. ⏳ Create statistical tables with cluster information
5. ⏳ Prepare Review 1 slides with results

---

## Notes

- All analyses completed successfully
- FDR correction applied to all thresholded maps
- Glass brain plots generated with Z > 2.3 threshold
- Ready for statistical reporting and visualization
- All files saved in BIDS-compliant structure

---

**Analysis completed**: August 18, 2026 at 19:50:49 UTC  
**Total runtime**: ~8 minutes (second-level analysis only)  
**Total outputs**: 36 files (24 statistical maps + 12 publication figures)

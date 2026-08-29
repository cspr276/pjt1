# Second-Level GLM Analysis Status

## Completed Tasks

### First-Level Analysis ✅
- **Status**: COMPLETE
- **Subjects**: 71/71 (100%)
- **Contrast Maps**: 213 total (71 subjects × 3 contrasts)
- **Location**: `/root/fMRI/analysis/first_level/contrast_maps/`
- **Contrasts**:
  - words_vs_baseline
  - sentences_vs_baseline
  - reversed_vs_baseline

### Second-Level Analysis - Partially Complete ⏳
- **Status**: IN PROGRESS
- **Script**: `/root/fMRI/scripts/python/run_second_level_only.py`

#### Generated Z-Maps (10 files)
1. **One-Sample T-Tests (Group-Level Activation)**:
   - HC_words_vs_baseline_zmap.nii.gz
   - HC_sentences_vs_baseline_zmap.nii.gz
   - HC_reversed_vs_baseline_zmap.nii.gz
   - AVH+_words_vs_baseline_zmap.nii.gz
   - AVH+_sentences_vs_baseline_zmap.nii.gz
   - AVH+_reversed_vs_baseline_zmap.nii.gz
   - AVH-_words_vs_baseline_zmap.nii.gz
   - AVH-_sentences_vs_baseline_zmap.nii.gz
   - AVH-_reversed_vs_baseline_zmap.nii.gz

2. **Two-Sample T-Tests (Group Comparisons)**:
   - AVH_gt_HC_words_vs_baseline_zmap.nii.gz (1 of 3 complete)

#### Remaining Tasks
- Complete two-sample t-tests for:
  - sentences_vs_baseline (AVH+ > HC)
  - reversed_vs_baseline (AVH+ > HC)
- Generate thresholded maps (FDR corrected)
- Generate publication figures (glass brain plots)
- Create statistical tables

## Issues Fixed

### 1. Design Matrix Error ✅
- **Problem**: 'subject_id' column (object type) included in design matrix
- **Solution**: Created `design_matrix_numeric` excluding object-type columns

### 2. File Path Issues ✅
- **Problem**: Missing "sub-" prefix in file paths
- **Solution**: Updated to `sub-{subject_id}_{contrast}_zmap.nii.gz`

### 3. Group Labels ✅
- **Problem**: Used 'SZ' instead of 'AVH+', 'AVH-'
- **Solution**: Updated to correct groups: ['HC', 'AVH+', 'AVH-']

### 4. Missing Values ✅
- **Problem**: 2 subjects missing IQ values
- **Solution**: Fill with median: `df['iq'].fillna(df['iq'].median())`

### 5. Sex Encoding ✅
- **Problem**: Sex column was string ('male'/'female')
- **Solution**: Encode as numeric: `(df['sex'] == 'male').astype(int)`

### 6. Threshold API Error ✅
- **Problem**: `threshold_stats_img()` used `height_threshold` parameter
- **Solution**: Changed to `alpha=0.05` parameter

## Participant Groups

| Group | Count | Description |
|-------|-------|-------------|
| HC | 25 | Healthy Controls |
| AVH+ | 23 | Schizophrenia with Auditory Hallucinations |
| AVH- | 23 | Schizophrenia without Auditory Hallucinations |
| **Total** | **71** | |

## Analysis Parameters

- **TR**: 2.0 seconds
- **Smoothing**: FWHM = 5.0mm
- **Motion Parameters**: 6 (trans_x, trans_y, trans_z, rot_x, rot_y, rot_z)
- **HRF Model**: SPM
- **Drift Model**: Cosine
- **High-Pass Filter**: 0.008 Hz
- **Cluster Threshold**: Z > 2.3, cluster p < 0.05 (FDR corrected)
- **Design Matrix**: intercept, age, sex, iq

## Output Locations

- **First-Level Maps**: `/root/fMRI/analysis/first_level/contrast_maps/`
- **Second-Level Maps**: `/root/fMRI/analysis/second_level/stat_maps/`
- **Figures**: `/root/fMRI/analysis/second_level/figures/`
- **Tables**: `/root/fMRI/analysis/second_level/tables/`
- **Logs**: `/root/fMRI/logs/`

## Next Steps

1. Wait for second-level analysis to complete (currently running)
2. Verify all z-maps are generated (expected: 12 total)
3. Generate publication figures using `generate_figures_only.py`
4. Create statistical tables with cluster information
5. Prepare Review 1 slides with results

## Notes

- The script is currently running in the background
- All unthresholded z-maps are being saved successfully
- Thresholding step may fail but unthresholded maps are still usable
- Can generate figures from unthresholded maps with manual thresholding

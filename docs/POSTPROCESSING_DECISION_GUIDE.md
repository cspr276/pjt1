# Postprocessing Decision Guide
## Step-by-Step Decision Making for Post-fMRIPrep Pipeline

**Purpose**: Guide you through each postprocessing decision with clear criteria  
**Use**: Follow this document sequentially during implementation  
**Dataset**: N=71 subjects (25 HC, 23 AVH-, 23 AVH+)  
**Goal**: Prepare fMRI data for β-VAE + BrainGNN analysis  

---

## OVERVIEW: What Needs to Be Done

fMRIPrep has completed standard preprocessing. You now need to:

```
fMRIPrep Output (MNI space, unsmoothed)
    ↓
STEP 1: Quality Control Decision
    ↓
STEP 2: Data Splitting (Train/Val/Test)
    ↓
STEP 3: Spatial Smoothing
    ↓
STEP 4: Temporal Filtering
    ↓
STEP 5: Confound Regression
    ↓
STEP 6: Normalization
    ↓
STEP 7: ROI Extraction
    ↓
Ready for VAE Input
```

---

## STEP 1: QUALITY CONTROL DECISIONS

### Decision 1.1: Should I exclude any subjects?

**Check**: Motion parameters (FD - Framewise Displacement)

**How to check**:
```python
# For each subject, compute:
mean_FD = mean(framewise_displacement)
pct_high_motion = (FD > 0.5mm).sum() / total_timepoints

# Thresholds:
mean_FD > 0.3 mm → CONSIDER EXCLUSION
pct_high_motion > 30% → CONSIDER EXCLUSION
```

**Decision criteria**:
| Condition | Action | Reason |
|-----------|--------|--------|
| mean_FD < 0.2 mm | ✅ Include | Excellent quality |
| 0.2 ≤ mean_FD < 0.3 mm | ✅ Include | Good quality |
| 0.3 ≤ mean_FD < 0.5 mm | ⚠️ Flag for review | Moderate quality, may affect results |
| mean_FD ≥ 0.5 mm | ❌ Consider exclusion | Poor quality, high motion artifacts |
| > 40% timepoints with FD > 0.5 mm | ❌ Exclude | Severe motion contamination |

**Your dataset status**:
- ✅ Quality metrics already computed in `/root/fMRI/quality_analysis/metrics/quality_metrics.csv`
- ✅ All 71 subjects passed initial QC
- ⚠️ Need to check FD/DVARS specifically (not yet extracted)

**DECISION FOR YOUR DATASET**:
```
ACTION: Extract FD and DVARS from confounds files
METHOD: Load confounds_timeseries.tsv, compute mean_FD per subject
CRITERIA: Exclude if mean_FD > 0.5 mm OR > 40% high-motion volumes
EXPECTED: Most subjects should pass (fMRIPrep already QC'd)
```

---

### Decision 1.2: Should I censor (remove) high-motion timepoints?

**Options**:
1. **No censoring** - Keep all timepoints, regress out motion
2. **Spike regression** - Add binary regressors for high-motion volumes
3. **Scrubbing** - Remove high-motion volumes entirely

**Decision criteria**:
| Condition | Recommendation | Why |
|-----------|----------------|-----|
| < 10% high-motion volumes | Option 1 (no censoring) | Motion regression sufficient |
| 10-30% high-motion volumes | Option 2 (spike regression) | Balance data retention vs quality |
| > 30% high-motion volumes | Option 3 (scrubbing) OR exclude subject | Too much contamination |

**DECISION FOR YOUR DATASET**:
```
CHOICE: Option 1 (no censoring) + motion regression
RATIONALE: 
  - VAE needs continuous timepoints (temporal structure)
  - Motion regression handles artifacts
  - Small N=71 means every timepoint valuable
  - If specific subject has >30% high-motion, flag for review
```

---

## STEP 2: DATA SPLITTING DECISIONS

### Decision 2.1: How to split the data?

**Critical rule**: Subject-wise split (NOT voxel-wise or timepoint-wise)

**Why**: 
- Voxel/timepoint splits cause ~3.4× data leakage with N=71
- Subject correlations violate independence
- Must split by subject to get unbiased estimates

**Decision criteria**:
| Split Type | Train | Val | Test | Use Case |
|-----------|-------|-----|------|----------|
| 60/20/20 | 43 | 14 | 14 | Larger test set (more reliable) |
| **70/15/15** | **50** | **11** | **10** | **RECOMMENDED** (balance) |
| 80/10/10 | 57 | 7 | 7 | More training data (riskier with small N) |

**DECISION FOR YOUR DATASET**:
```
CHOICE: 70/15/15 split (50 train, 11 val, 10 test)
RATIONALE:
  - Standard split ratio
  - Maintains reasonable test set size (10 subjects)
  - Validation set large enough for hyperparameter tuning (11 subjects)
  - Training set maximized while leaving adequate holdout
```

---

### Decision 2.2: Should I stratify by group?

**Options**:
1. **Random split** - Randomly assign subjects to train/val/test
2. **Stratified split** - Maintain group proportions (HC/AVH-/AVH+)

**Decision criteria**:
| Condition | Recommendation |
|-----------|----------------|
| Balanced groups (equal N per group) | Random OR stratified (both OK) |
| Imbalanced groups | Stratified (maintains proportions) |
| Multi-class classification | Stratified (critical) |

**Your dataset**:
- HC: 25 subjects (35%)
- AVH-: 23 subjects (32%)
- AVH+: 23 subjects (32%)
- Groups are roughly balanced

**DECISION FOR YOUR DATASET**:
```
CHOICE: Stratified split
RATIONALE:
  - Multi-class task (3 groups)
  - Ensures each split has representative samples
  - Prevents unlucky split (e.g., all AVH+ in test)
  
EXPECTED DISTRIBUTION:
  Train (50): ~17 HC, ~17 AVH-, ~16 AVH+
  Val (11): ~4 HC, ~4 AVH-, ~3 AVH+
  Test (10): ~4 HC, ~3 AVH-, ~3 AVH+
```

---

### Decision 2.3: When to split?

**Options**:
1. **Split first, then preprocess** - Split → fit preprocessing on train → apply to all
2. **Preprocess all, then split** - Preprocess all data → split (WRONG - causes leakage)

**DECISION FOR YOUR DATASET**:
```
CHOICE: Split FIRST (Option 1)
CRITICAL: This is non-negotiable for preventing leakage

ORDER OF OPERATIONS:
  1. Load all 71 subjects
  2. Create stratified split (save indices)
  3. Fit all preprocessing on TRAINING SET ONLY
  4. Apply fitted preprocessing to val/test
  5. Never refit on val/test
```

---

## STEP 3: SPATIAL SMOOTHING DECISIONS

### Decision 3.1: Should I smooth?

**fMRIPrep output**: Unsmoothed BOLD in MNI space

**Options**:
1. **No smoothing** - Use raw fMRIPrep output
2. **Smooth before analysis** - Apply Gaussian smoothing

**Decision criteria**:
| Analysis Type | Recommendation | FWHM |
|---------------|----------------|------|
| Voxel-wise analysis | Smooth (reduce noise) | 5-8 mm |
| ROI-based analysis | Smooth (optional) | 4-6 mm |
| VAE input | Smooth (recommended) | 5 mm |
| Already smoothed in GLM | Check if needed | N/A |

**Your situation**:
- ✅ GLM analysis used 5mm smoothing
- ❌ fMRIPrep BOLD is unsmoothed
- ✅ VAE benefits from smoothing (reduces high-freq noise)

**DECISION FOR YOUR DATASET**:
```
CHOICE: Apply 5mm FWHM Gaussian smoothing
RATIONALE:
  - Matches GLM smoothing (consistency)
  - Reduces high-frequency noise
  - Preserves spatial structure VAE needs
  - Standard for fMRI analysis
  
PARAMETERS:
  FWHM = 5.0 mm
  σ (sigma) = FWHM / 2.355 ≈ 2.12 mm
  Apply to: All subjects (train/val/test)
  Fitting: NOT needed (fixed operation, no parameters to learn)
```

---

### Decision 3.2: What smoothing kernel size?

**Options**: FWHM = 4, 5, 6, 8 mm

**Decision criteria**:
| FWHM | Effect | Use Case |
|------|--------|----------|
| 4 mm | Minimal smoothing | High-res analysis, preserves detail |
| **5 mm** | **Moderate smoothing** | **STANDARD for fMRI** |
| 6 mm | More smoothing | Noisy data, lower-res |
| 8 mm | Heavy smoothing | Very noisy data, loses detail |

**DECISION FOR YOUR DATASET**:
```
CHOICE: 5 mm FWHM
RATIONALE:
  - Standard choice for 2mm³ resolution
  - Matches your GLM analysis (consistency)
  - Balances noise reduction vs spatial precision
  - Published work uses 5mm for similar tasks
```

---

## STEP 4: TEMPORAL FILTERING DECISIONS

### Decision 4.1: Should I apply high-pass filtering?

**fMRIPrep**: Does NOT apply high-pass filtering by default

**Your GLM**: Used cosine drift model (0.008 Hz high-pass)

**Options**:
1. **No filtering** - Use raw timeseries
2. **High-pass filter** - Remove slow drifts (< 0.008 Hz)
3. **Band-pass filter** - Keep specific frequency range

**Decision criteria**:
| Analysis Type | Recommendation | Cutoff |
|---------------|----------------|--------|
| GLM analysis | High-pass (drift model) | 0.008 Hz |
| Functional connectivity | High-pass | 0.008-0.01 Hz |
| VAE input | High-pass | 0.008 Hz |
| Dynamic FC | Band-pass (optional) | 0.01-0.1 Hz |

**DECISION FOR YOUR DATASET**:
```
CHOICE: High-pass filter at 0.008 Hz
RATIONALE:
  - Removes scanner drift and physiological noise
  - Matches GLM cutoff (consistency)
  - Standard for fMRI connectivity analysis
  - VAE benefits from drift removal
  
IMPLEMENTATION:
  Method: Cosine basis functions (Discrete Cosine Transform)
  Cutoff: 0.008 Hz (128 second period)
  TR: 2.0 seconds
  Apply to: All subjects
  
CRITICAL: Fit basis on TRAINING SET, apply to all
  - Compute cosine basis from training data
  - Use same basis for val/test (no refitting)
```

---

### Decision 4.2: Should I apply low-pass filtering?

**Options**:
1. **No low-pass** - Keep all frequencies
2. **Low-pass filter** - Remove high frequencies (> 0.1 Hz)

**Decision criteria**:
| Condition | Recommendation |
|-----------|----------------|
| Task-based fMRI | No low-pass (task frequencies important) |
| Resting-state FC | Low-pass optional (0.1 Hz cutoff) |
| VAE input | No low-pass (let VAE learn temporal structure) |

**DECISION FOR YOUR DATASET**:
```
CHOICE: NO low-pass filtering
RATIONALE:
  - Task-based design (speech perception)
  - Task-related signal may be in higher frequencies
  - VAE should learn relevant temporal dynamics
  - Low-pass may remove task-relevant information
```

---

## STEP 5: CONFOUND REGRESSION DECISIONS

### Decision 5.1: Which confounds to regress?

**Available from fMRIPrep**:
- ✅ Motion parameters (6 DOF: trans_x/y/z, rot_x/y/z)
- ✅ Motion derivatives (can compute)
- ✅ Motion squares (can compute)
- ✅ CSF signal (from tissue segmentation)
- ✅ White matter signal (from tissue segmentation)
- ✅ Global signal (can compute from brain mask)
- ⚠️ aCompCor (not extracted, would need ICA)

**Decision criteria**:

| Confound Type | Include? | Reason |
|---------------|----------|--------|
| **Motion (6 DOF)** | ✅ YES | Bulk head motion, non-neural |
| **Motion derivatives** | ✅ YES | Motion acceleration, captures artifacts |
| **Motion squares** | ✅ YES | Nonlinear motion effects |
| **CSF signal** | ✅ YES | Physiological noise (cardiac, respiratory) |
| **White matter** | ✅ YES | WM noise, non-neural |
| **Global signal** | ❌ NO | Controversial, may remove neural signal |
| **aCompCor** | ⚠️ OPTIONAL | Better than CSF/WM, but requires extraction |

**DECISION FOR YOUR DATASET**:
```
CHOICE: Motion (6 DOF) + derivatives + squares + CSF + WM
TOTAL: 6 + 6 + 6 + 1 + 1 = 20 confound regressors

RATIONALE:
  - Motion: Standard, removes bulk artifacts
  - Derivatives + squares: Captures nonlinear motion effects
  - CSF + WM: Removes physiological noise
  - Global signal: EXCLUDED (controversial, may remove neural signal)
  - aCompCor: Not included (not extracted, CSF/WM sufficient)

IMPLEMENTATION:
  1. Load confounds_timeseries.tsv
  2. Extract: trans_x, trans_y, trans_z, rot_x, rot_y, rot_z
  3. Compute derivatives: np.gradient(motion, axis=0)
  4. Compute squares: motion ** 2
  5. Extract CSF: mean(BOLD[CSF_mask])
  6. Extract WM: mean(BOLD[WM_mask])
  7. Concatenate: (240 timepoints, 20 confounds)
```

---

### Decision 5.2: How to regress confounds?

**Options**:
1. **Ordinary Least Squares (OLS)** - Standard linear regression
2. **Robust regression (RLM)** - Handles outliers better
3. **L2-regularized regression (Ridge)** - Prevents overfitting

**Decision criteria**:
| Condition | Recommendation |
|-----------|----------------|
| Clean data, no outliers | OLS |
| Some outliers (motion spikes) | RLM |
| Many confounds, small N | Ridge |

**DECISION FOR YOUR DATASET**:
```
CHOICE: Robust Linear Model (RLM)
RATIONALE:
  - Motion data often has outliers (spike volumes)
  - RLM downweights outliers automatically
  - More stable than OLS for fMRI
  - Available in statsmodels: sm.RLM()
  
IMPLEMENTATION:
  For each subject:
    design_matrix = [1s | motion | d_motion | motion² | CSF | WM]
    model = sm.RLM(BOLD, design_matrix)
    results = model.fit()
    BOLD_clean = BOLD - design_matrix @ results.params
```

---

### Decision 5.3: When to fit confound regression?

**Options**:
1. **Fit on all data, then split** - WRONG (leakage)
2. **Fit on train, apply to all** - CORRECT (no leakage)

**DECISION FOR YOUR DATASET**:
```
CHOICE: Fit on TRAINING SET ONLY
CRITICAL: This prevents leakage

PROCEDURE:
  1. Split subjects into train/val/test
  2. For TRAINING subjects:
     - Fit confound regression (learn β coefficients)
     - Save β coefficients
  3. For VAL/TEST subjects:
     - Load β coefficients from training
     - Apply: BOLD_clean = BOLD - design_matrix @ β_train
     - DO NOT refit on val/test
  
WHY: Fitting on all data lets test data influence preprocessing
```

---

## STEP 6: NORMALIZATION DECISIONS

### Decision 6.1: Should I normalize the data?

**Options**:
1. **No normalization** - Use raw values
2. **Z-score normalization** - Mean=0, Std=1
3. **Min-max scaling** - Scale to [0, 1]

**Decision criteria**:
| Analysis Type | Recommendation |
|---------------|----------------|
| VAE input | Z-score (standard practice) |
| GNN input | Z-score (helps convergence) |
| FC computation | Z-score (makes correlations comparable) |

**DECISION FOR YOUR DATASET**:
```
CHOICE: Z-score normalization
RATIONALE:
  - VAE assumes standardized inputs
  - Helps VAE training stability
  - Makes FC correlations comparable across subjects
  - Standard for deep learning
  
IMPLEMENTATION:
  For each voxel/ROI:
    z = (x - mean) / std
```

---

### Decision 6.2: When to compute normalization statistics?

**Options**:
1. **Compute on all data** - WRONG (leakage)
2. **Compute on train, apply to all** - CORRECT

**DECISION FOR YOUR DATASET**:
```
CHOICE: Compute mean/std on TRAINING SET ONLY
CRITICAL: Prevents leakage

PROCEDURE:
  1. Compute mean_train, std_train from training BOLD
  2. For training subjects:
     BOLD_norm = (BOLD - mean_train) / std_train
  3. For val/test subjects:
     BOLD_norm = (BOLD - mean_train) / std_train
     (use training statistics, NOT their own)
  
WHY: Test data should not influence preprocessing
```

---

### Decision 6.3: Normalize globally or per-subject?

**Options**:
1. **Global normalization** - One mean/std across all training subjects
2. **Per-subject normalization** - Each subject normalized separately

**Decision criteria**:
| Condition | Recommendation |
|-----------|----------------|
| Comparing across subjects | Global (preserves relative differences) |
| Within-subject analysis only | Per-subject (removes individual differences) |
| Classification task | Global (maintains group differences) |

**DECISION FOR YOUR DATASET**:
```
CHOICE: Global normalization (across training subjects)
RATIONALE:
  - Classification task (HC vs AVH- vs AVH+)
  - Want to preserve between-subject differences
  - Per-subject normalization would remove group signal
  - Global normalization maintains disease-related variance
  
IMPLEMENTATION:
  1. Stack all training BOLD: (50 subjects × voxels × timepoints)
  2. Compute global_mean, global_std
  3. Apply to all subjects: (BOLD - global_mean) / global_std
```

---

## STEP 7: ROI EXTRACTION DECISIONS

### Decision 7.1: Which atlas to use?

**Options**:
- Schaefer-200 (7 or 17 networks)
- Schaefer-400 (7 or 17 networks)
- AAL (116 regions)
- Harvard-Oxford (96 regions)
- Custom atlas

**Decision criteria** (for N=71):
| Atlas | # ROIs | DoF Ratio (N/ROIs) | Recommendation |
|-------|--------|-------------------|----------------|
| Schaefer-200 | 200 | 0.36 | ✅ PRIMARY |
| Schaefer-400 | 400 | 0.18 | ⚠️ Sensitivity only |
| AAL | 116 | 0.61 | ✅ Alternative |
| Harvard-Oxford | 96 | 0.74 | ✅ Alternative |

**DECISION FOR YOUR DATASET**:
```
CHOICE: Schaefer-200 (7 networks)
RATIONALE:
  - Statistical power: 71/200 = 0.36 DoF ratio (acceptable)
  - Schaefer-400: 71/400 = 0.18 (under-powered)
  - Functional parcellation (matches functional data)
  - Widely used in fMRI connectivity studies
  - 7 networks align with known brain systems
  
SENSITIVITY ANALYSIS: Re-run with Schaefer-400 to test robustness
```

---

### Decision 7.2: How to extract ROI signals?

**Options**:
1. **Mean** - Average BOLD across all voxels in ROI
2. **Median** - Median BOLD (robust to outliers)
3. **PCA** - First principal component (captures variance)
4. **All voxels** - Keep all voxels (too high-dimensional)

**Decision criteria**:
| Method | Pros | Cons | Use Case |
|--------|------|------|----------|
| Mean | Simple, standard | Sensitive to outliers | ✅ RECOMMENDED |
| Median | Robust | Less common | Noisy data |
| PCA | Captures variance | More complex | High-variance ROIs |

**DECISION FOR YOUR DATASET**:
```
CHOICE: Mean extraction
RATIONALE:
  - Standard approach in fMRI connectivity
  - Simple and interpretable
  - Reduces dimensionality (350k voxels → 200 ROIs)
  - Preserves temporal dynamics (240 timepoints per ROI)
  
IMPLEMENTATION:
  For each subject:
    For each of 200 ROIs:
      mask = atlas == ROI_id
      ROI_timeseries = mean(BOLD[mask, :], axis=0)
    Result: (200 ROIs, 240 timepoints)
```

---

### Decision 7.3: Should I apply atlas in MNI or native space?

**Options**:
1. **MNI space** - Transform atlas to MNI, extract from MNI BOLD
2. **Native space** - Transform BOLD to native T1w, extract there

**Decision criteria**:
| Space | Pros | Cons |
|-------|------|------|
| MNI | Standard, comparable across subjects | Registration errors |
| Native | Subject-specific anatomy | Need to transform atlas per subject |

**Your situation**:
- ✅ fMRIPrep provides BOLD in MNI space
- ✅ Schaefer atlas available in MNI space
- ✅ All subjects already in same space

**DECISION FOR YOUR DATASET**:
```
CHOICE: MNI space
RATIONALE:
  - fMRIPrep already normalized to MNI
  - Schaefer atlas available in MNI
  - No additional transformations needed
  - Standard for group-level analysis
  
IMPLEMENTATION:
  1. Load Schaefer-200 atlas (MNI 2mm resolution)
  2. Load preprocessed BOLD (MNI 2mm³)
  3. Extract ROI signals directly (no transforms)
```

---

## SUMMARY: COMPLETE POSTPROCESSING PIPELINE

### Final Decision Summary

| Step | Decision | Parameters | Fitting |
|------|----------|------------|----------|
| **1. QC** | Include all subjects (check FD) | mean_FD < 0.5 mm | N/A |
| **2. Split** | 70/15/15 subject-wise, stratified | 50/11/10 subjects | N/A |
| **3. Smoothing** | Gaussian, 5mm FWHM | σ = 2.12 mm | Fixed (no fitting) |
| **4. High-pass** | Cosine basis, 0.008 Hz | TR=2.0s, cutoff=0.008 Hz | Fit on train |
| **5. Confounds** | Motion+deriv+sq+CSF+WM | 20 regressors | Fit on train |
| **6. Z-score** | Global normalization | mean/std | Fit on train |
| **7. ROI** | Schaefer-200, mean extraction | 200 ROIs | Fixed (atlas) |

---

### Implementation Order (CRITICAL)

```
STEP 0: Load all 71 subjects
    ↓
STEP 1: Create stratified split (save indices!)
    ↓
STEP 2: Apply smoothing (fixed, all subjects)
    ↓
STEP 3: Fit high-pass filter on TRAINING SET
    ↓
STEP 4: Apply high-pass to all (using train basis)
    ↓
STEP 5: Fit confound regression on TRAINING SET
    ↓
STEP 6: Apply confound regression to all (using train β)
    ↓
STEP 7: Fit z-score on TRAINING SET
    ↓
STEP 8: Apply z-score to all (using train mean/std)
    ↓
STEP 9: Extract ROI timeseries (Schaefer-200)
    ↓
OUTPUT: Cleaned ROI timeseries (200 × 240) per subject
```

---

### Quality Checks After Each Step

**After smoothing**:
- [ ] Visual check: Image looks smoother (less noisy)
- [ ] No NaN or inf values
- [ ] Mean signal similar to pre-smoothing

**After high-pass filtering**:
- [ ] Low-frequency drift removed (plot timeseries)
- [ ] Mean ≈ 0 (drift removed)
- [ ] No edge artifacts at start/end

**After confound regression**:
- [ ] Residuals uncorrelated with confounds (check r < 0.1)
- [ ] No NaN values
- [ ] Variance reduced (noise removed)

**After z-score**:
- [ ] Mean ≈ 0 (across training set)
- [ ] Std ≈ 1 (across training set)
- [ ] Val/test have similar stats (using train normalization)

**After ROI extraction**:
- [ ] Shape: (200, 240) per subject
- [ ] No NaN or inf
- [ ] Timeseries look reasonable (no huge spikes)

---

## RED FLAGS (Stop and Investigate)

❌ **If you see**:
- NaN or inf values → Numerical instability
- Mean/std very different from expected → Preprocessing error
- Test accuracy > 95% → Likely leakage
- Val loss much higher than train → Overfitting
- ROI timeseries all zeros → Atlas misalignment
- Correlation with confounds > 0.3 → Confound regression failed

---

## NEXT STEPS AFTER POSTPROCESSING

Once postprocessing complete, you'll have:
- ✅ Cleaned ROI timeseries (200 ROIs × 240 timepoints) per subject
- ✅ Train/val/test split (50/11/10 subjects)
- ✅ All preprocessing fitted on train only (no leakage)

**Then proceed to**:
1. VAE training (on training set)
2. Latent encoding (freeze VAE, encode all)
3. Functional connectivity computation
4. Graph construction
5. GNN training

---

## QUICK REFERENCE: PARAMETERS

```python
# Data split
TRAIN_SIZE = 50  # subjects
VAL_SIZE = 11    # subjects
TEST_SIZE = 10   # subjects
STRATIFIED = True

# Smoothing
FWHM = 5.0  # mm
SIGMA = FWHM / 2.355  # ≈ 2.12 mm

# High-pass filter
TR = 2.0  # seconds
HIGH_PASS_CUTOFF = 0.008  # Hz (128 second period)

# Confounds
N_MOTION = 6  # DOF
N_MOTION_DERIV = 6
N_MOTION_SQUARE = 6
N_CSF = 1
N_WM = 1
TOTAL_CONFOUNDS = 20

# ROI extraction
ATLAS = "Schaefer-200"
N_ROIS = 200
EXTRACTION_METHOD = "mean"

# Normalization
NORM_METHOD = "z-score"
NORM_SCOPE = "global"  # across training subjects
```

---

## DOCUMENT STATUS

✅ **Complete** - All postprocessing decisions specified  
✅ **Actionable** - Clear criteria for each decision  
✅ **Ordered** - Implementation sequence provided  
✅ **Quality-checked** - Verification steps included  

**Use this document**: During implementation to make decisions step-by-step

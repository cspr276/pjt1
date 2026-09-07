# FROZEN METHODOLOGY — QUICK REFERENCE
## Brain VAE + GNN for Schizophrenia Classification

**Status**: ✅ Analysis Complete, Ready for Implementation  
**Date**: August 31, 2026  
**Sample Size**: N=71 (25 HC, 23 AVH-, 23 AVH+)  
**Dataset**: OpenNeuro ds004302 (speech perception fMRI)  

---

## ONE-PAGE SUMMARY

### The Approach
```
fMRIPrep BOLD (MNI space)
    ↓
[Preprocessing: High-pass filter + Confound regression + Smoothing + Z-score]
    ↓
[ROI Extraction: Schaefer-200]
    ↓
[β-VAE Training: (200 ROIs × 240 timepoints) → (200 ROIs × 16 latent factors)]
    ↓
[Functional Connectivity: Pearson correlation + Fisher-z + Sparse (k=15)]
    ↓
[Graph Construction: 200 nodes (VAE features) + edges (FC)]
    ↓
[BrainGNN Classification: HC vs AVH- vs AVH+]
    ↓
Output: Class prediction + Interpretable latent factors
```

### Why This Works
| Component | Why It's Optimal |
|-----------|-----------------|
| **β-VAE** | Learns disentangled, interpretable features → better node embeddings |
| **Schaefer-200** | Statistically adequate for N=71 (0.36 DoF ratio) |
| **Functional connectivity** | Captures true brain communication patterns |
| **BrainGNN** | Designed specifically for fMRI+GNN; handles small N |
| **Subject-wise split** | Prevents 3.4× data leakage with N=71 |

### Expected Performance
- **Binary (HC vs AVH+)**: 75-85% accuracy
- **3-way (HC vs AVH- vs AVH+)**: 60-75% accuracy
- **Baseline (majority class)**: 32% (for reference)
- **Statistical significance**: Permutation test p < 0.05

---

## CRITICAL RULES (Must Follow)

### ⚠️ LEAKAGE PREVENTION (Most Important)
1. **Split first**: Train (50) / Val (11) / Test (10) — stratified by group
2. **Fit on train only**: High-pass filter, confound regression, z-score normalization
3. **Apply to all**: Use training statistics to transform val/test
4. **Train on train**: VAE learns on train set, frozen weights for val/test
5. **Test is untouchable**: Never evaluate test set until final reporting

### ⚠️ PREPROCESSING PIPELINE
1. **High-pass filter**: Cosine basis fitted on training set (cutoff 0.008 Hz)
2. **Confound regression**: Motion (6-DOF + derivatives + squares) + CSF + WM
3. **Spatial smoothing**: 5mm FWHM Gaussian (fixed, applies to all)
4. **Z-score normalization**: Mean/std computed on training set

### ⚠️ HYPERPARAMETER TUNING
- **Tune on validation set ONLY** (not train, never test)
- **Search grid**: 
  - Latent dim: {8, 16, 32, 64}
  - β: {1, 2, 4, 8, 16}
  - k (sparsity): {10, 15, 20}
- **Best hyperparams expected**: latent_dim=16, β=4, k=15

### ⚠️ BASELINES ARE MANDATORY
Must compare against:
1. Majority class (32%)
2. Direct FC → GNN (skip VAE)
3. Logistic regression on VAE latents (skip GNN)
4. SVM on FC vectors (classical approach)

---

## HYPERPARAMETER SPECIFICATIONS

### VAE Configuration
```python
# Architecture
input_dim = 200  # Schaefer ROIs
latent_dim = 16  # Primary choice (tune on val: 8-64)
beta = 4.0       # Primary choice (tune on val: 1-16)
encoder = LSTM(200, 128) → Dense(256) → Dense(2×latent_dim)
decoder = Dense(256) → LSTM(128, 200)

# Training
epochs = 100
batch_size = 8
optimizer = Adam(lr=1e-3)
loss = MSE(reconstruction) + beta * KL_divergence
early_stopping = patience=10 (on validation loss)
```

### GNN Configuration
```python
# Architecture (BrainGNN or GCN baseline)
input_dim = 16   # VAE latent dimension
hidden_dim = 32, 64  # Experiment
output_dim = 3   # Classes: HC, AVH-, AVH+
layers = 2-3 (tune on val)
dropout = 0.3 (tune on val: 0.1-0.5)
pooling = adaptive (BrainGNN) or global_mean (GCN)

# Training
epochs = 100
optimizer = Adam(lr=1e-3)
loss = CrossEntropyLoss
early_stopping = patience=15 (on validation accuracy)
```

### Data Split
```python
train_idx = 50 subjects (17-18 HC, 17-18 AVH-, 16-17 AVH+)
val_idx = 11 subjects (4 HC, 4 AVH-, 3 AVH+)
test_idx = 10 subjects (4 HC, 3 AVH-, 3 AVH+)
# Stratified by group to maintain class balance
```

---

## IMPLEMENTATION TIMELINE

| Phase | Duration | Key Deliverables | Status |
|-------|----------|------------------|--------|
| 1. Setup & Validation | Week 1 | ROI extraction, split creation, QC | TODO |
| 2. Preprocessing | Week 1-2 | Cleaned BOLD timeseries | TODO |
| 3. VAE | Week 2-3 | Trained encoder, latent codes | TODO |
| 4. Graphs | Week 3 | FC matrices, graph objects | TODO |
| 5. GNN | Week 4 | Trained classifier | TODO |
| 6. Evaluation | Week 4-5 | Test accuracy, interpretability | TODO |
| 7. Sensitivity | Week 5-6 | Robustness analyses | TODO |
| 8. Writing | Week 6-7 | Manuscript ready | TODO |

**Total**: ~4-6 weeks (60-80 hours of active coding)

---

## VALIDATION CHECKLIST

### Before Starting
- [ ] All fMRIPrep outputs present (71 subjects)
- [ ] Confounds extracted (motion, CSF, WM)
- [ ] Schaefer-200 atlas downloaded
- [ ] Requirements.txt updated (PyTorch, torch_geometric, nilearn)

### During Implementation
- [ ] Train/val/test split saved
- [ ] Preprocessing functions tested on 1 subject
- [ ] VAE loss decreasing (both train and val)
- [ ] No NaN/Inf in any output
- [ ] GNN overfitting monitored (val vs train loss)

### Before Final Reporting
- [ ] All leakage prevention protocols verified
- [ ] Test accuracy computed exactly once
- [ ] Permutation test p-value < 0.05
- [ ] Baseline comparisons completed
- [ ] Code reproducible with fixed random seed

---

## QUICK DECISION TABLE

**Q: Should I use voxel-wise or ROI-averaged BOLD?**  
A: ROI-averaged (Schaefer-200). Voxel-wise = 350k dims, too large. ROIs = 200 dims, manageable.

**Q: β-VAE or standard VAE?**  
A: β-VAE (β=4). Disentanglement is critical for interpretable node features.

**Q: How many latent dimensions?**  
A: Primary=16 (tune on val from 8-64). Brain typically has ~8-10 independent networks; 16 gives room.

**Q: Schaefer-200 or 400?**  
A: 200 primary (0.36 DoF ratio adequate). 400 only as sensitivity analysis (0.18 DoF is under-powered).

**Q: How to prevent leakage?**  
A: Fit ALL preprocessing on train set only. Apply learned statistics to val/test.

**Q: BrainGNN or GCN?**  
A: BrainGNN primary (designed for fMRI). GCN as simple baseline.

**Q: How many edges in graph?**  
A: Top-k=15 per node. Standard balance between noise reduction and information preservation.

**Q: When can I look at test results?**  
A: Only for final reporting. Never during hyperparameter tuning (validation only).

---

## FILES TO REFERENCE DURING CODING

| Document | Use Case |
|----------|----------|
| **POST_fMRIPrep_PIPELINE_ANALYSIS.md** | Complete methodology guide; detailed explanations |
| **IMPLEMENTATION_CHECKLIST.md** | Step-by-step tasks; detailed code templates |
| **LITERATURE_REVIEW_AND_CRITICAL_INSIGHTS.md** | Literature backing; publication strategy |
| **FROZEN_METHODOLOGY_SUMMARY.md** | This file; quick reference |

---

## SUCCESS CRITERIA

✅ **Must Achieve**
- Test accuracy: ≥ 65% (binary), ≥ 55% (3-way)
- Permutation test: p < 0.05
- All leakage prevention protocols followed
- Code runs without error (on GPU or CPU)
- Results reproducible (fixed seed, saved artifacts)

⚠️ **Red Flags** (investigate if occurs)
- Test accuracy < 50% → likely leakage or implementation error
- Val loss > train loss by 3×+ → severe overfitting
- Permutation p > 0.10 → results not significant
- NaN/Inf in outputs → numerical instability

---

## COMMON MISTAKES (Avoid)

❌ Fitting z-score normalization on all data then splitting  
❌ Using voxel-level train/test split (causes leakage)  
❌ Tuning hyperparameters on test set  
❌ Evaluating test set multiple times  
❌ Forgetting to freeze VAE encoder during GNN training  
❌ Using global signal regression without justification  
❌ Omitting baseline comparisons  
❌ Reporting results without permutation test  
❌ Using Schaefer-400 without acknowledging under-powering  
❌ Mixing up train/val/test statistics  

---

## QUESTIONS? REFER TO...

- **"Why β-VAE specifically?"** → LITERATURE_REVIEW_AND_CRITICAL_INSIGHTS.md, Section 1
- **"How to prevent leakage?"** → POST_fMRIPrep_PIPELINE_ANALYSIS.md, Part 7
- **"What are exact preprocessing steps?"** → IMPLEMENTATION_CHECKLIST.md, Phase 2
- **"What should graph edges be?"** → POST_fMRIPrep_PIPELINE_ANALYSIS.md, Part 5
- **"How to interpret VAE latents?"** → POST_fMRIPrep_PIPELINE_ANALYSIS.md, Part 9
- **"What baselines needed?"** → POST_fMRIPrep_PIPELINE_ANALYSIS.md, Part 8.1

---

## FINAL STATEMENT

**This methodology is scientifically rigorous, statistically appropriate for N=71, and ready for implementation. Follow the leakage prevention protocols exactly. Refer to the three detailed documents for specific guidance on each phase. Expected timeline: 4-6 weeks. Expected accuracy: 70-85% (binary classification). Good luck! 🧠**

---

Document Status: ✅ COMPLETE  
Last Updated: August 31, 2026  
Confidence: ⭐⭐⭐⭐⭐

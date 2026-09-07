# Implementation Checklist: β-VAE + BrainGNN for Schizophrenia fMRI

**Objective**: Learn latent brain representations for auditory hallucination classification  
**Approach**: β-VAE (ROI node features) + Functional connectivity (edges) + BrainGNN  
**Dataset**: N=71 subjects (25 HC, 23 AVH-, 23 AVH+), Schaefer-200 ROIs  

---

## PRE-IMPLEMENTATION VERIFICATION

- [ ] **Check fMRIPrep outputs exist**
  - [ ] 71 subjects × `*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz`
  - [ ] 71 subjects × `*_desc-confounds_timeseries.tsv`
  - [ ] 71 subjects × `*_desc-brain_mask.nii.gz`
  - [ ] 71 subjects × tissue probability maps (GM, WM, CSF)

- [ ] **Load Schaefer-200 atlas**
  - [ ] Download from nilearn or FSL
  - [ ] Verify 200 ROIs in MNI space
  - [ ] Create binary mask for each ROI

- [ ] **Check data integrity**
  - [ ] All BOLD files 240 timepoints (after discarding 5 volumes)
  - [ ] All confound files have same row count as BOLD timepoints
  - [ ] No missing/corrupted files

---

## PHASE 1: DATA SPLIT & INITIAL PREPROCESSING

- [ ] **Create train/val/test split (70/15/15)**
  - [ ] Stratified by group (HC/AVH-/AVH+)
  - [ ] Train: 50 (17/17/16), Val: 11 (4/4/3), Test: 10 (4/3/3)
  - [ ] Save split indices to file (reproducibility)

- [ ] **Extract confounds (training set baseline)**
  - [ ] Load confounds_timeseries.tsv for train subjects
  - [ ] Extract: trans_x, trans_y, trans_z, rot_x, rot_y, rot_z
  - [ ] Compute derivatives: d(motion)/dt
  - [ ] Compute squares: motion^2
  - [ ] Extract CSF signal: mean(BOLD[CSF mask])
  - [ ] Extract WM signal: mean(BOLD[WM mask])
  - [ ] Concatenate: (240 timepoints, 18 confounds per subject)
  - [ ] Save confound matrix

- [ ] **Quality control (pre-filtering)**
  - [ ] Extract FD (framewise displacement) from confounds
  - [ ] Extract DVARS (temporal signal change)
  - [ ] Flag subjects with mean FD > 0.3 mm
  - [ ] Flag subjects with > 30% timepoints FD > 0.5 mm
  - [ ] Document QC results (include in methods)

---

## PHASE 2: PREPROCESSING PIPELINE (Fit on Training Data)

- [ ] **High-pass filter design (cosine basis)**
  - [ ] Sampling rate: fs = 1/TR = 1/2.0 = 0.5 Hz
  - [ ] High-pass cutoff: f_hp = 0.008 Hz
  - [ ] Create cosine basis: n_basis = ceil(2 × TR × max_freq × n_volumes)
  - [ ] Fit basis on training BOLD only
  - [ ] Save basis for application to val/test

- [ ] **Confound regression (fit on train, apply to all)**
  - [ ] For training subjects:
    - [ ] Design matrix: [1s | motion | d_motion | motion^2 | CSF | WM]
    - [ ] Fit robust linear regression (statsmodels.RLM)
    - [ ] Save β coefficients
  - [ ] Apply to val/test subjects:
    - [ ] Compute residuals: BOLD_clean = BOLD - design_matrix @ β_train

- [ ] **Spatial smoothing (fixed operation, apply to all)**
  - [ ] FWHM = 5.0 mm
  - [ ] Use scipy.ndimage.gaussian_filter with σ = FWHM / 2.355
  - [ ] Apply to all subjects (no fitting needed)

- [ ] **Z-score normalization (fit on train, apply to all)**
  - [ ] Compute mean/std of training BOLD
  - [ ] Normalize: BOLD_clean = (BOLD_clean - train_mean) / train_std
  - [ ] Apply same scaling to val/test

- [ ] **Verify preprocessing**
  - [ ] Plot before/after for 1-2 subjects
  - [ ] Check mean ≈ 0, std ≈ 1 (after z-score)
  - [ ] Verify confounds removed (no motion correlation)
  - [ ] Check for NaNs or outliers

---

## PHASE 3: ROI EXTRACTION

- [ ] **Extract ROI-averaged BOLD (Schaefer-200)**
  - [ ] For each subject:
    - [ ] Load preprocessed BOLD (MNI 2mm³)
    - [ ] For each of 200 ROIs:
      - [ ] Mask BOLD to ROI
      - [ ] Compute mean: ROI_ts = mean(BOLD_in_ROI), shape (240,)
    - [ ] Concatenate: (200, 240) matrix per subject
  - [ ] Output: ROI_train (50, 200, 240), ROI_val (11, 200, 240), ROI_test (10, 200, 240)

- [ ] **Sanity checks**
  - [ ] No NaN or inf values
  - [ ] Timeseries look reasonable (no huge jumps)
  - [ ] Correlation between ROIs ~ expected values

---

## PHASE 4: β-VAE IMPLEMENTATION & TRAINING

- [ ] **Design VAE architecture**
  - [ ] Input: (200, 240) [ROI timeseries]
  - [ ] Encoder:
    - [ ] LSTM(input_size=200, hidden_size=128, num_layers=1)
    - [ ] Linear(128, 256) + ReLU
    - [ ] Linear(256, 2×latent_dim) → μ, log_σ²
  - [ ] Latent sampling: z = μ + ε ⊙ σ, ε ~ N(0,1)
  - [ ] Decoder:
    - [ ] Linear(latent_dim, 256) + ReLU
    - [ ] LSTM(256, 128)
    - [ ] Linear(128, 200) → reconstructed (200, 240)
  - [ ] Loss:
    - [ ] Reconstruction: MSE(BOLD, BOLD_recon)
    - [ ] KL divergence: -0.5 × Σ(1 + log_σ² - μ² - σ²)
    - [ ] β-VAE loss: MSE + β × KL, where β starts at 4 (tune 1-16)

- [ ] **Hyperparameter tuning grid**
  - [ ] latent_dim ∈ {8, 16, 32, 64}
  - [ ] β ∈ {1, 2, 4, 8, 16}
  - [ ] Validation metric: Reconstruction error + KL divergence

- [ ] **Training on training set only**
  - [ ] Input: ROI_train (50, 200, 240)
  - [ ] Batch size: 8
  - [ ] Epochs: 100
  - [ ] Optimizer: Adam(lr=1e-3)
  - [ ] Validation: Evaluate on ROI_val every epoch
  - [ ] Early stopping: Stop if val loss doesn't improve for 10 epochs
  - [ ] Save best model (lowest validation loss)

- [ ] **Encoder freeze for val/test**
  - [ ] Load best trained VAE
  - [ ] Extract encoder module
  - [ ] Freeze all weights (no_grad)
  - [ ] Use for inference only on val/test

---

## PHASE 5: LATENT ENCODING

- [ ] **Encode all subjects (train + val + test)**
  - [ ] For each subject:
    - [ ] z_mean, z_std = encoder(ROI_subject)
    - [ ] z = z_mean  # Use mean for deterministic encoding
    - [ ] Shape: (200, latent_dim)
  - [ ] Output:
    - [ ] z_train (50, 200, 16) or (50, 200, 32)
    - [ ] z_val (11, 200, 16)
    - [ ] z_test (10, 200, 16)

- [ ] **Latent space analysis (optional but recommended)**
  - [ ] PCA on z_train: reduce to 2-3 dims
  - [ ] Plot: HC vs AVH- vs AVH+ in latent space
  - [ ] Correlate latent dims with PSYRATS scores
  - [ ] Document: Which latent factors most predictive?

---

## PHASE 6: FUNCTIONAL CONNECTIVITY & GRAPHS

- [ ] **Compute FC for all subjects**
  - [ ] For each subject:
    - [ ] FC = np.corrcoef(ROI_subject)  # (200, 200)
    - [ ] Apply Fisher-z transform: z = 0.5 × log((1+FC)/(1-FC))
    - [ ] Threshold to top-k edges:
      - [ ] For each ROI: keep top-k=15 correlations
      - [ ] Set others to 0
    - [ ] Result: Sparse adjacency matrix A (200, 200)

- [ ] **Build graph objects**
  - [ ] For each subject in train/val/test:
    - [ ] Graph structure:
      - [ ] Nodes: 200 (Schaefer ROI IDs)
      - [ ] Node features: z_subject (200, 16) from VAE
      - [ ] Edge index: Non-zero entries of A (COO format)
      - [ ] Edge weights: FC values
      - [ ] Node labels (for attribution): ROI names, coordinates
    - [ ] Graph label: 0 (HC), 1 (AVH-), 2 (AVH+)

- [ ] **Verify graphs**
  - [ ] Check edge count: ~1500 (15 × 200 / 2)
  - [ ] Check node feature shape: (200, 16)
  - [ ] Check label distribution in train/val/test
  - [ ] Visualize 1-2 graphs (network plots)

---

## PHASE 7: GNN IMPLEMENTATION & TRAINING

- [ ] **Implement BrainGNN (or GCN baseline)**
  - [ ] Framework: PyTorch Geometric (torch_geometric)
  - [ ] Layers:
    - [ ] GCNConv(16, 32)
    - [ ] [Optional: Adaptive pooling layer]
    - [ ] GCNConv(32, 64)
    - [ ] Global mean/max pooling
    - [ ] Linear(64, 3) → 3 class logits
  - [ ] Dropout: 0.3 between layers
  - [ ] Loss: CrossEntropyLoss

- [ ] **Training loop**
  - [ ] Input: graphs_train (50 graphs)
  - [ ] Optimizer: Adam(lr=1e-3)
  - [ ] Epochs: 100
  - [ ] Per epoch:
    - [ ] Forward pass on all train graphs
    - [ ] Compute loss
    - [ ] Backward + optimize
    - [ ] Evaluate on val set
    - [ ] Log metrics
  - [ ] Early stopping: No improvement on val loss for 15 epochs
  - [ ] Save best model

- [ ] **Validation monitoring**
  - [ ] Per epoch: train loss, train accuracy, val accuracy
  - [ ] Plot: Learning curves (loss vs epoch)
  - [ ] Check for overfitting (val loss diverges from train loss)

---

## PHASE 8: FINAL EVALUATION (Test Set Only)

- [ ] **Inference on test set**
  - [ ] Load best GNN model
  - [ ] Forward pass on graphs_test
  - [ ] Get predictions: softmax(logits)
  - [ ] Get class labels: argmax(predictions)

- [ ] **Compute metrics**
  - [ ] Accuracy: (# correct) / 10
  - [ ] Per-class Precision, Recall, F1
  - [ ] Macro-averaged F1
  - [ ] Confusion matrix (3×3)
  - [ ] ROC-AUC (One-vs-Rest for multi-class)

- [ ] **Statistical significance**
  - [ ] Permutation test:
    - [ ] Shuffle labels 1000 times
    - [ ] Re-train GNN on permuted labels
    - [ ] Compute accuracy on test
    - [ ] P-value = (# permutations ≥ true accuracy) / 1000
  - [ ] Report p < 0.05 as significant

---

## PHASE 9: INTERPRETATION & VISUALIZATIONS

- [ ] **Latent space interpretation**
  - [ ] PCA of VAE latent codes
  - [ ] 2D/3D scatter plot: HC vs AVH- vs AVH+
  - [ ] Test stat: t-test or ANOVA on each latent dim
  - [ ] Heatmap: Top latent dimensions by effect size

- [ ] **Node/edge importance**
  - [ ] Gradient-based saliency: d(loss)/d(node_features)
  - [ ] Identify ROIs most important for prediction
  - [ ] Plot on brain surface (glass brain or cortex)
  - [ ] Report top-10 most important ROIs

- [ ] **Brain visualizations**
  - [ ] ROI classification confidence on cortex
  - [ ] Connectivity edges colored by importance
  - [ ] Latent factor activations per ROI

---

## PHASE 10: BASELINE COMPARISONS

- [ ] **Majority class baseline** (32% = 23/71)
  - [ ] Report as sanity check

- [ ] **Direct FC → GNN (without VAE)**
  - [ ] Use raw FC matrix as node features (or FC eigenvalues)
  - [ ] Train same GNN
  - [ ] Compare accuracy vs β-VAE+GNN

- [ ] **Logistic regression on VAE latents**
  - [ ] Use z_test directly (skip GNN)
  - [ ] Train LR classifier on train set
  - [ ] Evaluate on test set
  - [ ] Compare vs GNN (shows GNN adds value)

- [ ] **Standard FC + SVM** (classical baseline)
  - [ ] Compute FC for all subjects
  - [ ] Vectorize: 200×200 → 20k features
  - [ ] Train SVM on train set
  - [ ] Evaluate on test set

---

## PHASE 11: SENSITIVITY ANALYSES (Optional)

- [ ] **Schaefer-400 analysis**
  - [ ] Repeat Phases 3-8 with 400 ROIs
  - [ ] Compare accuracy: should be ≥ Schaefer-200

- [ ] **Confound variations**
  - [ ] Test: Motion only (no CSF/WM)
  - [ ] Test: Motion + global signal (controversial)
  - [ ] Report: Does confound choice matter?

- [ ] **FC edge type**
  - [ ] Pearson (current)
  - [ ] Partial correlation
  - [ ] Mutual information (if time permits)
  - [ ] Report: Robust to FC metric?

- [ ] **Sparsity level**
  - [ ] k-NN: k ∈ {10, 15, 20, 30}
  - [ ] Report: Does accuracy change with k?

- [ ] **Cross-validation on training**
  - [ ] 5-fold CV on train set (50 subjects)
  - [ ] Select hyperparams (latent_dim, β, learning rate)
  - [ ] Report: Mean ± std CV accuracy
  - [ ] Confirms robustness to subject variability

---

## PHASE 12: DOCUMENTATION & WRITING

- [ ] **Methods section**
  - [ ] Participants & dataset (N=71, groups, demographics)
  - [ ] fMRIPrep preprocessing summary
  - [ ] Post-fMRIPrep pipeline: confounds, smoothing, filtering
  - [ ] ROI selection: Schaefer-200 justification
  - [ ] β-VAE architecture & training (β=4, latent_dim=16, etc.)
  - [ ] Functional connectivity: Pearson + Fisher-z + sparsity
  - [ ] BrainGNN architecture
  - [ ] Train/val/test split (50/11/10, stratified)
  - [ ] Leakage prevention (fit preprocessing on train only)

- [ ] **Results section**
  - [ ] Test set accuracy + CI
  - [ ] Confusion matrix (3×3)
  - [ ] Per-class F1 scores
  - [ ] Permutation test p-value
  - [ ] Comparison to baselines (table)
  - [ ] Latent space analysis (figure)
  - [ ] Important ROIs (figure)

- [ ] **Discussion**
  - [ ] Interpretation of findings
  - [ ] Comparison to published schizophrenia classification work
  - [ ] Limitations (small N, single site, single task)
  - [ ] Clinical implications
  - [ ] Future directions

- [ ] **Code release**
  - [ ] GitHub repository (anonymized for review)
  - [ ] Clean, reproducible code
  - [ ] Requirements.txt with all package versions
  - [ ] README with setup & usage instructions
  - [ ] Random seed fixed (reproducibility)

---

## FINAL CHECKLIST

- [ ] **Reproducibility**
  - [ ] Random seed: 42 (set everywhere)
  - [ ] Train/val/test split saved
  - [ ] Hyperparameters documented
  - [ ] Results logged (no manual reporting)

- [ ] **Statistical rigor**
  - [ ] All preprocessing fit on train only
  - [ ] Test set untouched until final evaluation
  - [ ] Significance test (permutation)
  - [ ] Baselines reported

- [ ] **Interpretability**
  - [ ] Latent space analyzed
  - [ ] Important ROIs identified
  - [ ] Brain visualizations generated
  - [ ] Figures publication-ready

- [ ] **Quality**
  - [ ] No NaNs or infs in any output
  - [ ] All code tested on small subset first
  - [ ] Results validated by visual inspection
  - [ ] Consistency checks (e.g., accuracy ≤ 100%)

---

## SUCCESS CRITERIA

✅ **Expected outcomes**:
- Test accuracy: 70-85% (HC vs AVH+ binary), 60-75% (3-way)
- Baseline comparison: β-VAE+GNN > Direct-FC+GNN & > SVM
- Latent space: ≥3 interpretable factors
- Permutation test: p < 0.05
- Reproducibility: Code runs without error

❌ **Red flags** (investigate if seen):
- Test accuracy < 50%: Likely leakage or wrong architecture
- Val loss > train loss by >3x: Severe overfitting
- Permutation test p > 0.10: Results not significant
- Code doesn't run: Missing dependencies or path errors

---

**Status**: Ready for implementation  
**Estimated time**: 4-6 weeks with testing and interpretation  
**Key resource**: GPU (for VAE/GNN training, ~2-4 hours total)

#!/usr/bin/env python3
"""
Exact Journal Replicas (Figure 1, Figure 2, Figure 3) from PLoS ONE (e0276975)

Generates publication-quality figures matching the exact layout from:
"Speech Perception in Schizophrenia" - PLoS ONE e0276975

Author: Generated for fMRI analysis pipeline
Date: 2026-08-19
"""

import os
import sys
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns

from nilearn import datasets, image, glm, plotting, surface
from nilearn.glm import threshold_stats_img
from nilearn.maskers import NiftiSpheresMasker
from nilearn.image import resample_to_img

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/fMRI/logs/plos_one_figures.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# PATHS AND CONFIGURATION
# ============================================================================

BASE_DIR = Path('/root/fMRI')
DATA_DIR = BASE_DIR / 'ds004302-download'
ANALYSIS_DIR = BASE_DIR / 'analysis'
FIRST_LEVEL_DIR = ANALYSIS_DIR / 'first_level' / 'contrast_maps'
STAT_MAPS_DIR = ANALYSIS_DIR / 'second_level' / 'stat_maps'
OUTPUT_DIR = ANALYSIS_DIR / 'figures_v2'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load templates
logger.info("Loading MNI152 template and brain mask...")
mni_template = datasets.load_mni152_template(resolution=1)
mni_mask = datasets.load_mni152_brain_mask(resolution=2)

logger.info("Loading fsaverage surfaces...")
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')

# Configuration
AXIAL_CUTS = [-12, -2, 8, 18, 28, 38, 48]
DPI = 400

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_zmap(raw_zmap_path, z_thresh=2.3, k_thresh=50, two_sided=False):
    """
    Mask and threshold statistical map.
    
    Parameters:
    -----------
    raw_zmap_path : Path
        Path to raw z-score map
    z_thresh : float
        Height threshold (Z ≥ threshold)
    k_thresh : int
        Cluster extent threshold (k ≥ voxels)
    two_sided : bool
        Whether to apply two-sided thresholding
    
    Returns:
    --------
    thresh_img : nibabel.Nifti1Image or None
        Thresholded and masked statistical map
    """
    if not Path(raw_zmap_path).exists():
        logger.warning(f"File not found: {raw_zmap_path}")
        return None
    
    raw_img = nib.load(str(raw_zmap_path))
    
    # Resample mask to match image
    mask_resampled = resample_to_img(mni_mask, raw_img, interpolation='nearest')
    
    # Apply mask
    masked = image.math_img("img * mask", img=raw_img, mask=mask_resampled)
    
    # Apply cluster-extent thresholding
    thresh_img, _ = threshold_stats_img(
        masked,
        threshold=z_thresh,
        cluster_threshold=k_thresh,
        two_sided=two_sided,
        height_control=None
    )
    
    return thresh_img

def plot_activation_row(ax_axial, ax_sag, ax_pial_l, ax_pial_r, stat_img, z_thresh=2.3):
    """
    Renders Axial + Sagittal + 3D Left/Right Pial Lateral views.
    
    Parameters:
    -----------
    ax_axial : matplotlib.axes.Axes
        Axes for axial multi-slice display
    ax_sag : matplotlib.axes.Axes
        Axes for sagittal slice
    ax_pial_l : matplotlib.axes.Axes
        Axes for left hemisphere pial surface
    ax_pial_r : matplotlib.axes.Axes
        Axes for right hemisphere pial surface
    stat_img : nibabel.Nifti1Image
        Statistical map to plot
    z_thresh : float
        Z-score threshold
    """
    # 1. Axial cuts
    plotting.plot_stat_map(
        stat_img,
        bg_img=mni_template,
        display_mode='z',
        cut_coords=AXIAL_CUTS,
        threshold=z_thresh,
        cmap='hot',
        black_bg=False,
        draw_cross=False,
        colorbar=False,
        axes=ax_axial
    )
    
    # 2. Sagittal midline cut
    plotting.plot_stat_map(
        stat_img,
        bg_img=mni_template,
        display_mode='x',
        cut_coords=[0],
        threshold=z_thresh,
        cmap='hot',
        black_bg=False,
        draw_cross=False,
        colorbar=False,
        axes=ax_sag
    )
    
    # 3. 3D Left Pial
    tex_l = surface.vol_to_surf(stat_img, fsaverage.pial_left, interpolation='linear')
    plotting.plot_surf_stat_map(
        fsaverage.pial_left,
        tex_l,
        hemi='left',
        view='lateral',
        threshold=z_thresh,
        bg_map=fsaverage.sulc_left,
        cmap='hot',
        colorbar=False,
        axes=ax_pial_l,
        bg_on_data=True
    )
    
    # 4. 3D Right Pial
    tex_r = surface.vol_to_surf(stat_img, fsaverage.pial_right, interpolation='linear')
    plotting.plot_surf_stat_map(
        fsaverage.pial_right,
        tex_r,
        hemi='right',
        view='lateral',
        threshold=z_thresh,
        bg_map=fsaverage.sulc_right,
        cmap='hot',
        colorbar=False,
        axes=ax_pial_r,
        bg_on_data=True
    )

def extract_roi_betas(coords, contrast_name):
    """
    Extracts mean parameter estimates from spherical ROIs across all subjects.
    
    Parameters:
    -----------
    coords : tuple
        MNI coordinates (x, y, z)
    contrast_name : str
        Name of the contrast
    
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with columns: Group, Beta
    """
    logger.info(f"Extracting ROI betas at {coords} for {contrast_name}...")
    
    participants_file = DATA_DIR / 'participants.tsv'
    if not participants_file.exists():
        logger.warning(f"Participants file not found: {participants_file}")
        return pd.DataFrame()
    
    participants_df = pd.read_csv(participants_file, sep='\t')
    participants_df = participants_df[participants_df['participant_id'].notna()].copy()
    
    logger.info(f"Found {len(participants_df)} participants")
    
    # Initialize masker
    masker = NiftiSpheresMasker(seeds=[coords], radius=6, smoothing_fwhm=None)
    
    records = []
    
    for _, row in participants_df.iterrows():
        sid = row['participant_id']
        group = row.get('group', 'Unknown')
        
        # Try different file naming patterns
        possible_paths = [
            FIRST_LEVEL_DIR / f"{sid}_{contrast_name}_zmap.nii.gz",
            FIRST_LEVEL_DIR / f"{sid}_{contrast_name}.nii.gz",
            FIRST_LEVEL_DIR / sid / f"{contrast_name}.nii.gz",
        ]
        
        fpath = None
        for p in possible_paths:
            if p.exists():
                fpath = p
                break
        
        if fpath and fpath.exists():
            try:
                val = masker.fit_transform(str(fpath))
                beta_val = float(np.mean(val))
                records.append({
                    'Group': group,
                    'Beta': beta_val,
                    'Subject': sid
                })
            except Exception as e:
                logger.warning(f"Failed to extract {sid}: {e}")
    
    df = pd.DataFrame(records)
    logger.info(f"Extracted {len(df)} subjects")
    
    return df

# ============================================================================
# FIGURE GENERATION FUNCTIONS
# ============================================================================

def generate_figure_1():
    """
    Generate Figure 1: Words Condition
    
    Layout:
    - Row A: Healthy Controls (HC)
    - Row B: Schizophrenia Patients (SCHZ)
    """
    logger.info("\n" + "="*80)
    logger.info("GENERATING FIGURE 1 (Words Condition)")
    logger.info("="*80)
    
    hc_img = clean_zmap(STAT_MAPS_DIR / "HC_words_vs_baseline_zmap.nii.gz")
    scz_img = clean_zmap(STAT_MAPS_DIR / "SCHZ_words_vs_baseline_zmap.nii.gz")
    
    if hc_img is None or scz_img is None:
        logger.error("Missing statistical maps for Figure 1")
        return
    
    fig = plt.figure(figsize=(22, 6.5))
    gs = GridSpec(2, 4, figure=fig, 
                  width_ratios=[5.5, 1.0, 1.2, 1.2], 
                  hspace=0.3, wspace=0.1)
    
    # Row A: HC
    logger.info("  Rendering Row A: Healthy Controls...")
    ax_axial_hc = fig.add_subplot(gs[0, 0])
    ax_sag_hc = fig.add_subplot(gs[0, 1], projection='3d')
    ax_pial_l_hc = fig.add_subplot(gs[0, 2], projection='3d')
    ax_pial_r_hc = fig.add_subplot(gs[0, 3], projection='3d')
    
    plot_activation_row(ax_axial_hc, ax_sag_hc, ax_pial_l_hc, ax_pial_r_hc, hc_img)
    ax_axial_hc.set_title("A: Healthy Controls (HC) — Words", 
                         fontsize=13, fontweight='bold', loc='left', pad=10)
    
    # Row B: SCHZ
    logger.info("  Rendering Row B: Schizophrenia Patients...")
    ax_axial_scz = fig.add_subplot(gs[1, 0])
    ax_sag_scz = fig.add_subplot(gs[1, 1], projection='3d')
    ax_pial_l_scz = fig.add_subplot(gs[1, 2], projection='3d')
    ax_pial_r_scz = fig.add_subplot(gs[1, 3], projection='3d')
    
    plot_activation_row(ax_axial_scz, ax_sag_scz, ax_pial_l_scz, ax_pial_r_scz, scz_img)
    ax_axial_scz.set_title("B: Schizophrenia Patients (SCHZ) — Words", 
                          fontsize=13, fontweight='bold', loc='left', pad=10)
    
    # Save
    out_file = OUTPUT_DIR / "Figure1_words_condition.png"
    plt.savefig(str(out_file), dpi=DPI, pad_inches=0.5)
    plt.close(fig)
    logger.info(f"  ✓ Saved: {out_file.name}")

def generate_figure_2():
    """
    Generate Figure 2: Sentences Condition & Group Differences
    
    Layout:
    - Row A: HC activation
    - Row B: SCHZ activation
    - Panel C: Left Heschl's Gyrus hypoactivation (MNI: -56, -2, 2)
    - Panel D: ROI Beta Boxplot
    """
    logger.info("\n" + "="*80)
    logger.info("GENERATING FIGURE 2 (Sentences Condition & Differences)")
    logger.info("="*80)
    
    hc_img = clean_zmap(STAT_MAPS_DIR / "HC_sentences_vs_baseline_zmap.nii.gz")
    scz_img = clean_zmap(STAT_MAPS_DIR / "SCHZ_sentences_vs_baseline_zmap.nii.gz")
    
    # Try to load group difference map
    diff_path = STAT_MAPS_DIR / "HC_gt_SCHZ_sentences_vs_baseline_zmap.nii.gz"
    if diff_path.exists():
        diff_img = clean_zmap(diff_path, z_thresh=2.0, k_thresh=30, two_sided=True)
    else:
        logger.warning("Group difference map not found, skipping difference panels")
        diff_img = None
    
    if hc_img is None or scz_img is None:
        logger.error("Missing statistical maps for Figure 2")
        return
    
    fig = plt.figure(figsize=(22, 12))
    gs = GridSpec(4, 4, figure=fig, 
                  width_ratios=[5.5, 1.0, 1.2, 1.2], 
                  height_ratios=[1, 1, 0.9, 1.2], 
                  hspace=0.45, wspace=0.15)
    
    # Row A: HC
    logger.info("  Rendering Row A: Healthy Controls...")
    ax_axial_hc = fig.add_subplot(gs[0, 0])
    ax_sag_hc = fig.add_subplot(gs[0, 1], projection='3d')
    ax_pial_l_hc = fig.add_subplot(gs[0, 2], projection='3d')
    ax_pial_r_hc = fig.add_subplot(gs[0, 3], projection='3d')
    
    plot_activation_row(ax_axial_hc, ax_sag_hc, ax_pial_l_hc, ax_pial_r_hc, hc_img)
    ax_axial_hc.set_title("A: Healthy Controls (HC) — Sentences", 
                         fontsize=13, fontweight='bold', loc='left', pad=10)
    
    # Row B: SCHZ
    logger.info("  Rendering Row B: Schizophrenia Patients...")
    ax_axial_scz = fig.add_subplot(gs[1, 0])
    ax_sag_scz = fig.add_subplot(gs[1, 1], projection='3d')
    ax_pial_l_scz = fig.add_subplot(gs[1, 2], projection='3d')
    ax_pial_r_scz = fig.add_subplot(gs[1, 3], projection='3d')
    
    plot_activation_row(ax_axial_scz, ax_sag_scz, ax_pial_l_scz, ax_pial_r_scz, scz_img)
    ax_axial_scz.set_title("B: Schizophrenia Patients (SCHZ) — Sentences", 
                          fontsize=13, fontweight='bold', loc='left', pad=10)
    
    # Row C: Focal Left Heschl's Gyrus (MNI: -56, -2, 2)
    if diff_img is not None:
        logger.info("  Rendering Panel C: Group Difference...")
        ax_c_axial = fig.add_subplot(gs[2, 0:2])
        plotting.plot_stat_map(
            diff_img,
            bg_img=mni_template,
            display_mode='z',
            cut_coords=[2],
            threshold=2.0,
            cmap='cold_hot',
            black_bg=False,
            draw_cross=True,
            colorbar=True,
            axes=ax_c_axial
        )
        ax_c_axial.set_title("C: Reduced Activation in Left Heschl's Gyrus (z=2)", 
                            fontsize=11, fontweight='bold', loc='left')
    
    # Row D: Boxplot of Heschl's Gyrus
    logger.info("  Rendering Panel D: ROI Boxplot...")
    ax_box = fig.add_subplot(gs[3, 0:2])
    df_heschl = extract_roi_betas((-56, -2, 2), "sentences_vs_baseline")
    
    if not df_heschl.empty:
        # Map group names if needed
        group_order = ['AVH-', 'AVH+', 'HC']
        palette = {'AVH-': '#74c476', 'AVH+': '#fd8d3c', 'HC': '#6baed6'}
        
        sns.boxplot(x='Group', y='Beta', data=df_heschl, order=group_order,
                    palette=palette, width=0.4, ax=ax_box, showfliers=False)
        sns.stripplot(x='Group', y='Beta', data=df_heschl, order=group_order,
                      color='black', alpha=0.6, jitter=0.2, size=5, ax=ax_box)
        ax_box.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax_box.set_ylabel("Beta / Parameter Estimate", fontsize=10)
        ax_box.set_title("Left Heschl's Gyrus (MNI: -56, -2, 2)", fontsize=11, fontweight='bold')
        sns.despine(ax=ax_box)
    else:
        ax_box.text(0.5, 0.5, 'ROI data not available\n(First-level maps required)',
                   ha='center', va='center', fontsize=12, transform=ax_box.transAxes)
        ax_box.axis('off')
    
    # Save
    out_file = OUTPUT_DIR / "Figure2_sentences_complete.png"
    plt.savefig(str(out_file), dpi=DPI, pad_inches=0.5)
    plt.close(fig)
    logger.info(f"  ✓ Saved: {out_file.name}")

def generate_figure_3():
    """
    Generate Figure 3: Reversed Speech Condition & Group Differences
    
    Layout:
    - Row A: HC activation
    - Row B: SCHZ activation
    - Panel C: Hypoactivation in Left Auditory Cortex
    - Panel D: ROI Beta Boxplot
    """
    logger.info("\n" + "="*80)
    logger.info("GENERATING FIGURE 3 (Reversed Speech Condition)")
    logger.info("="*80)
    
    hc_img = clean_zmap(STAT_MAPS_DIR / "HC_reversed_vs_baseline_zmap.nii.gz")
    scz_img = clean_zmap(STAT_MAPS_DIR / "SCHZ_reversed_vs_baseline_zmap.nii.gz")
    
    # Try to load group difference map
    diff_path = STAT_MAPS_DIR / "HC_gt_SCHZ_reversed_vs_baseline_zmap.nii.gz"
    if diff_path.exists():
        diff_img = clean_zmap(diff_path, z_thresh=2.0, k_thresh=30, two_sided=True)
    else:
        logger.warning("Group difference map not found, skipping difference panels")
        diff_img = None
    
    if hc_img is None or scz_img is None:
        logger.error("Missing statistical maps for Figure 3")
        return
    
    fig = plt.figure(figsize=(22, 12))
    gs = GridSpec(4, 4, figure=fig, 
                  width_ratios=[5.5, 1.0, 1.2, 1.2], 
                  height_ratios=[1, 1, 0.9, 1.2], 
                  hspace=0.45, wspace=0.15)
    
    # Row A: HC
    logger.info("  Rendering Row A: Healthy Controls...")
    ax_axial_hc = fig.add_subplot(gs[0, 0])
    ax_sag_hc = fig.add_subplot(gs[0, 1], projection='3d')
    ax_pial_l_hc = fig.add_subplot(gs[0, 2], projection='3d')
    ax_pial_r_hc = fig.add_subplot(gs[0, 3], projection='3d')
    
    plot_activation_row(ax_axial_hc, ax_sag_hc, ax_pial_l_hc, ax_pial_r_hc, hc_img)
    ax_axial_hc.set_title("A: Healthy Controls (HC) — Reversed Speech", 
                         fontsize=13, fontweight='bold', loc='left', pad=10)
    
    # Row B: SCHZ
    logger.info("  Rendering Row B: Schizophrenia Patients...")
    ax_axial_scz = fig.add_subplot(gs[1, 0])
    ax_sag_scz = fig.add_subplot(gs[1, 1], projection='3d')
    ax_pial_l_scz = fig.add_subplot(gs[1, 2], projection='3d')
    ax_pial_r_scz = fig.add_subplot(gs[1, 3], projection='3d')
    
    plot_activation_row(ax_axial_scz, ax_sag_scz, ax_pial_l_scz, ax_pial_r_scz, scz_img)
    ax_axial_scz.set_title("B: Schizophrenia Patients (SCHZ) — Reversed Speech", 
                          fontsize=13, fontweight='bold', loc='left', pad=10)
    
    # Row C: Focal Cluster
    if diff_img is not None:
        logger.info("  Rendering Panel C: Group Difference...")
        ax_c = fig.add_subplot(gs[2, 0:2])
        plotting.plot_stat_map(
            diff_img,
            bg_img=mni_template,
            display_mode='z',
            cut_coords=[2],
            threshold=2.0,
            cmap='cold_hot',
            black_bg=False,
            draw_cross=True,
            colorbar=True,
            axes=ax_c
        )
        ax_c.set_title("C: Hypoactivation in Left Auditory Cortex (z=2)", 
                      fontsize=11, fontweight='bold', loc='left')
    
    # Row D: Boxplot
    logger.info("  Rendering Panel D: ROI Boxplot...")
    ax_box = fig.add_subplot(gs[3, 0:2])
    df_rev = extract_roi_betas((-56, -2, 2), "reversed_vs_baseline")
    
    if not df_rev.empty:
        group_order = ['AVH-', 'AVH+', 'HC']
        palette = {'AVH-': '#74c476', 'AVH+': '#fd8d3c', 'HC': '#6baed6'}
        
        sns.boxplot(x='Group', y='Beta', data=df_rev, order=group_order,
                    palette=palette, width=0.4, ax=ax_box, showfliers=False)
        sns.stripplot(x='Group', y='Beta', data=df_rev, order=group_order,
                      color='black', alpha=0.6, jitter=0.2, size=5, ax=ax_box)
        ax_box.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax_box.set_ylabel("Beta / Parameter Estimate", fontsize=10)
        ax_box.set_title("Primary Auditory Cortex ROI", fontsize=11, fontweight='bold')
        sns.despine(ax=ax_box)
    else:
        ax_box.text(0.5, 0.5, 'ROI data not available\n(First-level maps required)',
                   ha='center', va='center', fontsize=12, transform=ax_box.transAxes)
        ax_box.axis('off')
    
    # Save
    out_file = OUTPUT_DIR / "Figure3_reversed_speech_complete.png"
    plt.savefig(str(out_file), dpi=DPI, pad_inches=0.5)
    plt.close(fig)
    logger.info(f"  ✓ Saved: {out_file.name}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    logger.info("="*80)
    logger.info("EXACT PLOS ONE JOURNAL REPLICA GENERATION")
    logger.info("="*80)
    logger.info("\nGenerating Figure 1, Figure 2, and Figure 3 from PLoS ONE (e0276975)")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Generate all figures
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("FIGURE REPRODUCTION COMPLETE")
    logger.info("="*80)
    logger.info(f"\nAll figures saved to: {OUTPUT_DIR}")
    logger.info("\nGenerated files:")
    
    # List all generated files
    generated_files = sorted(OUTPUT_DIR.glob("*.png"))
    for f in generated_files:
        logger.info(f"  - {f.name}")
    
    logger.info("="*80)

if __name__ == '__main__':
    main()

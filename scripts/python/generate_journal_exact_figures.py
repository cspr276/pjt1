#!/usr/bin/env python3
"""
Exact PLoS ONE Single-Row Pial 3D + Slice Composite Figures
Generates publication-quality figures matching PLoS ONE (e0276975) layout:
- Axial multi-slice gallery (z=-12 to 48)
- Sagittal midline slice
- 3D Left/Right anatomical pial surfaces with sulcal shading

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

from nilearn import datasets, image, glm, plotting, surface
from nilearn.glm import threshold_stats_img

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/fMRI/logs/journal_exact_figures.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# PATHS AND CONFIGURATION
# ============================================================================

BASE_DIR = Path('/root/fMRI')
ANALYSIS_DIR = BASE_DIR / 'analysis'
STAT_MAPS_DIR = ANALYSIS_DIR / 'second_level' / 'stat_maps'
OUTPUT_DIR = ANALYSIS_DIR / 'figures_journal_exact'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Templates
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
    Mask statistical map within intracranial space and apply cluster extent threshold.
    
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
    
    # Resample to mask resolution if needed
    resampled = image.resample_to_img(raw_img, mni_mask, interpolation='continuous')
    
    # Apply intracranial mask
    masked = image.math_img("img * mask", img=resampled, mask=mni_mask)
    
    # Apply cluster-extent thresholding
    thresh_img, _ = threshold_stats_img(
        masked,
        threshold=z_thresh,
        cluster_threshold=k_thresh,
        two_sided=two_sided,
        height_control=None
    )
    
    return thresh_img

def plot_journal_row(ax_axial, ax_sagittal, ax_pial_l, ax_pial_r, stat_img, 
                     cmap='hot', z_thresh=2.3):
    """
    Render one complete horizontal row matching PLoS ONE Figure 1/2 layout.
    
    Layout: [Axial Slices] + [Sagittal] + [Left Pial 3D] + [Right Pial 3D]
    
    Parameters:
    -----------
    ax_axial : matplotlib.axes.Axes
        Axes for axial multi-slice display
    ax_sagittal : matplotlib.axes.Axes
        Axes for sagittal slice
    ax_pial_l : matplotlib.axes.Axes
        Axes for left hemisphere pial surface
    ax_pial_r : matplotlib.axes.Axes
        Axes for right hemisphere pial surface
    stat_img : nibabel.Nifti1Image
        Statistical map to plot
    cmap : str
        Colormap name
    z_thresh : float
        Z-score threshold
    """
    
    # 1. Axial multi-slice gallery (7 slices)
    plotting.plot_stat_map(
        stat_img,
        bg_img=mni_template,
        display_mode='z',
        cut_coords=AXIAL_CUTS,
        threshold=z_thresh,
        cmap=cmap,
        black_bg=False,
        draw_cross=False,
        colorbar=False,
        axes=ax_axial
    )
    
    # 2. Sagittal midline slice
    plotting.plot_stat_map(
        stat_img,
        bg_img=mni_template,
        display_mode='x',
        cut_coords=[0],
        threshold=z_thresh,
        cmap=cmap,
        black_bg=False,
        draw_cross=False,
        colorbar=False,
        axes=ax_sagittal
    )
    
    # 3. 3D Left Anatomical Pial Surface (Realistic folded cortex)
    tex_l = surface.vol_to_surf(stat_img, fsaverage.pial_left, interpolation='linear')
    plotting.plot_surf_stat_map(
        fsaverage.pial_left,  # PIAL MESH (anatomical, not inflated)
        tex_l,
        hemi='left',
        view='lateral',
        threshold=z_thresh,
        bg_map=fsaverage.sulc_left,  # Sulcal depth for 3D shading
        cmap=cmap,
        colorbar=False,
        axes=ax_pial_l,
        bg_on_data=True  # Show background on data for better depth perception
    )
    
    # 4. 3D Right Anatomical Pial Surface
    tex_r = surface.vol_to_surf(stat_img, fsaverage.pial_right, interpolation='linear')
    plotting.plot_surf_stat_map(
        fsaverage.pial_right,  # PIAL MESH
        tex_r,
        hemi='right',
        view='lateral',
        threshold=z_thresh,
        bg_map=fsaverage.sulc_right,  # Sulcal depth for 3D shading
        cmap=cmap,
        colorbar=False,
        axes=ax_pial_r,
        bg_on_data=True  # Show background on data for better depth perception
    )

# ============================================================================
# MAIN FIGURE GENERATION
# ============================================================================

def generate_condition_figure(contrast_name, z_thresh=2.3, k_thresh=50):
    """
    Generate Figure 1 & Figure 2 style multi-row composites.
    
    Creates a 2-row figure:
    - Row A: Healthy Controls (HC)
    - Row B: Schizophrenia Patients (SCHZ)
    
    Each row contains: Axial slices + Sagittal + Left Pial + Right Pial
    
    Parameters:
    -----------
    contrast_name : str
        Name of the contrast (e.g., 'sentences_vs_baseline')
    z_thresh : float
        Z-score threshold
    k_thresh : int
        Cluster extent threshold
    """
    logger.info(f"\nProcessing {contrast_name}...")
    
    # Load and clean statistical maps
    hc_path = STAT_MAPS_DIR / f"HC_{contrast_name}_zmap.nii.gz"
    scz_path = STAT_MAPS_DIR / f"SCHZ_{contrast_name}_zmap.nii.gz"
    
    hc_img = clean_zmap(hc_path, z_thresh=z_thresh, k_thresh=k_thresh)
    scz_img = clean_zmap(scz_path, z_thresh=z_thresh, k_thresh=k_thresh)
    
    if hc_img is None or scz_img is None:
        logger.error(f"Missing statistical maps for {contrast_name}")
        return
    
    # Create figure with 2 rows
    fig = plt.figure(figsize=(24, 8))
    gs = GridSpec(2, 10, figure=fig, 
                  width_ratios=[6, 1.2, 1.4, 1.4] + [0]*6, 
                  hspace=0.35, wspace=0.15)
    
    # Row 1: Healthy Controls (A)
    logger.info("  Rendering Row A: Healthy Controls...")
    ax_axial_hc = fig.add_subplot(gs[0, 0:6])
    ax_sag_hc = fig.add_subplot(gs[0, 6], projection='3d')
    ax_pial_l_hc = fig.add_subplot(gs[0, 7:8], projection='3d')
    ax_pial_r_hc = fig.add_subplot(gs[0, 8:9], projection='3d')
    
    plot_journal_row(ax_axial_hc, ax_sag_hc, ax_pial_l_hc, ax_pial_r_hc, 
                     hc_img, cmap='hot', z_thresh=z_thresh)
    ax_axial_hc.set_title(
        f"A: Healthy Controls (HC) — {contrast_name.replace('_', ' ').title()}", 
        fontsize=13, fontweight='bold', loc='left', pad=10, color='black'
    )
    
    # Row 2: Schizophrenia Patients (B)
    logger.info("  Rendering Row B: Schizophrenia Patients...")
    ax_axial_scz = fig.add_subplot(gs[1, 0:6])
    ax_sag_scz = fig.add_subplot(gs[1, 6], projection='3d')
    ax_pial_l_scz = fig.add_subplot(gs[1, 7:8], projection='3d')
    ax_pial_r_scz = fig.add_subplot(gs[1, 8:9], projection='3d')
    
    plot_journal_row(ax_axial_scz, ax_sag_scz, ax_pial_l_scz, ax_pial_r_scz, 
                     scz_img, cmap='hot', z_thresh=z_thresh)
    ax_axial_scz.set_title(
        f"B: Schizophrenia Patients (SCHZ) — {contrast_name.replace('_', ' ').title()}", 
        fontsize=13, fontweight='bold', loc='left', pad=10, color='black'
    )
    
    # Save figure (use pad_inches instead of bbox_inches='tight' for 3D axes compatibility)
    out_file = OUTPUT_DIR / f"journal_exact_{contrast_name}.png"
    plt.savefig(str(out_file), dpi=DPI, pad_inches=0.5)
    plt.close(fig)
    logger.info(f"  ✓ Saved exact journal figure: {out_file.name}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("EXACT PLOS ONE JOURNAL-STYLE FIGURE GENERATION")
    logger.info("=" * 80)
    logger.info("\nGenerating exact journal-style composites...")
    logger.info("Layout: [Axial Slices] + [Sagittal] + [Left Pial 3D] + [Right Pial 3D]")
    logger.info("Surface: Anatomical pial (not inflated) with sulcal shading")
    
    # Generate figures for all conditions
    conditions = ['sentences_vs_baseline', 'words_vs_baseline', 'reversed_vs_baseline']
    
    for cond in conditions:
        generate_condition_figure(cond, z_thresh=2.3, k_thresh=50)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nAll figures saved to: {OUTPUT_DIR}")
    logger.info("\nGenerated files:")
    
    # List all generated files
    generated_files = sorted(OUTPUT_DIR.glob("*.png"))
    for f in generated_files:
        logger.info(f"  - {f.name}")
    
    logger.info("=" * 80)

if __name__ == '__main__':
    main()

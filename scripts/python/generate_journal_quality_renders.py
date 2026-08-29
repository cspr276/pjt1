#!/usr/bin/env python3
"""
Journal-Quality 3D Surface Renders, Orthogonal Views & ROI Boxplots
PLoS ONE Reproduction: Speech Perception in Schizophrenia

This script generates high-end visualizations matching PLoS ONE standards:
1. 3D high-density cortical surface renders (multi-view)
2. Orthogonal focus views at Left Heschl's Gyrus
3. ROI parameter extraction boxplots
4. Multi-condition auditory profiles

Author: Generated for fMRI analysis pipeline
Date: 2026-08-18
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/fMRI/logs/journal_quality_renders.log'),
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
OUTPUT_DIR = ANALYSIS_DIR / 'figures_high_res_presentation'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Templates (High Resolution)
logger.info("Loading MNI152 template and brain mask...")
mni_hires = datasets.load_mni152_template(resolution=1)
mni_mask = datasets.load_mni152_brain_mask(resolution=2)

logger.info("Loading fsaverage surfaces...")
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')

DPI = 400

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def mask_and_threshold(zmap_path, z_thresh=2.3, cluster_size=50, two_sided=False):
    """Load, mask, and threshold a statistical map."""
    if not Path(zmap_path).exists():
        logger.warning(f"File missing: {zmap_path}")
        return None
    
    z_img = nib.load(str(zmap_path))
    
    # Resample to mask resolution if needed
    resampled = image.resample_to_img(z_img, mni_mask, interpolation='continuous')
    
    # Apply intracranial mask
    masked = image.math_img("img * mask", img=resampled, mask=mni_mask)
    
    # Apply cluster-extent thresholding
    thresh_img, _ = threshold_stats_img(
        masked,
        threshold=z_thresh,
        cluster_threshold=cluster_size,
        two_sided=two_sided,
        height_control=None
    )
    
    return thresh_img

# ============================================================================
# 1. 3D SURFACE MULTI-VIEW RENDERING
# ============================================================================

def render_surface_multiview(zmap_path, out_png, title, z_thresh=2.3, cmap='hot'):
    """
    Generate multi-view 3D surface renders (Lateral, Medial views).
    
    Parameters:
    -----------
    zmap_path : Path
        Path to z-score map
    out_png : Path
        Output file path
    title : str
        Figure title
    z_thresh : float
        Z-score threshold
    cmap : str
        Colormap name
    """
    logger.info(f"  Rendering 3D multi-view: {Path(zmap_path).name}")
    
    thresh_img = mask_and_threshold(
        zmap_path,
        z_thresh=z_thresh,
        cluster_size=50,
        two_sided=(cmap == 'cold_hot')
    )
    
    if thresh_img is None:
        return
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.15, wspace=0.08)
    
    views = [
        ('left', 'lateral', gs[0, 0], 'Left Lateral View'),
        ('right', 'lateral', gs[0, 1], 'Right Lateral View'),
        ('left', 'medial', gs[1, 0], 'Left Medial View'),
        ('right', 'medial', gs[1, 1], 'Right Medial View')
    ]
    
    for hemi, view, grid_pos, v_title in views:
        ax = fig.add_subplot(grid_pos, projection='3d')
        
        # Select appropriate surface
        pial = fsaverage.pial_left if hemi == 'left' else fsaverage.pial_right
        infl = fsaverage.infl_left if hemi == 'left' else fsaverage.infl_right
        sulc = fsaverage.sulc_left if hemi == 'left' else fsaverage.sulc_right
        
        # Project volumetric data to surface
        texture = surface.vol_to_surf(thresh_img, pial, interpolation='linear')
        
        # Plot surface
        plotting.plot_surf_stat_map(
            infl,
            texture,
            hemi=hemi,
            view=view,
            threshold=z_thresh,
            bg_map=sulc,
            cmap=cmap,
            colorbar=True if view == 'lateral' and hemi == 'right' else False,
            axes=ax,
            title=None
        )
        ax.set_title(v_title, fontsize=12, fontweight='bold', pad=4, color='black')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(str(out_png), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  ✓ Saved 3D Multi-View: {out_png.name}")

# ============================================================================
# 2. ORTHOGONAL SLICE FOCAL VIEWS (LEFT HESCHL'S GYRUS)
# ============================================================================

def render_orthogonal_focus(zmap_path, out_png, title, cut_coords=(-56, -2, 2), 
                           z_thresh=2.0, cmap='cold_hot'):
    """
    Generate orthogonal slice views centered on a specific coordinate.
    
    Parameters:
    -----------
    zmap_path : Path
        Path to z-score map
    out_png : Path
        Output file path
    title : str
        Figure title
    cut_coords : tuple
        MNI coordinates for centering
    z_thresh : float
        Z-score threshold
    cmap : str
        Colormap name
    """
    logger.info(f"  Rendering orthogonal view: {Path(zmap_path).name}")
    
    thresh_img = mask_and_threshold(
        zmap_path,
        z_thresh=z_thresh,
        cluster_size=30,
        two_sided=True
    )
    
    if thresh_img is None:
        return
    
    fig = plt.figure(figsize=(12, 4.5))
    
    display = plotting.plot_stat_map(
        thresh_img,
        bg_img=mni_hires,
        cut_coords=cut_coords,
        display_mode='ortho',
        threshold=z_thresh,
        title=None,
        cmap=cmap,
        black_bg=False,
        draw_cross=True,
        figure=fig
    )
    
    plt.suptitle(title, fontsize=13, fontweight='bold', y=0.98, color='black')
    plt.savefig(str(out_png), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  ✓ Saved Orthogonal View: {out_png.name}")

# ============================================================================
# 3. LEFT HESCHL'S GYRUS ROI PARAMETER EXTRACTION & BOXPLOTS
# ============================================================================

def generate_heschl_gyrus_boxplots():
    """
    Extract ROI parameter estimates and generate boxplots.
    
    ROI: Left Heschl's Gyrus (MNI: -56, -2, 2)
    Radius: 8mm sphere
    """
    logger.info("\nGenerating Heschl's Gyrus ROI boxplots...")
    
    # Load participant data
    participants_df = pd.read_csv(DATA_DIR / 'participants.tsv', sep='\t')
    participants_df = participants_df[participants_df['participant_id'].notna()].copy()
    
    logger.info(f"  Loaded {len(participants_df)} participants")
    
    # Define ROI coordinates (Left Heschl's Gyrus)
    heschl_coords = [(-56, -2, 2)]
    sphere_masker = NiftiSpheresMasker(seeds=heschl_coords, radius=8)
    
    contrasts = ['words_vs_baseline', 'sentences_vs_baseline', 'reversed_vs_baseline']
    
    for contrast in contrasts:
        logger.info(f"  Processing {contrast}...")
        records = []
        
        for _, row in participants_df.iterrows():
            sid = row['participant_id'].replace('sub-', '')
            
            # Try multiple possible file naming conventions
            fpath = FIRST_LEVEL_DIR / f"sub-{sid}_{contrast}_zmap.nii.gz"
            if not fpath.exists():
                fpath = FIRST_LEVEL_DIR / f"{sid}_{contrast}_zmap.nii.gz"
            
            if fpath.exists():
                try:
                    # Extract mean value from ROI
                    result = sphere_masker.fit_transform(str(fpath))
                    # Handle both 1D and 2D array outputs
                    if result.ndim == 1:
                        val = result[0]
                    else:
                        val = result[0, 0]
                    records.append({
                        'Subject': sid,
                        'Group': row['group'],
                        'Beta': val
                    })
                except Exception as e:
                    logger.warning(f"    Failed to extract ROI for {sid}: {e}")
        
        if not records:
            logger.warning(f"    No data found for {contrast}")
            continue
        
        df_roi = pd.DataFrame(records)
        logger.info(f"    Extracted ROI values for {len(df_roi)} subjects")
        
        # Generate boxplot
        plt.figure(figsize=(6.5, 5.5))
        sns.set_theme(style='ticks')
        
        # Color palette matching PLoS ONE paper
        palette = {'AVH-': '#74c476', 'AVH+': '#fd8d3c', 'HC': '#6baed6'}
        
        # Boxplot
        ax = sns.boxplot(
            x='Group', y='Beta', data=df_roi,
            order=['AVH-', 'AVH+', 'HC'],
            palette=palette,
            width=0.45,
            boxprops=dict(alpha=0.85),
            fliersize=0
        )
        
        # Jittered scatter overlay
        sns.stripplot(
            x='Group', y='Beta', data=df_roi,
            order=['AVH-', 'AVH+', 'HC'],
            color='black',
            alpha=0.6,
            jitter=0.2,
            size=6
        )
        
        # Add reference line at zero
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        
        # Styling
        sns.despine(top=True, right=True)
        plt.title(
            f"Heschl's Gyrus Activation: {contrast.replace('_', ' ').title()}\n"
            f"(MNI: -56, -2, 2)",
            fontsize=12,
            fontweight='bold',
            pad=12
        )
        plt.ylabel("Parameter Estimate (Mean Z-score)", fontsize=11)
        plt.xlabel("Subject Group", fontsize=11)
        
        # Save
        out_box = OUTPUT_DIR / f"boxplot_heschls_gyrus_{contrast}.png"
        plt.savefig(str(out_box), dpi=DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"    ✓ Saved ROI Boxplot: {out_box.name}")

# ============================================================================
# 4. MULTI-CONDITION AUDITORY PROFILE
# ============================================================================

def generate_multi_condition_comparison():
    """
    Generate comparative panels showing auditory activation across conditions.
    """
    logger.info("\nGenerating multi-condition auditory profile...")
    
    conditions = ['words_vs_baseline', 'sentences_vs_baseline', 'reversed_vs_baseline']
    
    # Create figure with 3 columns (one per condition)
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.15)
    
    for idx, cond in enumerate(conditions):
        logger.info(f"  Processing {cond}...")
        
        zmap_path = STAT_MAPS_DIR / f"HC_{cond}_zmap.nii.gz"
        thresh_img = mask_and_threshold(zmap_path, z_thresh=2.5, cluster_size=50)
        
        if thresh_img is None:
            continue
        
        ax = fig.add_subplot(gs[0, idx], projection='3d')
        
        # Project to left hemisphere
        texture = surface.vol_to_surf(thresh_img, fsaverage.pial_left, interpolation='linear')
        
        plotting.plot_surf_stat_map(
            fsaverage.infl_left,
            texture,
            hemi='left',
            view='lateral',
            threshold=2.5,
            bg_map=fsaverage.sulc_left,
            cmap='hot',
            colorbar=True if idx == 2 else False,
            axes=ax,
            title=None
        )
        
        ax.set_title(
            cond.replace('_', ' ').title(),
            fontsize=13,
            fontweight='bold',
            pad=6,
            color='black'
        )
    
    fig.suptitle(
        "Auditory Cortex Activation Across Speech Conditions (Healthy Controls)",
        fontsize=15,
        fontweight='bold',
        y=1.02
    )
    
    out_multi = OUTPUT_DIR / "multi_condition_auditory_profile.png"
    plt.savefig(str(out_multi), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  ✓ Saved Multi-Condition Profile: {out_multi.name}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("JOURNAL-QUALITY VISUAL ASSET GENERATION")
    logger.info("=" * 80)
    
    # 1. 3D Multi-View Renders for all conditions
    logger.info("\n" + "=" * 80)
    logger.info("1. GENERATING 3D MULTI-VIEW SURFACE RENDERS")
    logger.info("=" * 80)
    
    conditions = ['sentences_vs_baseline', 'words_vs_baseline', 'reversed_vs_baseline']
    
    for cond in conditions:
        logger.info(f"\nProcessing {cond}...")
        
        # Healthy Controls
        render_surface_multiview(
            STAT_MAPS_DIR / f"HC_{cond}_zmap.nii.gz",
            OUTPUT_DIR / f"3D_multiview_HC_{cond}.png",
            title=f"Healthy Controls: {cond.replace('_', ' ').title()}",
            z_thresh=2.5,
            cmap='hot'
        )
        
        # Schizophrenia Patients
        render_surface_multiview(
            STAT_MAPS_DIR / f"SCHZ_{cond}_zmap.nii.gz",
            OUTPUT_DIR / f"3D_multiview_SCHZ_{cond}.png",
            title=f"Schizophrenia Patients: {cond.replace('_', ' ').title()}",
            z_thresh=2.5,
            cmap='hot'
        )
        
        # Group Difference
        render_surface_multiview(
            STAT_MAPS_DIR / f"HC_gt_SCHZ_{cond}_zmap.nii.gz",
            OUTPUT_DIR / f"3D_multiview_diff_{cond}.png",
            title=f"HC > SCHZ Group Difference: {cond.replace('_', ' ').title()}",
            z_thresh=2.0,
            cmap='cold_hot'
        )
    
    # 2. Orthogonal Focal Views for Heschl's Gyrus Hypoactivation
    logger.info("\n" + "=" * 80)
    logger.info("2. GENERATING ORTHOGONAL FOCAL VIEWS")
    logger.info("=" * 80)
    
    render_orthogonal_focus(
        STAT_MAPS_DIR / "HC_gt_SCHZ_sentences_vs_baseline_zmap.nii.gz",
        OUTPUT_DIR / "ortho_heschl_gyrus_sentences_diff.png",
        title="Focal Hypoactivation in Left Heschl's Gyrus (Sentences: HC > SCHZ)",
        cut_coords=(-56, -2, 2),
        z_thresh=2.0,
        cmap='cold_hot'
    )
    
    render_orthogonal_focus(
        STAT_MAPS_DIR / "HC_gt_SCHZ_reversed_vs_baseline_zmap.nii.gz",
        OUTPUT_DIR / "ortho_heschl_gyrus_reversed_diff.png",
        title="Focal Hypoactivation in Primary Auditory Cortex (Reversed: HC > SCHZ)",
        cut_coords=(-56, -2, 2),
        z_thresh=2.0,
        cmap='cold_hot'
    )
    
    render_orthogonal_focus(
        STAT_MAPS_DIR / "HC_gt_SCHZ_words_vs_baseline_zmap.nii.gz",
        OUTPUT_DIR / "ortho_heschl_gyrus_words_diff.png",
        title="Focal Hypoactivation in Primary Auditory Cortex (Words: HC > SCHZ)",
        cut_coords=(-56, -2, 2),
        z_thresh=2.0,
        cmap='cold_hot'
    )
    
    # 3. ROI Boxplots
    logger.info("\n" + "=" * 80)
    logger.info("3. GENERATING ROI PARAMETER BOXPLOTS")
    logger.info("=" * 80)
    
    generate_heschl_gyrus_boxplots()
    
    # 4. Multi-Condition Auditory Profile
    logger.info("\n" + "=" * 80)
    logger.info("4. GENERATING MULTI-CONDITION AUDITORY PROFILE")
    logger.info("=" * 80)
    
    generate_multi_condition_comparison()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nAll high-resolution figures saved to: {OUTPUT_DIR}")
    logger.info("\nGenerated files:")
    
    # List all generated files
    generated_files = sorted(OUTPUT_DIR.glob("*.png"))
    for f in generated_files:
        logger.info(f"  - {f.name}")
    
    logger.info("=" * 80)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Publication-Grade GLM Figure Generation for Presentation
PLoS ONE Reproduction: Speech Perception in Schizophrenia

This script generates high-quality composite figures matching PLoS ONE standards:
1. Intracranial brain masking (no edge bleed)
2. Two-stage thresholding (Z ≥ 2.3, k ≥ 50 voxels)
3. Multi-slice axial layouts on MNI152 background
4. 3D cortical surface projections on fsaverage5

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

from nilearn import datasets, image, glm, plotting, surface
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/fMRI/logs/publication_figures.log'),
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
SECOND_LEVEL_DIR = ANALYSIS_DIR / 'second_level'
STAT_MAPS_DIR = SECOND_LEVEL_DIR / 'stat_maps'
OUTPUT_PPT_DIR = ANALYSIS_DIR / 'figures_for_ppt'

# Create output directories
OUTPUT_PPT_DIR.mkdir(parents=True, exist_ok=True)
STAT_MAPS_DIR.mkdir(parents=True, exist_ok=True)

# Load templates and surfaces
logger.info("Loading MNI152 template and brain mask...")
mni_template = datasets.load_mni152_template(resolution=2)
mni_mask = datasets.load_mni152_brain_mask(resolution=2)

logger.info("Loading fsaverage5 surfaces...")
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')

# Visualization parameters
CUT_COORDS = [-12, -2, 8, 18, 28, 38, 48]
DPI = 400

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_clean_participants():
    """Load and clean participant demographics."""
    df = pd.read_csv(DATA_DIR / 'participants.tsv', sep='\t')
    df = df[df['participant_id'].notna()].copy()
    df['subject_id'] = df['participant_id'].str.replace('sub-', '')
    df['iq'] = df['iq'].fillna(df['iq'].median())
    df['sex_num'] = (df['sex'] == 'male').astype(int)
    
    logger.info(f"Loaded {len(df)} participants")
    logger.info(f"  HC: {len(df[df['group'] == 'HC'])}")
    logger.info(f"  AVH+: {len(df[df['group'] == 'AVH+'])}")
    logger.info(f"  AVH-: {len(df[df['group'] == 'AVH-'])}")
    
    return df

# ============================================================================
# SECOND-LEVEL MODEL FUNCTIONS
# ============================================================================

def run_or_load_group_model(df, contrast, group_name):
    """
    Run or load group-level one-sample t-test.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Participant dataframe
    contrast : str
        Contrast name (e.g., 'words_vs_baseline')
    group_name : str
        Group name ('HC', 'SCHZ', 'AVH+', 'AVH-')
    
    Returns:
    --------
    zmap : nibabel.Nifti1Image
        Group-level z-score map
    """
    out_path = STAT_MAPS_DIR / f"{group_name}_{contrast}_zmap.nii.gz"
    
    if out_path.exists():
        logger.info(f"  Loaded existing group map: {out_path.name}")
        return nib.load(str(out_path))
    
    logger.info(f"  Computing new group map: {group_name} - {contrast}")
    
    # Select subjects based on group
    if group_name == 'HC':
        sub_df = df[df['group'] == 'HC']
    elif group_name == 'SCHZ':
        # Combine AVH+ and AVH- groups
        sub_df = df[df['group'].isin(['AVH+', 'AVH-'])]
    else:
        sub_df = df[df['group'] == group_name]
    
    # Collect first-level maps and covariates
    maps = []
    covars = []
    
    for _, row in sub_df.iterrows():
        sid = row['subject_id']
        
        # Try both naming conventions
        fpath = FIRST_LEVEL_DIR / f"sub-{sid}_{contrast}_zmap.nii.gz"
        if not fpath.exists():
            fpath = FIRST_LEVEL_DIR / f"{sid}_{contrast}_zmap.nii.gz"
        
        if fpath.exists():
            maps.append(str(fpath))
            covars.append({
                'age': row['age'],
                'sex': row['sex_num'],
                'iq': row['iq']
            })
    
    if len(maps) == 0:
        logger.error(f"  No maps found for {group_name} - {contrast}")
        return None
    
    logger.info(f"    Found {len(maps)} subjects for {group_name}")
    
    # Build design matrix with covariates
    dm = pd.DataFrame(covars)
    dm['intercept'] = 1.0
    
    # Standardize continuous covariates
    for col in ['age', 'iq']:
        dm[col] = (dm[col] - dm[col].mean()) / dm[col].std()
    
    # Fit second-level model
    model = SecondLevelModel(smoothing_fwhm=None, mask_img=mni_mask)
    model.fit(maps, design_matrix=dm[['intercept', 'age', 'sex', 'iq']])
    
    # Compute one-sample t-test (intercept)
    zmap = model.compute_contrast('intercept', output_type='z_score')
    
    # Save to disk
    nib.save(zmap, str(out_path))
    logger.info(f"  Saved: {out_path.name}")
    
    return zmap

def run_or_load_two_sample(df, contrast):
    """
    Run or load two-sample t-test (HC > SCHZ).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Participant dataframe
    contrast : str
        Contrast name
    
    Returns:
    --------
    zmap : nibabel.Nifti1Image
        Two-sample z-score map
    """
    out_path = STAT_MAPS_DIR / f"HC_gt_SCHZ_{contrast}_zmap.nii.gz"
    
    if out_path.exists():
        logger.info(f"  Loaded existing two-sample map: {out_path.name}")
        return nib.load(str(out_path))
    
    logger.info(f"  Computing new two-sample map: HC > SCHZ - {contrast}")
    
    # Separate groups
    hc_df = df[df['group'] == 'HC']
    scz_df = df[df['group'].isin(['AVH+', 'AVH-'])]
    
    # Collect maps and covariates
    maps = []
    covars = []
    
    # HC subjects (group = 1)
    for _, row in hc_df.iterrows():
        sid = row['subject_id']
        fpath = FIRST_LEVEL_DIR / f"sub-{sid}_{contrast}_zmap.nii.gz"
        if not fpath.exists():
            fpath = FIRST_LEVEL_DIR / f"{sid}_{contrast}_zmap.nii.gz"
        
        if fpath.exists():
            maps.append(str(fpath))
            covars.append({
                'group': 1,
                'age': row['age'],
                'sex': row['sex_num'],
                'iq': row['iq']
            })
    
    # SCHZ subjects (group = -1)
    for _, row in scz_df.iterrows():
        sid = row['subject_id']
        fpath = FIRST_LEVEL_DIR / f"sub-{sid}_{contrast}_zmap.nii.gz"
        if not fpath.exists():
            fpath = FIRST_LEVEL_DIR / f"{sid}_{contrast}_zmap.nii.gz"
        
        if fpath.exists():
            maps.append(str(fpath))
            covars.append({
                'group': -1,
                'age': row['age'],
                'sex': row['sex_num'],
                'iq': row['iq']
            })
    
    if len(maps) == 0:
        logger.error(f"  No maps found for HC > SCHZ - {contrast}")
        return None
    
    logger.info(f"    Found {len([c for c in covars if c['group'] == 1])} HC, "
                f"{len([c for c in covars if c['group'] == -1])} SCHZ")
    
    # Build design matrix
    dm = pd.DataFrame(covars)
    dm['intercept'] = 1.0
    
    # Standardize continuous covariates
    for col in ['age', 'iq']:
        dm[col] = (dm[col] - dm[col].mean()) / dm[col].std()
    
    # Fit second-level model
    model = SecondLevelModel(smoothing_fwhm=None, mask_img=mni_mask)
    model.fit(maps, design_matrix=dm[['intercept', 'group', 'age', 'sex', 'iq']])
    
    # Compute two-sample t-test (group contrast)
    zmap = model.compute_contrast('group', output_type='z_score')
    
    # Save to disk
    nib.save(zmap, str(out_path))
    logger.info(f"  Saved: {out_path.name}")
    
    return zmap

# ============================================================================
# MASKING AND THRESHOLDING FUNCTIONS
# ============================================================================

def mask_and_clean_zmap(raw_zmap, threshold=2.3, cluster_size=50, two_sided=False):
    """
    Apply intracranial masking and cluster-extent thresholding.
    
    Parameters:
    -----------
    raw_zmap : nibabel.Nifti1Image
        Raw z-score map
    threshold : float
        Height threshold (Z ≥ threshold)
    cluster_size : int
        Minimum cluster size in voxels
    two_sided : bool
        Whether to apply two-sided thresholding
    
    Returns:
    --------
    thresh_img : nibabel.Nifti1Image
        Thresholded and masked map
    """
    # Resample z-map to match MNI mask resolution
    resampled_zmap = image.resample_to_img(raw_zmap, mni_mask, interpolation='continuous')
    
    # Apply intracranial mask
    masked = image.math_img("img * mask", img=resampled_zmap, mask=mni_mask)
    
    # Apply cluster-extent thresholding
    thresh_img, _ = threshold_stats_img(
        masked,
        threshold=threshold,
        cluster_threshold=cluster_size,
        two_sided=two_sided,
        height_control=None
    )
    
    return thresh_img

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_publication_composite(contrast_name, hc_zmap, scz_zmap, diff_zmap=None, out_png=None):
    """
    Generate publication-ready multi-row composite figure.
    
    Layout:
    - Row 1: HC group (axial slices + left/right surface)
    - Row 2: SCHZ group (axial slices + left/right surface)
    - Row 3: HC > SCHZ difference (axial slices + left/right surface)
    
    Parameters:
    -----------
    contrast_name : str
        Name of the contrast
    hc_zmap : nibabel.Nifti1Image
        HC group z-map
    scz_zmap : nibabel.Nifti1Image
        SCHZ group z-map
    diff_zmap : nibabel.Nifti1Image or None
        HC > SCHZ difference z-map
    out_png : Path
        Output file path
    """
    rows = 3 if diff_zmap is not None else 2
    fig = plt.figure(figsize=(18, 6.2 * rows))
    gs = GridSpec(rows * 2, 4, figure=fig, hspace=0.55, wspace=0.2, height_ratios=[1.2, 1.0] * rows)
    
    # Define data for each row
    data_tuples = [
        ("Healthy Controls (HC)", hc_zmap, 'hot', False, 2.3, 100, 0),
        ("Schizophrenia Patients (SCHZ)", scz_zmap, 'hot', False, 2.3, 100, 1),
    ]
    
    if diff_zmap is not None:
        data_tuples.append(("HC > SCHZ (Hypoactivation)", diff_zmap, 'cold_hot', True, 2.0, 30, 2))
    
    for label, raw_map, cmap, two_sided, z_thresh, k_thresh, r_idx in data_tuples:
        logger.info(f"  Plotting: {label}")
        
        # Apply masking and thresholding
        clean_map = mask_and_clean_zmap(
            raw_map,
            threshold=z_thresh,
            cluster_size=k_thresh,
            two_sided=two_sided
        )
        
        # 1. Axial slice gallery
        ax_axial = fig.add_subplot(gs[r_idx * 2, :])
        plotting.plot_stat_map(
            clean_map,
            bg_img=mni_template,
            display_mode='z',
            cut_coords=CUT_COORDS,
            threshold=z_thresh,
            title=None,  # Suppress internal black box title
            cmap=cmap,
            black_bg=False,
            draw_cross=False,
            axes=ax_axial
        )
        ax_axial.set_title(
            f"{label}: {contrast_name.replace('_', ' ').title()}",
            fontsize=13,
            fontweight='bold',
            color='black',
            loc='center',
            pad=10
        )
        
        # 2. 3D Left lateral surface
        ax_surf_l = fig.add_subplot(gs[r_idx * 2 + 1, 0:2], projection='3d')
        tex_l = surface.vol_to_surf(clean_map, fsaverage.pial_left)
        plotting.plot_surf_stat_map(
            fsaverage.infl_left,
            tex_l,
            hemi='left',
            view='lateral',
            threshold=z_thresh,
            bg_map=fsaverage.sulc_left,
            cmap=cmap,
            title=None,
            axes=ax_surf_l
        )
        ax_surf_l.set_title(f"{label} - Left Hemisphere", fontsize=11, color='black', pad=6)
        
        # 3. 3D Right lateral surface
        ax_surf_r = fig.add_subplot(gs[r_idx * 2 + 1, 2:4], projection='3d')
        tex_r = surface.vol_to_surf(clean_map, fsaverage.pial_right)
        plotting.plot_surf_stat_map(
            fsaverage.infl_right,
            tex_r,
            hemi='right',
            view='lateral',
            threshold=z_thresh,
            bg_map=fsaverage.sulc_right,
            cmap=cmap,
            title=None,
            axes=ax_surf_r
        )
        ax_surf_r.set_title(f"{label} - Right Hemisphere", fontsize=11, color='black', pad=6)
    
    # Save figure
    plt.savefig(str(out_png), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  ✓ Successfully generated: {out_png.name}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("PUBLICATION-GRADE FIGURE GENERATION")
    logger.info("=" * 80)
    
    # Load participant data
    logger.info("\nLoading participants...")
    df = load_clean_participants()
    
    # Define contrasts
    contrasts = ['words_vs_baseline', 'sentences_vs_baseline', 'reversed_vs_baseline']
    
    # Process each contrast
    for contrast in contrasts:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing: {contrast}")
        logger.info(f"{'=' * 80}")
        
        # Load or compute group maps
        logger.info("\n1. Loading/computing group maps...")
        hc_map = run_or_load_group_model(df, contrast, 'HC')
        scz_map = run_or_load_group_model(df, contrast, 'SCHZ')
        
        # Load or compute two-sample map
        logger.info("\n2. Loading/computing two-sample map...")
        diff_map = run_or_load_two_sample(df, contrast)
        
        # Generate publication composite
        logger.info("\n3. Generating publication composite...")
        out_name = f"fig_{contrast}_presentation.png"
        
        # Skip difference map for words_vs_baseline (as per user request)
        plot_publication_composite(
            contrast_name=contrast,
            hc_zmap=hc_map,
            scz_zmap=scz_map,
            diff_zmap=diff_map if contrast != 'words_vs_baseline' else None,
            out_png=OUTPUT_PPT_DIR / out_name
        )
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("ALL FIGURES SAVED TO: analysis/figures_for_ppt/")
    logger.info("=" * 80)
    
    # List generated files
    logger.info("\nGenerated files:")
    for f in sorted(OUTPUT_PPT_DIR.glob("fig_*_presentation.png")):
        logger.info(f"  - {f.name}")

if __name__ == '__main__':
    main()

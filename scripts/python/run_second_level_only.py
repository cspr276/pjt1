#!/usr/bin/env python3
"""
Second-Level GLM Analysis Only
Skip first-level (already complete) and run second-level + visualizations
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
from datetime import datetime
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
from nilearn import plotting, image
import matplotlib.pyplot as plt

# Setup
PROJECT_ROOT = Path('/root/fMRI')
ANALYSIS_DIR = PROJECT_ROOT / 'analysis'
FIRST_LEVEL_DIR = ANALYSIS_DIR / 'first_level' / 'contrast_maps'
SECOND_LEVEL_DIR = ANALYSIS_DIR / 'second_level'
PARTICIPANTS_FILE = PROJECT_ROOT / 'ds004302-download' / 'participants.tsv'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_participants():
    """Load participant data"""
    df = pd.read_csv(PARTICIPANTS_FILE, sep='\t')
    df = df[df['participant_id'].notna()]
    df['subject_id'] = df['participant_id'].str.replace('sub-', '')
    
    # Handle missing IQ values - fill with median
    df['iq'] = df['iq'].fillna(df['iq'].median())
    
    # Encode sex as numeric (male=1, female=0)
    df['sex_numeric'] = (df['sex'] == 'male').astype(int)
    
    return df

def run_second_level_glm(participants_df, contrast_name, group):
    """Run second-level GLM for a specific contrast and group"""
    logger.info("=" * 80)
    logger.info(f"Second-Level Analysis: {contrast_name} - Group: {group}")
    logger.info("=" * 80)
    
    # Filter by group
    group_df = participants_df[participants_df['group'] == group].copy()
    
    logger.info(f"Group {group}: {len(group_df)} subjects")
    
    # Load contrast maps
    contrast_maps = []
    subject_data = []
    
    for _, row in group_df.iterrows():
        subject_id = row['subject_id']
        # Files have "sub-" prefix
        map_path = FIRST_LEVEL_DIR / f"sub-{subject_id}_{contrast_name}_zmap.nii.gz"
        
        if map_path.exists():
            contrast_maps.append(str(map_path))
            subject_data.append({
                'subject_id': subject_id,
                'age': row['age'],
                'sex': row['sex_numeric'],
                'iq': row['iq']
            })
        else:
            logger.warning(f"Missing map: {map_path}")
    
    if len(contrast_maps) < 5:
        logger.error(f"Insufficient subjects ({len(contrast_maps)}) for {group} {contrast_name}")
        return None
    
    logger.info(f"Loaded {len(contrast_maps)} contrast maps")
    
    # Create design matrix with covariates
    design_matrix = pd.DataFrame(subject_data)
    design_matrix['intercept'] = 1
    
    # Normalize covariates
    for col in ['age', 'iq']:
        design_matrix[col] = (design_matrix[col] - design_matrix[col].mean()) / design_matrix[col].std()
    
    # Keep only numeric columns for the design matrix
    design_matrix_numeric = design_matrix[['intercept', 'age', 'sex', 'iq']].copy()
    
    logger.info(f"Design matrix shape: {design_matrix_numeric.shape}")
    logger.info(f"Design matrix columns: {list(design_matrix_numeric.columns)}")
    
    # Fit second-level model
    second_level_model = SecondLevelModel(smoothing_fwhm=None)
    second_level_model.fit(contrast_maps, design_matrix=design_matrix_numeric)
    
    # Compute one-sample t-test (intercept)
    z_map = second_level_model.compute_contrast('intercept', output_type='z_score')
    
    # Save unthresholded map
    output_dir = SECOND_LEVEL_DIR / 'stat_maps'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f'{group}_{contrast_name}_zmap.nii.gz'
    nib.save(z_map, str(output_path))
    logger.info(f"Saved unthresholded z-map: {output_path}")
    
    # Apply cluster thresholding
    thresholded_map, threshold = threshold_stats_img(
        z_map,
        alpha=0.05,
        height_control='fdr',
        cluster_threshold=10
    )
    
    # Save thresholded map
    thresholded_path = output_dir / f'{group}_{contrast_name}_thresholded.nii.gz'
    nib.save(thresholded_map, str(thresholded_path))
    logger.info(f"Saved thresholded map: {thresholded_path}")
    
    return z_map, thresholded_map

def run_two_sample_ttest(participants_df, contrast_name):
    """Run two-sample t-test comparing AVH+ vs HC"""
    logger.info("=" * 80)
    logger.info(f"Two-Sample t-test: {contrast_name} (AVH+ > HC)")
    logger.info("=" * 80)
    
    # Separate groups - compare AVH+ vs HC
    hc_df = participants_df[participants_df['group'] == 'HC'].copy()
    avh_df = participants_df[participants_df['group'] == 'AVH+'].copy()
    
    logger.info(f"HC: {len(hc_df)} subjects, AVH+: {len(avh_df)} subjects")
    
    # Load contrast maps
    all_maps = []
    all_subjects = []
    
    for _, row in hc_df.iterrows():
        subject_id = row['subject_id']
        # Files have "sub-" prefix
        map_path = FIRST_LEVEL_DIR / f"sub-{subject_id}_{contrast_name}_zmap.nii.gz"
        if map_path.exists():
            all_maps.append(str(map_path))
            all_subjects.append({
                'subject_id': subject_id,
                'group': 0,  # HC
                'age': row['age'],
                'sex': row['sex_numeric'],
                'iq': row['iq']
            })
    
    for _, row in avh_df.iterrows():
        subject_id = row['subject_id']
        # Files have "sub-" prefix
        map_path = FIRST_LEVEL_DIR / f"sub-{subject_id}_{contrast_name}_zmap.nii.gz"
        if map_path.exists():
            all_maps.append(str(map_path))
            all_subjects.append({
                'subject_id': subject_id,
                'group': 1,  # AVH+
                'age': row['age'],
                'sex': row['sex_numeric'],
                'iq': row['iq']
            })
    
    if len(all_maps) < 10:
        logger.error(f"Insufficient subjects ({len(all_maps)}) for two-sample t-test")
        return None
    
    logger.info(f"Loaded {len(all_maps)} contrast maps")
    
    # Create design matrix
    design_matrix = pd.DataFrame(all_subjects)
    design_matrix['intercept'] = 1
    
    # Normalize covariates
    for col in ['age', 'iq']:
        design_matrix[col] = (design_matrix[col] - design_matrix[col].mean()) / design_matrix[col].std()
    
    # Keep only numeric columns for the design matrix
    design_matrix_numeric = design_matrix[['intercept', 'group', 'age', 'sex', 'iq']].copy()
    
    logger.info(f"Design matrix shape: {design_matrix_numeric.shape}")
    
    # Fit second-level model
    second_level_model = SecondLevelModel(smoothing_fwhm=None)
    second_level_model.fit(all_maps, design_matrix=design_matrix_numeric)
    
    # Compute contrast: group1 > group2
    z_map = second_level_model.compute_contrast('group', output_type='z_score')
    
    # Save unthresholded map
    output_dir = SECOND_LEVEL_DIR / 'stat_maps'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f'AVH_gt_HC_{contrast_name}_zmap.nii.gz'
    nib.save(z_map, str(output_path))
    logger.info(f"Saved unthresholded z-map: {output_path}")
    
    # Apply cluster thresholding
    thresholded_map, threshold = threshold_stats_img(
        z_map,
        alpha=0.05,
        height_control='fdr',
        cluster_threshold=10
    )
    
    # Save thresholded map
    thresholded_path = output_dir / f'AVH_gt_HC_{contrast_name}_thresholded.nii.gz'
    nib.save(thresholded_map, str(thresholded_path))
    logger.info(f"Saved thresholded map: {thresholded_path}")
    
    return z_map, thresholded_map

def generate_publication_figures(participants_df):
    """Generate publication-grade figures"""
    logger.info("=" * 80)
    logger.info("GENERATING PUBLICATION FIGURES")
    logger.info("=" * 80)
    
    figures_dir = SECOND_LEVEL_DIR / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    contrasts = ['words_vs_baseline', 'sentences_vs_baseline', 'reversed_vs_baseline']
    groups = ['HC', 'AVH+', 'AVH-']
    
    # Generate group-level activation maps
    for contrast in contrasts:
        for group in groups:
            zmap_path = SECOND_LEVEL_DIR / 'stat_maps' / f'{group}_{contrast}_zmap.nii.gz'
            if zmap_path.exists():
                logger.info(f"Generating figure for {group} - {contrast}")
                
                z_map = nib.load(str(zmap_path))
                
                # Create glass brain plot
                fig, ax = plt.subplots(figsize=(12, 6))
                plotting.plot_glass_brain(
                    z_map,
                    threshold=2.3,
                    title=f'{group}: {contrast.replace("_", " ").title()}',
                    display_mode='lyrz',
                    colorbar=True,
                    plot_abs=False,
                    figure=fig
                )
                
                output_path = figures_dir / f'{group}_{contrast}_glass_brain.png'
                fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved: {output_path}")
    
    # Generate two-sample t-test figures
    for contrast in contrasts:
        zmap_path = SECOND_LEVEL_DIR / 'stat_maps' / f'AVH_gt_HC_{contrast}_zmap.nii.gz'
        if zmap_path.exists():
            logger.info(f"Generating two-sample t-test figure for {contrast}")
            
            z_map = nib.load(str(zmap_path))
            
            # Create glass brain plot
            fig, ax = plt.subplots(figsize=(12, 6))
            plotting.plot_glass_brain(
                z_map,
                threshold=2.3,
                title=f'AVH+ > HC: {contrast.replace("_", " ").title()}',
                display_mode='lyrz',
                colorbar=True,
                plot_abs=False,
                figure=fig
            )
            
            output_path = figures_dir / f'AVH_gt_HC_{contrast}_glass_brain.png'
            fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved: {output_path}")
    
    logger.info("✓ Publication figures generated")

def main():
    logger.info("=" * 80)
    logger.info("SECOND-LEVEL GLM ANALYSIS")
    logger.info("=" * 80)
    
    # Load participants
    participants_df = load_participants()
    logger.info(f"Loaded participant data: {len(participants_df)} subjects")
    
    # Define contrasts
    contrasts = ['words_vs_baseline', 'sentences_vs_baseline', 'reversed_vs_baseline']
    groups = ['HC', 'AVH+', 'AVH-']
    
    # Run second-level analyses
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: ONE-SAMPLE T-TESTS (GROUP-LEVEL ACTIVATION)")
    logger.info("=" * 80)
    
    for contrast in contrasts:
        for group in groups:
            try:
                run_second_level_glm(participants_df, contrast, group)
            except Exception as e:
                logger.error(f"Failed {group} {contrast}: {e}")
    
    # Run two-sample t-tests
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: TWO-SAMPLE T-TESTS (GROUP COMPARISONS)")
    logger.info("=" * 80)
    
    for contrast in contrasts:
        try:
            run_two_sample_ttest(participants_df, contrast)
        except Exception as e:
            logger.error(f"Failed two-sample t-test for {contrast}: {e}")
    
    # Generate publication figures
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: PUBLICATION FIGURES")
    logger.info("=" * 80)
    
    generate_publication_figures(participants_df)
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ SECOND-LEVEL ANALYSIS COMPLETE")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()

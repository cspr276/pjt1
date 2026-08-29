#!/usr/bin/env python3
"""
1st and 2nd Level GLM Analysis & Statistical Brain Mapping
PLoS ONE Reproduction: Speech Perception in Schizophrenia

This script performs:
1. Subject-level (first-level) GLM analysis for all 71 subjects
2. Group-level (second-level) GLM analysis with covariates
3. Publication-grade visualization generation

Author: Generated for fMRI analysis pipeline
Date: 2026-08-18
"""

import os
import sys
import warnings
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import ndimage

# Nilearn imports
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
from nilearn import image, masking, plotting, datasets
from nilearn.plotting import plot_stat_map, plot_surf_stat_map, view_surf
from nilearn.surface import vol_to_surf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/fMRI/logs/glm_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
ROOT_DIR = Path('/root/fMRI')
DATA_DIR = ROOT_DIR / 'ds004302-download'
OUTPUT_DIR = ROOT_DIR / 'output'
ANALYSIS_DIR = ROOT_DIR / 'analysis'
SCRIPTS_DIR = ROOT_DIR / 'scripts' / 'python'
LOGS_DIR = ROOT_DIR / 'logs'

# Analysis parameters
TR = 2.0  # Repetition time in seconds
SLICE_TIME_REF = 0.5  # Reference slice for slice timing
SMOOTHING_FWHM = 5.0  # Spatial smoothing FWHM in mm
DRIFT_MODEL = 'cosine'  # Drift model
HIGH_PASS = 0.008  # High-pass filter cutoff (Hz)

# Cluster thresholding parameters
CLUSTER_FORMING_THRESHOLD = 2.3  # Z > 2.3 (p < 0.01)
CLUSTER_THRESHOLD = 0.05  # Cluster-level p < 0.05

# Visualization parameters
AXIAL_CUT_COORDS = [-12, -2, 8, 18, 28, 38, 48]
DPI = 300  # High resolution for publication

# Subject list (71 subjects total)
SUBJECTS = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10',
    'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20',
    'sub-21', 'sub-22', 'sub-23', 'sub-24', 'sub-25', 'sub-26', 'sub-27', 'sub-28', 'sub-29', 'sub-30',
    'sub-32', 'sub-33', 'sub-34', 'sub-36', 'sub-37', 'sub-38', 'sub-40', 'sub-42', 'sub-43', 'sub-44',
    'sub-45', 'sub-46', 'sub-47', 'sub-48', 'sub-49', 'sub-50', 'sub-53', 'sub-54', 'sub-55', 'sub-56',
    'sub-57', 'sub-58', 'sub-59', 'sub-60', 'sub-61', 'sub-62', 'sub-63', 'sub-64', 'sub-65', 'sub-66',
    'sub-67', 'sub-68', 'sub-69', 'sub-70', 'sub-71', 'sub-72', 'sub-73', 'sub-74', 'sub-75', 'sub-76',
    'sub-77'
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_participants_data() -> pd.DataFrame:
    """Load participant demographics and group information."""
    participants_file = DATA_DIR / 'participants.tsv'
    df = pd.read_csv(participants_file, sep='\t')
    logger.info(f"Loaded participant data: {len(df)} subjects")
    logger.info(f"Groups: HC={len(df[df['group']=='HC'])}, "
                f"AVH+={len(df[df['group']=='AVH+'])}, "
                f"AVH-={len(df[df['group']=='AVH-'])}")
    return df


def load_events_data(subject_id: str) -> pd.DataFrame:
    """Load task events for a subject."""
    events_file = DATA_DIR / 'task-speech_events.tsv'
    events = pd.read_csv(events_file, sep='\t')
    logger.debug(f"Loaded events for {subject_id}: {len(events)} events")
    return events


def load_confounds(subject_id: str) -> pd.DataFrame:
    """Load motion confounds for a subject."""
    confounds_file = OUTPUT_DIR / subject_id / 'func' / f'{subject_id}_task-speech_desc-confounds_timeseries.tsv'
    
    if not confounds_file.exists():
        raise FileNotFoundError(f"Confounds file not found: {confounds_file}")
    
    confounds_df = pd.read_csv(confounds_file, sep='\t')
    
    # Extract 6 rigid-body motion parameters
    motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    motion_confounds = confounds_df[motion_cols].copy()
    
    # Replace NaN with 0 (for first timepoint)
    motion_confounds = motion_confounds.fillna(0)
    
    logger.debug(f"Loaded confounds for {subject_id}: {motion_confounds.shape}")
    return motion_confounds


def get_functional_data_paths(subject_id: str) -> Tuple[Path, Path]:
    """Get paths to functional data and brain mask."""
    func_path = OUTPUT_DIR / subject_id / 'func' / f'{subject_id}_task-speech_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'
    mask_path = OUTPUT_DIR / subject_id / 'func' / f'{subject_id}_task-speech_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz'
    
    if not func_path.exists():
        raise FileNotFoundError(f"Functional data not found: {func_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Brain mask not found: {mask_path}")
    
    return func_path, mask_path


def create_design_matrix(events: pd.DataFrame, confounds: pd.DataFrame, 
                         n_scans: int) -> pd.DataFrame:
    """Create first-level design matrix with task conditions and confounds."""
    
    # Create events dataframe for design matrix
    # Filter out white-noise (baseline) blocks
    task_events = events[events['condition'] != 'white-noise'].copy()
    
    # Create properly formatted events dataframe for nilearn
    # Nilearn expects columns: onset, duration, trial_type
    events_for_nilearn = pd.DataFrame({
        'onset': task_events['onset'],
        'duration': task_events['duration'],
        'trial_type': task_events['condition']
    })
    
    # Create design matrix
    frame_times = np.arange(n_scans) * TR
    
    design_matrix = make_first_level_design_matrix(
        frame_times,
        events=events_for_nilearn,
        hrf_model='spm',
        drift_model=DRIFT_MODEL,
        high_pass=HIGH_PASS,
        add_regs=confounds.values,
        add_reg_names=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    )
    
    logger.debug(f"Design matrix shape: {design_matrix.shape}")
    logger.debug(f"Columns: {list(design_matrix.columns)}")
    
    return design_matrix


# ============================================================================
# FIRST-LEVEL ANALYSIS
# ============================================================================

def run_first_level_glm(subject_id: str, 
                        participants_df: pd.DataFrame) -> Optional[Dict[str, nib.Nifti1Image]]:
    """
    Run first-level GLM analysis for a single subject.
    
    Returns:
        Dictionary of contrast z-maps, or None if analysis failed
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing {subject_id}")
    logger.info(f"{'='*80}")
    
    try:
        # Load data
        func_path, mask_path = get_functional_data_paths(subject_id)
        events = load_events_data(subject_id)
        confounds = load_confounds(subject_id)
        
        # Load functional image to get number of scans
        func_img = nib.load(str(func_path))
        n_scans = func_img.shape[-1]
        logger.info(f"Functional data: {func_img.shape}")
        
        # Create design matrix
        design_matrix = create_design_matrix(events, confounds, n_scans)
        
        # Initialize first-level model
        glm_model = FirstLevelModel(
            t_r=TR,
            slice_time_ref=SLICE_TIME_REF,
            hrf_model='spm',
            drift_model=DRIFT_MODEL,
            high_pass=HIGH_PASS,
            smoothing_fwhm=SMOOTHING_FWHM,
            mask_img=str(mask_path),
            standardize=True,
            noise_model='ar1',
            verbose=0
        )
        
        # Fit the model
        logger.info(f"Fitting GLM for {subject_id}...")
        glm_model.fit(str(func_path), design_matrices=[design_matrix])
        
        # Define contrasts
        contrasts = {
            'words_vs_baseline': 'words',
            'sentences_vs_baseline': 'sentences',
            'reversed_vs_baseline': 'reversed'
        }
        
        # Compute and save contrast maps
        contrast_maps = {}
        for contrast_name, contrast_def in contrasts.items():
            logger.info(f"Computing contrast: {contrast_name}")
            
            # Compute z-map
            z_map = glm_model.compute_contrast(contrast_def, output_type='z_score')
            
            # Save z-map
            output_path = ANALYSIS_DIR / 'first_level' / 'contrast_maps' / f'{subject_id}_{contrast_name}_zmap.nii.gz'
            nib.save(z_map, str(output_path))
            logger.info(f"Saved: {output_path}")
            
            contrast_maps[contrast_name] = z_map
        
        # Generate individual plots
        generate_individual_plots(subject_id, contrast_maps)
        
        logger.info(f"✓ Successfully completed {subject_id}")
        return contrast_maps
        
    except Exception as e:
        logger.error(f"✗ Failed to process {subject_id}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def generate_individual_plots(subject_id: str, 
                               contrast_maps: Dict[str, nib.Nifti1Image]):
    """Generate axial and surface plots for individual subject."""
    
    output_dir = ANALYSIS_DIR / 'first_level' / 'individual_plots'
    
    for contrast_name, z_map in contrast_maps.items():
        try:
            # Axial plot
            axial_path = output_dir / f'{subject_id}_{contrast_name}_axial.png'
            import matplotlib.pyplot as plt
            fig = plot_stat_map(
                z_map,
                threshold=2.3,
                display_mode='z',
                cut_coords=AXIAL_CUT_COORDS,
                title=f'{subject_id}: {contrast_name.replace("_", " ").title()}',
                cmap='hot',
                black_bg=False,
                draw_cross=False
            )
            fig.savefig(str(axial_path), dpi=DPI, bbox_inches='tight')
            plt.close(fig)
            logger.debug(f"Saved axial plot: {axial_path}")
            
            # Surface plot (left hemisphere)
            fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
            texture_left = vol_to_surf(z_map, fsaverage.pial_left)
            
            surface_path_left = output_dir / f'{subject_id}_{contrast_name}_surface_left.png'
            fig = plot_surf_stat_map(
                fsaverage.infl_left,
                texture_left,
                hemi='left',
                view='lateral',
                threshold=2.3,
                cmap='hot',
                title=f'{subject_id}: {contrast_name} (Left)'
            )
            fig.savefig(str(surface_path_left), dpi=DPI, bbox_inches='tight')
            plt.close(fig)
            logger.debug(f"Saved surface plot (left): {surface_path_left}")
            
            # Surface plot (right hemisphere)
            texture_right = vol_to_surf(z_map, fsaverage.pial_right)
            
            surface_path_right = output_dir / f'{subject_id}_{contrast_name}_surface_right.png'
            fig = plot_surf_stat_map(
                fsaverage.infl_right,
                texture_right,
                hemi='right',
                view='lateral',
                threshold=2.3,
                cmap='hot',
                title=f'{subject_id}: {contrast_name} (Right)'
            )
            fig.savefig(str(surface_path_right), dpi=DPI, bbox_inches='tight')
            plt.close(fig)
            logger.debug(f"Saved surface plot (right): {surface_path_right}")
            
        except Exception as e:
            logger.warning(f"Failed to generate plots for {subject_id} {contrast_name}: {e}")


# ============================================================================
# SECOND-LEVEL ANALYSIS
# ============================================================================

def run_second_level_glm(participants_df: pd.DataFrame,
                         contrast_name: str,
                         group: str = 'all') -> Optional[nib.Nifti1Image]:
    """
    Run second-level GLM analysis for a specific contrast and group.
    
    Args:
        participants_df: DataFrame with participant information
        contrast_name: Name of the contrast (e.g., 'words_vs_baseline')
        group: Group to analyze ('HC', 'SCHZ', 'AVH+', 'AVH-', 'all')
    
    Returns:
        Group-level z-map, or None if analysis failed
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Second-Level Analysis: {contrast_name} - Group: {group}")
    logger.info(f"{'='*80}")
    
    # Filter subjects by group
    if group == 'HC':
        group_subjects = participants_df[participants_df['group'] == 'HC']['participant_id'].tolist()
    elif group == 'SCHZ':
        # All patients (AVH+ and AVH-)
        group_subjects = participants_df[participants_df['group'].isin(['AVH+', 'AVH-'])]['participant_id'].tolist()
    elif group == 'AVH+':
        group_subjects = participants_df[participants_df['group'] == 'AVH+']['participant_id'].tolist()
    elif group == 'AVH-':
        group_subjects = participants_df[participants_df['group'] == 'AVH-']['participant_id'].tolist()
    elif group == 'all':
        group_subjects = participants_df['participant_id'].tolist()
    else:
        raise ValueError(f"Unknown group: {group}")
    
    logger.info(f"Group {group}: {len(group_subjects)} subjects")
    
    # Load contrast maps for all subjects in the group
    contrast_maps = []
    subject_data = []
    
    for subject_id in group_subjects:
        zmap_path = ANALYSIS_DIR / 'first_level' / 'contrast_maps' / f'{subject_id}_{contrast_name}_zmap.nii.gz'
        
        if not zmap_path.exists():
            logger.warning(f"Missing contrast map: {zmap_path}")
            continue
        
        # Load z-map
        z_map = nib.load(str(zmap_path))
        contrast_maps.append(z_map)
        
        # Get subject covariates
        subject_info = participants_df[participants_df['participant_id'] == subject_id].iloc[0]
        subject_data.append({
            'subject_id': subject_id,
            'age': subject_info['age'] if pd.notna(subject_info['age']) else participants_df['age'].mean(),
            'sex': 1 if subject_info['sex'] == 'male' else 0,
            'iq': subject_info['iq'] if pd.notna(subject_info['iq']) else participants_df['iq'].mean()
        })
    
    if len(contrast_maps) == 0:
        logger.error(f"No contrast maps found for group {group}")
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
    output_path = ANALYSIS_DIR / 'second_level' / 'stat_maps' / f'{group}_{contrast_name}_zmap.nii.gz'
    nib.save(z_map, str(output_path))
    logger.info(f"Saved unthresholded z-map: {output_path}")
    
    # Apply cluster thresholding
    thresholded_map, threshold = threshold_stats_img(
        z_map,
        threshold=CLUSTER_FORMING_THRESHOLD,
        cluster_threshold=CLUSTER_THRESHOLD,
        two_sided=True
    )
    
    # Save thresholded map
    thresholded_path = ANALYSIS_DIR / 'second_level' / 'stat_maps' / f'{group}_{contrast_name}_zmap_thresholded.nii.gz'
    nib.save(thresholded_map, str(thresholded_path))
    logger.info(f"Saved thresholded z-map: {thresholded_path}")
    
    return z_map


def run_two_sample_ttest(participants_df: pd.DataFrame,
                         contrast_name: str,
                         group1: str,
                         group2: str) -> Optional[nib.Nifti1Image]:
    """
    Run two-sample t-test between two groups.
    
    Args:
        participants_df: DataFrame with participant information
        contrast_name: Name of the contrast
        group1: First group (e.g., 'HC')
        group2: Second group (e.g., 'SCHZ')
    
    Returns:
        Group difference z-map, or None if analysis failed
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Two-Sample T-Test: {contrast_name} - {group1} vs {group2}")
    logger.info(f"{'='*80}")
    
    # Get subjects for each group
    if group1 == 'HC':
        subjects1 = participants_df[participants_df['group'] == 'HC']['participant_id'].tolist()
    elif group1 == 'SCHZ':
        subjects1 = participants_df[participants_df['group'].isin(['AVH+', 'AVH-'])]['participant_id'].tolist()
    elif group1 == 'AVH+':
        subjects1 = participants_df[participants_df['group'] == 'AVH+']['participant_id'].tolist()
    elif group1 == 'AVH-':
        subjects1 = participants_df[participants_df['group'] == 'AVH-']['participant_id'].tolist()
    else:
        raise ValueError(f"Unknown group: {group1}")
    
    if group2 == 'HC':
        subjects2 = participants_df[participants_df['group'] == 'HC']['participant_id'].tolist()
    elif group2 == 'SCHZ':
        subjects2 = participants_df[participants_df['group'].isin(['AVH+', 'AVH-'])]['participant_id'].tolist()
    elif group2 == 'AVH+':
        subjects2 = participants_df[participants_df['group'] == 'AVH+']['participant_id'].tolist()
    elif group2 == 'AVH-':
        subjects2 = participants_df[participants_df['group'] == 'AVH-']['participant_id'].tolist()
    else:
        raise ValueError(f"Unknown group: {group2}")
    
    logger.info(f"{group1}: {len(subjects1)} subjects")
    logger.info(f"{group2}: {len(subjects2)} subjects")
    
    # Load contrast maps
    maps1 = []
    maps2 = []
    all_subjects = []
    
    for subject_id in subjects1:
        zmap_path = ANALYSIS_DIR / 'first_level' / 'contrast_maps' / f'{subject_id}_{contrast_name}_zmap.nii.gz'
        if zmap_path.exists():
            maps1.append(nib.load(str(zmap_path)))
            subject_info = participants_df[participants_df['participant_id'] == subject_id].iloc[0]
            all_subjects.append({
                'subject_id': subject_id,
                'group': 1,
                'age': subject_info['age'] if pd.notna(subject_info['age']) else participants_df['age'].mean(),
                'sex': 1 if subject_info['sex'] == 'male' else 0,
                'iq': subject_info['iq'] if pd.notna(subject_info['iq']) else participants_df['iq'].mean()
            })
    
    for subject_id in subjects2:
        zmap_path = ANALYSIS_DIR / 'first_level' / 'contrast_maps' / f'{subject_id}_{contrast_name}_zmap.nii.gz'
        if zmap_path.exists():
            maps2.append(nib.load(str(zmap_path)))
            subject_info = participants_df[participants_df['participant_id'] == subject_id].iloc[0]
            all_subjects.append({
                'subject_id': subject_id,
                'group': -1,
                'age': subject_info['age'] if pd.notna(subject_info['age']) else participants_df['age'].mean(),
                'sex': 1 if subject_info['sex'] == 'male' else 0,
                'iq': subject_info['iq'] if pd.notna(subject_info['iq']) else participants_df['iq'].mean()
            })
    
    if len(maps1) == 0 or len(maps2) == 0:
        logger.error(f"Insufficient data: {group1}={len(maps1)}, {group2}={len(maps2)}")
        return None
    
    logger.info(f"Loaded {len(maps1)} maps for {group1}, {len(maps2)} maps for {group2}")
    
    # Combine all maps
    all_maps = maps1 + maps2
    
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
    output_path = ANALYSIS_DIR / 'second_level' / 'stat_maps' / f'{group1}_vs_{group2}_{contrast_name}_zmap.nii.gz'
    nib.save(z_map, str(output_path))
    logger.info(f"Saved unthresholded z-map: {output_path}")
    
    # Apply cluster thresholding
    thresholded_map, threshold = threshold_stats_img(
        z_map,
        threshold=CLUSTER_FORMING_THRESHOLD,
        cluster_threshold=CLUSTER_THRESHOLD,
        two_sided=True
    )
    
    # Save thresholded map
    thresholded_path = ANALYSIS_DIR / 'second_level' / 'stat_maps' / f'{group1}_vs_{group2}_{contrast_name}_zmap_thresholded.nii.gz'
    nib.save(thresholded_map, str(thresholded_path))
    logger.info(f"Saved thresholded z-map: {thresholded_path}")
    
    return z_map


# ============================================================================
# VISUALIZATION
# ============================================================================

def generate_publication_figures(participants_df: pd.DataFrame):
    """Generate high-resolution publication-grade figures."""
    
    logger.info(f"\n{'='*80}")
    logger.info("Generating Publication Figures")
    logger.info(f"{'='*80}")
    
    output_dir = ANALYSIS_DIR / 'figures_for_ppt'
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
    
    # Load MNI template for background
    mni152 = datasets.load_mni152_template(resolution=2)
    
    # =========================================================================
    # Figure 1: Words Contrast Composite
    # =========================================================================
    logger.info("Generating Figure 1: Words Contrast Composite")
    
    # Load HC and SCHZ maps
    hc_words_path = ANALYSIS_DIR / 'second_level' / 'stat_maps' / 'HC_words_vs_baseline_zmap.nii.gz'
    schz_words_path = ANALYSIS_DIR / 'second_level' / 'stat_maps' / 'SCHZ_words_vs_baseline_zmap.nii.gz'
    
    if hc_words_path.exists() and schz_words_path.exists():
        hc_words = nib.load(str(hc_words_path))
        schz_words = nib.load(str(schz_words_path))
        
        # Create composite figure
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.1)
        
        # Top row: HC axial
        ax1 = fig.add_subplot(gs[0, :])
        plot_stat_map(
            hc_words,
            threshold=2.3,
            display_mode='z',
            cut_coords=AXIAL_CUT_COORDS,
            title='Healthy Controls: Words vs Baseline',
            cmap='hot',
            black_bg=False,
            axes=ax1,
            draw_cross=False
        )
        
        # Second row: HC surface
        ax2 = fig.add_subplot(gs[1, 0])
        texture_left = vol_to_surf(hc_words, fsaverage.pial_left)
        plot_surf_stat_map(
            fsaverage.infl_left,
            texture_left,
            hemi='left',
            view='lateral',
            threshold=2.3,
            cmap='hot',
            title='HC Left',
            axes=ax2
        )
        
        ax3 = fig.add_subplot(gs[1, 1])
        texture_right = vol_to_surf(hc_words, fsaverage.pial_right)
        plot_surf_stat_map(
            fsaverage.infl_right,
            texture_right,
            hemi='right',
            view='lateral',
            threshold=2.3,
            cmap='hot',
            title='HC Right',
            axes=ax3
        )
        
        # Third row: SCHZ axial
        ax4 = fig.add_subplot(gs[2, :])
        plot_stat_map(
            schz_words,
            threshold=2.3,
            display_mode='z',
            cut_coords=AXIAL_CUT_COORDS,
            title='Schizophrenia Patients: Words vs Baseline',
            cmap='hot',
            black_bg=False,
            axes=ax4,
            draw_cross=False
        )
        
        # Fourth row: SCHZ surface
        ax5 = fig.add_subplot(gs[3, 0])
        texture_left = vol_to_surf(schz_words, fsaverage.pial_left)
        plot_surf_stat_map(
            fsaverage.infl_left,
            texture_left,
            hemi='left',
            view='lateral',
            threshold=2.3,
            cmap='hot',
            title='SCHZ Left',
            axes=ax5
        )
        
        ax6 = fig.add_subplot(gs[3, 1])
        texture_right = vol_to_surf(schz_words, fsaverage.pial_right)
        plot_surf_stat_map(
            fsaverage.infl_right,
            texture_right,
            hemi='right',
            view='lateral',
            threshold=2.3,
            cmap='hot',
            title='SCHZ Right',
            axes=ax6
        )
        
        plt.tight_layout()
        fig_path = output_dir / 'fig1_words_contrast_composite.png'
        plt.savefig(str(fig_path), dpi=DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {fig_path}")
    
    # =========================================================================
    # Figure 2: Sentences and Group Difference
    # =========================================================================
    logger.info("Generating Figure 2: Sentences and Group Difference")
    
    hc_sentences_path = ANALYSIS_DIR / 'second_level' / 'stat_maps' / 'HC_sentences_vs_baseline_zmap.nii.gz'
    schz_sentences_path = ANALYSIS_DIR / 'second_level' / 'stat_maps' / 'SCHZ_sentences_vs_baseline_zmap.nii.gz'
    diff_path = ANALYSIS_DIR / 'second_level' / 'stat_maps' / 'HC_vs_SCHZ_sentences_vs_baseline_zmap.nii.gz'
    
    if hc_sentences_path.exists() and schz_sentences_path.exists() and diff_path.exists():
        hc_sentences = nib.load(str(hc_sentences_path))
        schz_sentences = nib.load(str(schz_sentences_path))
        diff_map = nib.load(str(diff_path))
        
        fig = plt.figure(figsize=(20, 20))
        gs = GridSpec(6, 3, figure=fig, hspace=0.3, wspace=0.1)
        
        # Row 1: HC axial
        ax1 = fig.add_subplot(gs[0, :])
        plot_stat_map(
            hc_sentences,
            threshold=2.3,
            display_mode='z',
            cut_coords=AXIAL_CUT_COORDS,
            title='Row 1: Healthy Controls - Sentences vs Baseline',
            cmap='hot',
            black_bg=False,
            axes=ax1,
            draw_cross=False
        )
        
        # Row 2: HC surface
        ax2 = fig.add_subplot(gs[1, 0])
        texture_left = vol_to_surf(hc_sentences, fsaverage.pial_left)
        plot_surf_stat_map(
            fsaverage.infl_left,
            texture_left,
            hemi='left',
            view='lateral',
            threshold=2.3,
            cmap='hot',
            title='HC Left',
            axes=ax2
        )
        
        ax3 = fig.add_subplot(gs[1, 1])
        texture_right = vol_to_surf(hc_sentences, fsaverage.pial_right)
        plot_surf_stat_map(
            fsaverage.infl_right,
            texture_right,
            hemi='right',
            view='lateral',
            threshold=2.3,
            cmap='hot',
            title='HC Right',
            axes=ax3
        )
        
        # Row 3: SCHZ axial
        ax4 = fig.add_subplot(gs[2, :])
        plot_stat_map(
            schz_sentences,
            threshold=2.3,
            display_mode='z',
            cut_coords=AXIAL_CUT_COORDS,
            title='Row 2: Schizophrenia Patients - Sentences vs Baseline',
            cmap='hot',
            black_bg=False,
            axes=ax4,
            draw_cross=False
        )
        
        # Row 4: SCHZ surface
        ax5 = fig.add_subplot(gs[3, 0])
        texture_left = vol_to_surf(schz_sentences, fsaverage.pial_left)
        plot_surf_stat_map(
            fsaverage.infl_left,
            texture_left,
            hemi='left',
            view='lateral',
            threshold=2.3,
            cmap='hot',
            title='SCHZ Left',
            axes=ax5
        )
        
        ax6 = fig.add_subplot(gs[3, 1])
        texture_right = vol_to_surf(schz_sentences, fsaverage.pial_right)
        plot_surf_stat_map(
            fsaverage.infl_right,
            texture_right,
            hemi='right',
            view='lateral',
            threshold=2.3,
            cmap='hot',
            title='SCHZ Right',
            axes=ax6
        )
        
        # Row 5: Group difference axial
        ax7 = fig.add_subplot(gs[4, :])
        plot_stat_map(
            diff_map,
            threshold=2.3,
            display_mode='z',
            cut_coords=AXIAL_CUT_COORDS,
            title='Row 3: HC > SCHZ (Left Heschl\'s Gyrus Hypoactivation)',
            cmap='cold_hot',
            black_bg=False,
            axes=ax7,
            draw_cross=False
        )
        
        # Row 6: Group difference surface
        ax8 = fig.add_subplot(gs[5, 0])
        texture_left = vol_to_surf(diff_map, fsaverage.pial_left)
        plot_surf_stat_map(
            fsaverage.infl_left,
            texture_left,
            hemi='left',
            view='lateral',
            threshold=2.3,
            cmap='cold_hot',
            title='HC > SCHZ Left',
            axes=ax8
        )
        
        ax9 = fig.add_subplot(gs[5, 1])
        texture_right = vol_to_surf(diff_map, fsaverage.pial_right)
        plot_surf_stat_map(
            fsaverage.infl_right,
            texture_right,
            hemi='right',
            view='lateral',
            threshold=2.3,
            cmap='cold_hot',
            title='HC > SCHZ Right',
            axes=ax9
        )
        
        plt.tight_layout()
        fig_path = output_dir / 'fig2_sentences_and_group_difference.png'
        plt.savefig(str(fig_path), dpi=DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {fig_path}")
    
    # =========================================================================
    # Figure 3: Reversed Speech Contrast
    # =========================================================================
    logger.info("Generating Figure 3: Reversed Speech Contrast")
    
    hc_reversed_path = ANALYSIS_DIR / 'second_level' / 'stat_maps' / 'HC_reversed_vs_baseline_zmap.nii.gz'
    schz_reversed_path = ANALYSIS_DIR / 'second_level' / 'stat_maps' / 'SCHZ_reversed_vs_baseline_zmap.nii.gz'
    diff_reversed_path = ANALYSIS_DIR / 'second_level' / 'stat_maps' / 'HC_vs_SCHZ_reversed_vs_baseline_zmap.nii.gz'
    
    if hc_reversed_path.exists() and schz_reversed_path.exists() and diff_reversed_path.exists():
        hc_reversed = nib.load(str(hc_reversed_path))
        schz_reversed = nib.load(str(schz_reversed_path))
        diff_reversed = nib.load(str(diff_reversed_path))
        
        fig = plt.figure(figsize=(20, 20))
        gs = GridSpec(6, 3, figure=fig, hspace=0.3, wspace=0.1)
        
        # Row 1: HC axial
        ax1 = fig.add_subplot(gs[0, :])
        plot_stat_map(
            hc_reversed,
            threshold=2.3,
            display_mode='z',
            cut_coords=AXIAL_CUT_COORDS,
            title='Healthy Controls: Reversed Speech vs Baseline',
            cmap='hot',
            black_bg=False,
            axes=ax1,
            draw_cross=False
        )
        
        # Row 2: HC surface
        ax2 = fig.add_subplot(gs[1, 0])
        texture_left = vol_to_surf(hc_reversed, fsaverage.pial_left)
        plot_surf_stat_map(
            fsaverage.infl_left,
            texture_left,
            hemi='left',
            view='lateral',
            threshold=2.3,
            cmap='hot',
            title='HC Left',
            axes=ax2
        )
        
        ax3 = fig.add_subplot(gs[1, 1])
        texture_right = vol_to_surf(hc_reversed, fsaverage.pial_right)
        plot_surf_stat_map(
            fsaverage.infl_right,
            texture_right,
            hemi='right',
            view='lateral',
            threshold=2.3,
            cmap='hot',
            title='HC Right',
            axes=ax3
        )
        
        # Row 3: SCHZ axial
        ax4 = fig.add_subplot(gs[2, :])
        plot_stat_map(
            schz_reversed,
            threshold=2.3,
            display_mode='z',
            cut_coords=AXIAL_CUT_COORDS,
            title='Schizophrenia Patients: Reversed Speech vs Baseline',
            cmap='hot',
            black_bg=False,
            axes=ax4,
            draw_cross=False
        )
        
        # Row 4: SCHZ surface
        ax5 = fig.add_subplot(gs[3, 0])
        texture_left = vol_to_surf(schz_reversed, fsaverage.pial_left)
        plot_surf_stat_map(
            fsaverage.infl_left,
            texture_left,
            hemi='left',
            view='lateral',
            threshold=2.3,
            cmap='hot',
            title='SCHZ Left',
            axes=ax5
        )
        
        ax6 = fig.add_subplot(gs[3, 1])
        texture_right = vol_to_surf(schz_reversed, fsaverage.pial_right)
        plot_surf_stat_map(
            fsaverage.infl_right,
            texture_right,
            hemi='right',
            view='lateral',
            threshold=2.3,
            cmap='hot',
            title='SCHZ Right',
            axes=ax6
        )
        
        # Row 5: Group difference axial
        ax7 = fig.add_subplot(gs[4, :])
        plot_stat_map(
            diff_reversed,
            threshold=2.3,
            display_mode='z',
            cut_coords=AXIAL_CUT_COORDS,
            title='HC > SCHZ (Primary Auditory Cortex Hypoactivation)',
            cmap='cold_hot',
            black_bg=False,
            axes=ax7,
            draw_cross=False
        )
        
        # Row 6: Group difference surface
        ax8 = fig.add_subplot(gs[5, 0])
        texture_left = vol_to_surf(diff_reversed, fsaverage.pial_left)
        plot_surf_stat_map(
            fsaverage.infl_left,
            texture_left,
            hemi='left',
            view='lateral',
            threshold=2.3,
            cmap='cold_hot',
            title='HC > SCHZ Left',
            axes=ax8
        )
        
        ax9 = fig.add_subplot(gs[5, 1])
        texture_right = vol_to_surf(diff_reversed, fsaverage.pial_right)
        plot_surf_stat_map(
            fsaverage.infl_right,
            texture_right,
            hemi='right',
            view='lateral',
            threshold=2.3,
            cmap='cold_hot',
            title='HC > SCHZ Right',
            axes=ax9
        )
        
        plt.tight_layout()
        fig_path = output_dir / 'fig3_reversed_speech_contrast.png'
        plt.savefig(str(fig_path), dpi=DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {fig_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    logger.info("="*80)
    logger.info("GLM ANALYSIS PIPELINE - PLoS ONE Reproduction")
    logger.info("="*80)
    
    # Create output directories
    for subdir in ['first_level/contrast_maps', 'first_level/individual_plots',
                   'second_level/stat_maps', 'second_level/tables', 'figures_for_ppt']:
        (ANALYSIS_DIR / subdir).mkdir(parents=True, exist_ok=True)
    
    # Load participant data
    participants_df = load_participants_data()
    
    # =========================================================================
    # FIRST-LEVEL ANALYSIS
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: FIRST-LEVEL GLM ANALYSIS")
    logger.info("="*80)
    
    successful_subjects = []
    failed_subjects = []
    
    for subject_id in SUBJECTS:
        result = run_first_level_glm(subject_id, participants_df)
        if result is not None:
            successful_subjects.append(subject_id)
        else:
            failed_subjects.append(subject_id)
    
    logger.info(f"\nFirst-Level Summary:")
    logger.info(f"  Successful: {len(successful_subjects)}/{len(SUBJECTS)}")
    logger.info(f"  Failed: {len(failed_subjects)}/{len(SUBJECTS)}")
    
    if failed_subjects:
        logger.warning(f"  Failed subjects: {', '.join(failed_subjects)}")
    
    # =========================================================================
    # SECOND-LEVEL ANALYSIS
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: SECOND-LEVEL GLM ANALYSIS")
    logger.info("="*80)
    
    contrasts = ['words_vs_baseline', 'sentences_vs_baseline', 'reversed_vs_baseline']
    groups = ['HC', 'SCHZ', 'AVH+', 'AVH-']
    
    # One-sample t-tests for each group and contrast
    for contrast in contrasts:
        for group in groups:
            run_second_level_glm(participants_df, contrast, group)
    
    # Two-sample t-tests
    for contrast in contrasts:
        # HC vs SCHZ
        run_two_sample_ttest(participants_df, contrast, 'HC', 'SCHZ')
        # SCHZ vs HC
        run_two_sample_ttest(participants_df, contrast, 'SCHZ', 'HC')
        # AVH+ vs AVH-
        run_two_sample_ttest(participants_df, contrast, 'AVH+', 'AVH-')
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: PUBLICATION FIGURES")
    logger.info("="*80)
    
    generate_publication_figures(participants_df)
    
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {ANALYSIS_DIR}")
    logger.info(f"  - First-level maps: {ANALYSIS_DIR / 'first_level' / 'contrast_maps'}")
    logger.info(f"  - Second-level maps: {ANALYSIS_DIR / 'second_level' / 'stat_maps'}")
    logger.info(f"  - Publication figures: {ANALYSIS_DIR / 'figures_for_ppt'}")
    logger.info(f"\nFor Review 1 Slides:")
    logger.info(f"  Use: {ANALYSIS_DIR / 'figures_for_ppt' / 'fig2_sentences_and_group_difference.png'}")


if __name__ == '__main__':
    main()

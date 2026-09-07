#!/usr/bin/env python3
"""
Post-fMRIPrep Processing for Excluded Subjects
================================================

Process the 8 subjects that were excluded due to motion (FD > 0.3mm)
with a relaxed threshold (FD < 0.5mm).

Excluded subjects:
- sub-02 (FD=0.371mm)
- sub-22 (FD=0.372mm)
- sub-27 (FD=0.306mm)
- sub-28 (FD=0.340mm)
- sub-34 (FD=0.456mm)
- sub-45 (FD=0.438mm)
- sub-69 (FD=0.325mm)
- sub-77 (FD=0.323mm)

Author: fMRI Analysis Pipeline
Date: 2026-09-07
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import signal, stats
from scipy.ndimage import gaussian_filter, zoom
from joblib import Parallel, delayed

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Pipeline configuration"""
    
    PROJECT_ROOT = Path('/root/fMRI')
    DATA_ROOT = PROJECT_ROOT / 'output'
    OUTPUT_ROOT = PROJECT_ROOT / 'output' / 'postprocessed'
    LOG_DIR = PROJECT_ROOT / 'logs' / 'postprocessing'
    
    TR = 2.0
    SMOOTHING_FWHM = 5.0
    HIGH_PASS_FREQ = 0.008
    
    # Relaxed threshold for excluded subjects
    MEAN_FD_THRESHOLD = 0.5  # mm (increased from 0.3)
    
    SCHAEFER_ATLAS = 'Schaefer2018_200Parcels_7Networks'
    N_ROIS = 200
    
    CONFOUND_COLUMNS = [
        'trans_x', 'trans_y', 'trans_z',
        'rot_x', 'rot_y', 'rot_z'
    ]
    
    N_JOBS = 8  # Process all 8 subjects in parallel
    BACKEND = 'loky'
    
    # Subjects to process
    EXCLUDED_SUBJECTS = [
        'sub-02', 'sub-22', 'sub-27', 'sub-28',
        'sub-34', 'sub-45', 'sub-69', 'sub-77'
    ]
    
    @classmethod
    def setup_logging(cls, log_file: Path):
        """Setup logging"""
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def download_schaefer_atlas(logger):
    """Download Schaefer-200 atlas if needed"""
    atlas_dir = Config.PROJECT_ROOT / 'atlases'
    atlas_dir.mkdir(exist_ok=True)
    
    # Check for existing atlas (with _order suffix)
    atlas_file = atlas_dir / f"{Config.SCHAEFER_ATLAS}_order_FSLMNI152_2mm.nii.gz"
    
    if not atlas_file.exists():
        # Try without _order suffix
        atlas_file = atlas_dir / f"{Config.SCHAEFER_ATLAS}_FSLMNI152_2mm.nii.gz"
    
    if atlas_file.exists():
        logger.info(f"Atlas already exists: {atlas_file}")
        return atlas_file
    
    # Download if not found
    logger.info(f"Downloading {Config.SCHAEFER_ATLAS} atlas...")
    raise RuntimeError("Atlas not found - please ensure atlas file exists")



def load_bold_data(subject_id: str, logger) -> Tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
    """Load preprocessed BOLD data"""
    # Try different naming patterns
    possible_paths = [
        Config.DATA_ROOT / subject_id / 'func' / f'{subject_id}_task-speech_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz',
        Config.DATA_ROOT / subject_id / 'func' / f'{subject_id}_task-speech_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',
    ]
    
    bold_path = None
    for path in possible_paths:
        if path.exists():
            bold_path = path
            break
    
    if bold_path is None:
        raise FileNotFoundError(f"BOLD file not found for {subject_id}. Tried: {[str(p) for p in possible_paths]}")
    
    img = nib.load(str(bold_path))
    data = img.get_fdata()
    affine = img.affine
    
    logger.info(f"Loaded {subject_id}: shape={data.shape}")
    return img, data, affine


def load_confounds(subject_id: str) -> pd.DataFrame:
    """Load confound regressors"""
    confound_path = Config.DATA_ROOT / subject_id / 'func' / f'{subject_id}_task-speech_desc-confounds_timeseries.tsv'
    
    if not confound_path.exists():
        raise FileNotFoundError(f"Confound file not found: {confound_path}")
    
    return pd.read_csv(str(confound_path), sep='\t')


def calculate_fd(confounds: pd.DataFrame) -> np.ndarray:
    """Calculate framewise displacement"""
    motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    
    if not all(col in confounds.columns for col in motion_cols):
        raise ValueError("Missing motion columns in confounds")
    
    motion_data = confounds[motion_cols].values
    
    # Calculate FD
    fd = np.zeros(len(motion_data))
    for i in range(1, len(motion_data)):
        fd[i] = np.sum(np.abs(motion_data[i] - motion_data[i-1]))
    
    return fd


def apply_smoothing(data: np.ndarray, fwhm: float) -> np.ndarray:
    """Apply Gaussian smoothing"""
    sigma = fwhm / 2.3548  # Convert FWHM to sigma
    
    smoothed = np.zeros_like(data)
    for t in range(data.shape[3]):
        smoothed[:, :, :, t] = gaussian_filter(data[:, :, :, t], sigma)
    
    return smoothed


def apply_temporal_filter(data: np.ndarray, tr: float, high_pass: float) -> np.ndarray:
    """Apply high-pass temporal filter"""
    n_timepoints = data.shape[3]
    
    # Design filter
    nyquist = 1.0 / (2 * tr)
    normalized_cutoff = high_pass / nyquist
    
    b, a = signal.butter(4, normalized_cutoff, btype='high')
    
    # Apply filter to each voxel
    filtered = np.zeros_like(data)
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            for z in range(data.shape[2]):
                filtered[x, y, z, :] = signal.filtfilt(b, a, data[x, y, z, :])
    
    return filtered


def apply_confound_regression(data: np.ndarray, confounds: pd.DataFrame) -> np.ndarray:
    """Regress out motion confounds"""
    motion_data = confounds[Config.CONFOUND_COLUMNS].values
    
    # Add intercept
    X = np.column_stack([np.ones(len(motion_data)), motion_data])
    
    # Regress out confounds
    cleaned = np.zeros_like(data)
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            for z in range(data.shape[2]):
                y_data = data[x, y, z, :]
                beta = np.linalg.lstsq(X, y_data, rcond=None)[0]
                cleaned[x, y, z, :] = y_data - X @ beta
    
    return cleaned


def extract_roi_timeseries(bold_data: np.ndarray, atlas_file: Path, 
                           brain_mask: np.ndarray, bold_affine: np.ndarray = None) -> np.ndarray:
    """Extract ROI time series using Schaefer-200 atlas"""
    
    # Load atlas
    atlas_img = nib.load(str(atlas_file))
    atlas_data = atlas_img.get_fdata()
    
    # Check if dimensions match
    atlas_shape = atlas_data.shape
    bold_shape = bold_data.shape[:3]
    
    if atlas_shape != bold_shape:
        # Resample atlas to match BOLD dimensions
        zoom_factors = [bold_shape[i] / atlas_shape[i] for i in range(3)]
        atlas_data = zoom(atlas_data, zoom_factors, order=0)
    
    # Extract time series for each ROI
    n_timepoints = bold_data.shape[3]
    roi_timeseries = np.zeros((Config.N_ROIS, n_timepoints))
    
    for roi_id in range(1, Config.N_ROIS + 1):
        roi_mask = (atlas_data == roi_id)
        
        # Apply brain mask
        roi_mask = roi_mask & (brain_mask > 0)
        
        if np.sum(roi_mask) > 0:
            roi_timeseries[roi_id - 1, :] = np.mean(bold_data[roi_mask], axis=0)
    
    return roi_timeseries


def create_brain_mask(bold_data: np.ndarray) -> np.ndarray:
    """Create brain mask from BOLD data"""
    mean_bold = np.mean(bold_data, axis=3)
    threshold = np.percentile(mean_bold[mean_bold > 0], 10)
    mask = mean_bold > threshold
    return mask


def process_subject(subject_id: str, atlas_file: Path, logger) -> Dict:
    """Process a single subject"""
    
    logger.info(f"Processing {subject_id}")
    
    try:
        # Load data
        bold_img, bold_data, affine = load_bold_data(subject_id, logger)
        confounds = load_confounds(subject_id)
        
        # Calculate FD
        fd = calculate_fd(confounds)
        mean_fd = np.mean(fd)
        
        logger.info(f"{subject_id}: Mean FD = {mean_fd:.3f} mm")
        
        # Check threshold (relaxed to 0.5mm)
        if mean_fd > Config.MEAN_FD_THRESHOLD:
            logger.warning(f"{subject_id}: Mean FD {mean_fd:.3f} > {Config.MEAN_FD_THRESHOLD} mm - EXCLUDED")
            return {
                'subject_id': subject_id,
                'status': 'excluded',
                'mean_fd': float(mean_fd),
                'reason': f'Mean FD {mean_fd:.3f} > {Config.MEAN_FD_THRESHOLD} mm'
            }
        
        # Create brain mask
        brain_mask = create_brain_mask(bold_data)
        
        # Step 1: Smoothing
        logger.info(f"{subject_id}: Applying smoothing...")
        smoothed_data = apply_smoothing(bold_data, Config.SMOOTHING_FWHM)
        
        # Save smoothed
        smoothed_path = Config.OUTPUT_ROOT / 'smoothed' / f'{subject_id}_smoothed.nii.gz'
        smoothed_img = nib.Nifti1Image(smoothed_data, affine)
        nib.save(smoothed_img, str(smoothed_path))
        
        # Step 2: Temporal filtering
        logger.info(f"{subject_id}: Applying temporal filter...")
        filtered_data = apply_temporal_filter(smoothed_data, Config.TR, Config.HIGH_PASS_FREQ)
        
        # Save filtered
        filtered_path = Config.OUTPUT_ROOT / 'filtered' / f'{subject_id}_filtered.nii.gz'
        filtered_img = nib.Nifti1Image(filtered_data, affine)
        nib.save(filtered_img, str(filtered_path))
        
        # Step 3: Confound regression
        logger.info(f"{subject_id}: Applying confound regression...")
        cleaned_data = apply_confound_regression(filtered_data, confounds)
        
        # Save cleaned
        cleaned_path = Config.OUTPUT_ROOT / 'cleaned' / f'{subject_id}_cleaned.nii.gz'
        cleaned_img = nib.Nifti1Image(cleaned_data, affine)
        nib.save(cleaned_img, str(cleaned_path))
        
        # Step 4: ROI extraction
        logger.info(f"{subject_id}: Extracting ROI time series...")
        roi_timeseries = extract_roi_timeseries(cleaned_data, atlas_file, brain_mask, affine)
        
        # Save ROI timeseries
        roi_path = Config.OUTPUT_ROOT / 'roi_timeseries' / f'{subject_id}_roi_timeseries.npy'
        np.save(str(roi_path), roi_timeseries)
        
        # Step 5: Quality metrics
        quality_metrics = {
            'subject_id': subject_id,
            'mean_fd': float(mean_fd),
            'max_fd': float(np.max(fd)),
            'mean_dvars': float(np.mean(np.abs(np.diff(np.mean(bold_data, axis=(0,1,2)))))),
            'n_timepoints': int(bold_data.shape[3]),
            'processing_date': datetime.now().isoformat(),
            'status': 'completed'
        }
        
        # Save quality metrics
        metrics_path = Config.OUTPUT_ROOT / 'quality_metrics' / f'{subject_id}_quality_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(quality_metrics, f, indent=2)
        
        logger.info(f"{subject_id}: ✓ Completed successfully")
        
        return {
            'subject_id': subject_id,
            'status': 'completed',
            'mean_fd': float(mean_fd),
            'quality_metrics': quality_metrics
        }
        
    except Exception as e:
        logger.error(f"{subject_id}: ✗ Error - {str(e)}")
        return {
            'subject_id': subject_id,
            'status': 'error',
            'error': str(e)
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main processing pipeline"""
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Config.LOG_DIR / f'excluded_subjects_{timestamp}.log'
    logger = Config.setup_logging(log_file)
    
    logger.info("=" * 70)
    logger.info("POST-fMRIPrep PROCESSING - EXCLUDED SUBJECTS")
    logger.info("=" * 70)
    logger.info(f"Started at: {datetime.now()}")
    logger.info(f"Processing {len(Config.EXCLUDED_SUBJECTS)} excluded subjects")
    logger.info(f"Relaxed FD threshold: {Config.MEAN_FD_THRESHOLD} mm")
    logger.info("")
    
    # Download atlas
    atlas_file = download_schaefer_atlas(logger)
    
    # Create output directories
    for subdir in ['smoothed', 'filtered', 'cleaned', 'roi_timeseries', 'quality_metrics']:
        (Config.OUTPUT_ROOT / subdir).mkdir(parents=True, exist_ok=True)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"PROCESSING {len(Config.EXCLUDED_SUBJECTS)} SUBJECTS IN PARALLEL")
    logger.info("=" * 70)
    
    # Process subjects in parallel
    results = Parallel(n_jobs=Config.N_JOBS, backend=Config.BACKEND)(
        delayed(process_subject)(subject_id, atlas_file, logger)
        for subject_id in Config.EXCLUDED_SUBJECTS
    )
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 70)
    
    completed = [r for r in results if r['status'] == 'completed']
    excluded = [r for r in results if r['status'] == 'excluded']
    errors = [r for r in results if r['status'] == 'error']
    
    logger.info(f"Total subjects: {len(results)}")
    logger.info(f"Completed: {len(completed)}")
    logger.info(f"Excluded: {len(excluded)}")
    logger.info(f"Errors: {len(errors)}")
    
    if excluded:
        logger.info("")
        logger.info("Excluded subjects:")
        for r in excluded:
            logger.info(f"  - {r['subject_id']}: {r['reason']}")
    
    if errors:
        logger.info("")
        logger.info("Error subjects:")
        for r in errors:
            logger.info(f"  - {r['subject_id']}: {r['error']}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_subjects': len(results),
        'completed': [r['subject_id'] for r in completed],
        'excluded': [r['subject_id'] for r in excluded],
        'errors': [r['subject_id'] for r in errors],
        'results': results
    }
    
    summary_path = Config.OUTPUT_ROOT / 'excluded_subjects_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    
    print("\n✓ Processing completed!")
    print(f"✓ Output directory: {Config.OUTPUT_ROOT}")
    print(f"✓ Log file: {log_file}")


if __name__ == '__main__':
    main()

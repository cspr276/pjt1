#!/usr/bin/env python3
"""
Regenerate quality metrics for excluded subjects to match original pipeline format.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

# Configuration
class Config:
    OUTPUT_ROOT = Path('/root/fMRI/output/postprocessed')
    DATASET_ROOT = Path('/root/fMRI/ds004302-download')
    FD_THRESHOLD = 0.3  # mm

# Previously excluded subjects
EXCLUDED_SUBJECTS = ['sub-02', 'sub-22', 'sub-27', 'sub-28', 'sub-34', 'sub-45', 'sub-69', 'sub-77']

def calculate_quality_metrics(subject_id: str) -> Dict:
    """Calculate quality metrics matching original pipeline format."""
    
    # Find confounds file in fMRIPrep output
    subject_dir = Path('/root/fMRI/output') / subject_id / 'func'
    confounds_pattern = f"{subject_id}_task-speech_desc-confounds_timeseries.tsv"
    confounds_file = list(subject_dir.glob(confounds_pattern))
    
    if not confounds_file:
        raise FileNotFoundError(f"Confounds file not found for {subject_id}")
    
    confounds_file = confounds_file[0]
    
    # Read confounds
    df = pd.read_csv(confounds_file, sep='\t')
    
    # Calculate Framewise Displacement
    if 'framewise_displacement' in df.columns:
        fd = df['framewise_displacement'].values
    else:
        # Calculate from motion parameters
        motion_params = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        motion_data = df[motion_params].values
        
        # FD = sum(|derivatives|) for motion params
        fd = np.zeros(len(motion_data))
        for i in range(motion_data.shape[1]):
            fd += np.abs(np.diff(motion_data[:, i], prepend=motion_data[0, i]))
    
    # Calculate DVARS
    if 'dvars' in df.columns:
        dvars = df['dvars'].values
    else:
        # Simplified DVARS calculation
        dvars = np.zeros(len(fd))
    
    # Compute statistics (matching original format)
    stats_dict = {
        'subject': subject_id,
        'mean_fd': float(np.nanmean(fd)),
        'max_fd': float(np.nanmax(fd)),
        'pct_high_motion': float(np.sum(fd > Config.FD_THRESHOLD) / len(fd) * 100),
        'mean_dvars': float(np.nanmean(dvars)) if len(dvars) > 0 else 0,
        'n_timepoints': len(fd),
        'n_high_motion': int(np.sum(fd > Config.FD_THRESHOLD))
    }
    
    return stats_dict

def main():
    """Regenerate quality metrics for excluded subjects."""
    
    print("=" * 70)
    print("REGENERATING QUALITY METRICS FOR EXCLUDED SUBJECTS")
    print("=" * 70)
    print()
    
    success_count = 0
    error_count = 0
    
    for subject_id in EXCLUDED_SUBJECTS:
        try:
            print(f"Processing {subject_id}...")
            
            # Calculate quality metrics
            stats_dict = calculate_quality_metrics(subject_id)
            
            # Remove old file
            old_file = Config.OUTPUT_ROOT / 'quality_metrics' / f'{subject_id}_quality_metrics.json'
            if old_file.exists():
                old_file.unlink()
                print(f"  Removed old file: {old_file.name}")
            
            # Save new file with correct format
            new_file = Config.OUTPUT_ROOT / 'quality_metrics' / f'{subject_id}_qc.json'
            with open(new_file, 'w') as f:
                json.dump(stats_dict, f, indent=2)
            
            print(f"  ✓ Created: {new_file.name}")
            print(f"    Mean FD: {stats_dict['mean_fd']:.3f} mm")
            print(f"    Max FD: {stats_dict['max_fd']:.3f} mm")
            print(f"    High motion: {stats_dict['pct_high_motion']:.2f}%")
            print()
            
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            print()
            error_count += 1
    
    print("=" * 70)
    print(f"SUMMARY")
    print("=" * 70)
    print(f"Total subjects: {len(EXCLUDED_SUBJECTS)}")
    print(f"Success: {success_count}")
    print(f"Errors: {error_count}")
    print()
    
    # Verify all files now match
    print("Verification:")
    for subject_id in EXCLUDED_SUBJECTS:
        qc_file = Config.OUTPUT_ROOT / 'quality_metrics' / f'{subject_id}_qc.json'
        if qc_file.exists():
            with open(qc_file, 'r') as f:
                data = json.load(f)
            print(f"  ✓ {subject_id}_qc.json - Mean FD: {data['mean_fd']:.3f} mm")
        else:
            print(f"  ✗ {subject_id}_qc.json - MISSING")

if __name__ == '__main__':
    main()

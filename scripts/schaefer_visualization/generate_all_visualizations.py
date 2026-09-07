#!/usr/bin/env python3
"""
Generate Schaefer-200 ROI visualizations for all subjects.

This script creates comprehensive visualizations for each subject's ROI timeseries,
including:
- First 10 ROI timeseries plots
- Full 200x200 connectivity matrix
- Zoomed connectivity (first 20 ROIs)
- Correlation distribution

Output: PNG files saved to /root/fMRI/analysis/schaefer_visualizations/
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
from datetime import datetime

# Configuration
ROI_TIMESERIES_DIR = Path('/root/fMRI/output/postprocessed/roi_timeseries')
QC_DIR = Path('/root/fMRI/output/postprocessed/quality_metrics')
OUTPUT_DIR = Path('/root/fMRI/analysis/schaefer_visualizations')
LOG_FILE = Path('/root/fMRI/logs/schaefer_visualization.log')

# Network colors for Schaefer-200 (7 networks)
NETWORK_COLORS = {
    'Visual': '#1E88E5',
    'Somatomotor': '#43A047',
    'Dorsal Attention': '#FB8C00',
    'Ventral Attention': '#E53935',
    'Limbic': '#8E24AA',
    'Frontoparietal': '#00ACC1',
    'Default Mode': '#FFB300'
}

def setup_logging():
    """Setup logging to file."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Redirect stdout and stderr to log file
    log_stream = open(LOG_FILE, 'a')
    sys.stdout = log_stream
    sys.stderr = log_stream
    
    print(f"\n{'='*80}")
    print(f"Schaefer-200 Visualization Generation")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

def load_roi_timeseries(subject_id: str) -> np.ndarray:
    """Load ROI timeseries for a subject."""
    roi_file = ROI_TIMESERIES_DIR / f"{subject_id}_roi_timeseries.npy"
    
    if not roi_file.exists():
        raise FileNotFoundError(f"ROI timeseries not found: {roi_file}")
    
    return np.load(roi_file)

def load_quality_metrics(subject_id: str) -> dict:
    """Load quality metrics for a subject."""
    qc_file = QC_DIR / f"{subject_id}_qc.json"
    
    if qc_file.exists():
        with open(qc_file, 'r') as f:
            return json.load(f)
    return None

def create_visualization(subject_id: str, roi_data: np.ndarray, qc_metrics: dict):
    """Create comprehensive visualization for a subject."""
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Add title with subject info
    title = f'Schaefer-200 ROI Analysis - {subject_id}'
    if qc_metrics:
        title += f"\nMean FD: {qc_metrics['mean_fd']:.3f}mm | High Motion: {qc_metrics['pct_high_motion']:.1f}%"
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Create grid for subplots
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    # 1. Plot first 10 ROI timeseries
    ax1 = fig.add_subplot(gs[0, :])
    for i in range(10):
        ax1.plot(roi_data[i, :] + i*10, label=f'ROI {i+1}', linewidth=0.8)
    ax1.set_xlabel('Timepoints', fontsize=11)
    ax1.set_ylabel('BOLD Signal (offset for visibility)', fontsize=11)
    ax1.set_title('First 10 ROI Timeseries', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # 2. Full connectivity matrix (200x200)
    ax2 = fig.add_subplot(gs[1, 0])
    corr_matrix = np.corrcoef(roi_data)
    im2 = ax2.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax2.set_xlabel('ROI', fontsize=11)
    ax2.set_ylabel('ROI', fontsize=11)
    ax2.set_title('Full Connectivity Matrix (200×200)', fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Correlation', fraction=0.046)
    
    # 3. Zoomed connectivity (first 20 ROIs)
    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.imshow(corr_matrix[:20, :20], cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax3.set_xlabel('ROI', fontsize=11)
    ax3.set_ylabel('ROI', fontsize=11)
    ax3.set_title('Zoomed Connectivity (First 20 ROIs)', fontsize=13, fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='Correlation', fraction=0.046)
    
    # 4. Correlation distribution
    ax4 = fig.add_subplot(gs[2, 0])
    corr_values = corr_matrix[np.triu_indices(200, k=1)]
    ax4.hist(corr_values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax4.axvline(np.mean(corr_values), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(corr_values):.3f}')
    ax4.set_xlabel('Correlation Value', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Distribution of Correlations', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. Network-level average connectivity
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Schaefer-200 network boundaries (approximate)
    network_sizes = [29, 31, 21, 13, 9, 27, 70]  # Approximate sizes for 7 networks
    network_names = ['Vis', 'Som', 'DA', 'VA', 'Lim', 'FP', 'DM']
    
    # Calculate network-level averages
    network_means = []
    start_idx = 0
    for size in network_sizes:
        end_idx = start_idx + size
        network_corr = corr_matrix[start_idx:end_idx, start_idx:end_idx]
        network_means.append(np.mean(network_corr[np.triu_indices(size, k=1)]))
        start_idx = end_idx
    
    colors = list(NETWORK_COLORS.values())
    bars = ax5.bar(network_names, network_means, color=colors, edgecolor='black', alpha=0.8)
    ax5.set_xlabel('Network', fontsize=11)
    ax5.set_ylabel('Mean Within-Network Correlation', fontsize=11)
    ax5.set_title('Network-Level Connectivity', fontsize=13, fontweight='bold')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, network_means):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom' if val > 0 else 'top',
                fontsize=9, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig

def main():
    """Main function to generate visualizations for all subjects."""
    
    setup_logging()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Get list of all subjects
    roi_files = sorted(ROI_TIMESERIES_DIR.glob('*_roi_timeseries.npy'))
    subjects = [f.stem.replace('_roi_timeseries', '') for f in roi_files]
    
    print(f"Found {len(subjects)} subjects to process\n")
    
    # Process each subject
    successful = 0
    failed = 0
    
    for i, subject_id in enumerate(subjects, 1):
        print(f"\n[{i}/{len(subjects)}] Processing {subject_id}...")
        
        try:
            # Load data
            roi_data = load_roi_timeseries(subject_id)
            qc_metrics = load_quality_metrics(subject_id)
            
            print(f"  ROI data shape: {roi_data.shape}")
            
            # Create visualization
            fig = create_visualization(subject_id, roi_data, qc_metrics)
            
            # Save figure
            output_file = OUTPUT_DIR / f"{subject_id}_schaefer200_visualization.png"
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  ✓ Saved: {output_file.name}")
            successful += 1
            
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            failed += 1
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print(f"VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total subjects: {len(subjects)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()

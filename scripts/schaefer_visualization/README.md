# Schaefer-200 ROI Visualization Generator

This folder contains scripts for generating comprehensive visualizations of Schaefer-200 ROI timeseries for all subjects.

## Script: `generate_all_visualizations.py`

### Purpose
Generates publication-quality visualizations for each subject's Schaefer-200 ROI timeseries data.

### Visualizations Generated (per subject)
Each PNG file contains 5 subplots:

1. **First 10 ROI Timeseries** - Time series plots of the first 10 ROIs (offset for visibility)
2. **Full Connectivity Matrix (200×200)** - Complete functional connectivity matrix
3. **Zoomed Connectivity (First 20 ROIs)** - Detailed view of connectivity between first 20 ROIs
4. **Distribution of Correlations** - Histogram of all ROI-ROI correlations
5. **Network-Level Connectivity** - Bar plot of mean within-network connectivity for 7 networks

### Output Location
- **Directory**: `/root/fMRI/analysis/schaefer_visualizations/`
- **Files**: `{subject_id}_schaefer200_visualization.png`
- **Resolution**: 150 DPI
- **Size**: ~1.1-1.2 MB per image

### Network Information
The Schaefer-200 atlas divides the cortex into 7 functional networks:
- **Visual (Vis)**: 29 ROIs
- **Somatomotor (Som)**: 31 ROIs
- **Dorsal Attention (DA)**: 21 ROIs
- **Ventral Attention (VA)**: 13 ROIs
- **Limbic (Lim)**: 9 ROIs
- **Frontoparietal (FP)**: 27 ROIs
- **Default Mode (DM)**: 70 ROIs

### Running the Script

#### In a tmux session (recommended for batch processing):
```bash
tmux new-session -s schaefer_viz
cd /root/fMRI/scripts/schaefer_visualization
python3 generate_all_visualizations.py
# Detach with Ctrl+B, D
```

#### Monitor progress:
```bash
# Check log file
tail -f /root/fMRI/logs/schaefer_visualization.log

# Count generated files
ls -1 /root/fMRI/analysis/schaefer_visualizations/ | wc -l

# Check if script is running
ps aux | grep generate_all_visualizations.py
```

### Quality Metrics Integration
Each visualization includes quality metrics in the title:
- Mean Framewise Displacement (FD)
- Percentage of high motion volumes

### Processing Time
- **Per subject**: ~2-3 seconds
- **Total (71 subjects)**: ~3-4 minutes

### Requirements
- Python 3.8+
- NumPy
- Matplotlib
- Seaborn
- JSON (standard library)

### Log File
All output is logged to: `/root/fMRI/logs/schaefer_visualization.log`

## Status
- **Script Location**: `/root/fMRI/scripts/schaefer_visualization/generate_all_visualizations.py`
- **Output Directory**: `/root/fMRI/analysis/schaefer_visualizations/`
- **Total Subjects**: 71
- **Generated**: 51/71 (as of 2026-09-07 14:11)
- **Status**: Running in background (tmux session: schaefer_viz)

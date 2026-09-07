# Post-Processing Pipeline - Status Report

**Date**: 2026-09-07  
**Status**: ✅ RUNNING  
**Pipeline**: Schaefer-200 ROI Extraction with Full Preprocessing

---

## 🎯 What's Happening

The post-processing pipeline is now running in a **tmux session** that will persist across VPN disconnections. The pipeline is processing all 71 subjects through the following steps:

1. **Quality Control** - Extract FD/DVARS metrics
2. **Spatial Smoothing** - 5mm FWHM Gaussian smoothing
3. **Temporal Filtering** - High-pass filter > 0.008 Hz
4. **Confound Regression** - Remove motion artifacts
5. **ROI Extraction** - Schaefer-200 parcellation

---

## 📊 Current Progress

Check progress anytime with:
```bash
/root/fMRI/scripts/shell/monitor_postprocessing.sh
```

Or manually:
```bash
# Count processed subjects
ls /root/fMRI/output/postprocessed/roi_timeseries/*.npy | wc -l

# Check latest log
tail -f /root/fMRI/logs/postprocessing/postprocessing_*.log
```

---

## 🔧 How to Manage the TMUX Session

### ATTACH to view execution:
```bash
tmux attach -t postprocessing
```

### DETACH (keep running in background):
Press: `Ctrl+B` then `D`

### CHECK if session is running:
```bash
tmux ls
```

### KILL the session (if needed):
```bash
tmux kill-session -t postprocessing
```

---

## 📁 Output Structure

```
/root/fMRI/output/postprocessed/
├── smoothed/              # Spatially smoothed BOLD data
│   └── sub-XX_smoothed.nii.gz
├── filtered/              # Temporally filtered BOLD data
│   └── sub-XX_filtered.nii.gz
├── cleaned/               # Confound-regressed BOLD data
│   └── sub-XX_cleaned.nii.gz
├── roi_timeseries/        # Schaefer-200 ROI time series (VAE input)
│   └── sub-XX_roi_timeseries.npy  # Shape: (200, 341)
├── quality_metrics/       # QC metrics per subject
│   └── sub-XX_qc.json
└── processing_summary.json # Overall summary
```

---

## 📝 Log Files

All logs are in: `/root/fMRI/logs/postprocessing/`

- `postprocessing_YYYYMMDD_HHMMSS.log` - Detailed pipeline log
- `tmux_session_YYYYMMDD_HHMMSS.log` - Full tmux session log

---

## ⚠️ Important Notes

### Data Safety
- ✅ Original fMRIPrep data is **NOT modified**
- ✅ All outputs go to `/root/fMRI/output/postprocessed/`
- ✅ Backups created in `/root/fMRI/backups/` (if enabled)
- ✅ Processing summary saved after completion

### VPN Disconnection
- ✅ Pipeline continues running in tmux session
- ✅ Reconnect and attach to tmux to monitor
- ✅ All output logged to files

### Estimated Time
- Per subject: ~5-10 minutes (depending on system)
- Total: ~6-12 hours for 71 subjects
- Progress: Check with monitor script

---

## 🚀 Next Steps After Completion

### 1. Verify Completion
```bash
cat /root/fMRI/output/postprocessed/processing_summary.json
```

### 2. Check Data Quality
```bash
# Review QC metrics
ls /root/fMRI/output/postprocessed/quality_metrics/

# Check for excluded subjects
grep "excluded" /root/fMRI/output/postprocessed/processing_summary.json
```

### 3. Proceed to VAE Training
The ROI time series files (`*_roi_timeseries.npy`) are ready for:
- β-VAE training
- Graph construction
- BrainGNN input

Each file contains:
- Shape: `(200, 341)` - 200 ROIs, 341 timepoints
- Format: NumPy array
- Ready for deep learning

---

## 📚 Documentation

- **Pipeline Details**: `/root/fMRI/docs/POST_fMRIPrep_PIPELINE_ANALYSIS.md`
- **Decision Guide**: `/root/fMRI/docs/POSTPROCESSING_DECISION_GUIDE.md`
- **TMUX Instructions**: `/root/fMRI/TMUX_SESSION_INSTRUCTIONS.txt`

---

## 🆘 Troubleshooting

### Pipeline appears frozen
```bash
# Check if process is running
tmux attach -t postprocessing

# Check disk space
df -h /root/fMRI

# Check latest log
tail -f /root/fMRI/logs/postprocessing/postprocessing_*.log
```

### Need to restart
```bash
# Kill current session
tmux kill-session -t postprocessing

# Restart
cd /root/fMRI/scripts/shell
./run_postprocessing_tmux.sh
```

### Check specific subject
```bash
# View subject QC
cat /root/fMRI/output/postprocessed/quality_metrics/sub-01_qc.json

# Check ROI time series
python3 -c "import numpy as np; data = np.load('/root/fMRI/output/postprocessed/roi_timeseries/sub-01_roi_timeseries.npy'); print(f'Shape: {data.shape}')"
```

---

## ✅ Summary

- **Pipeline**: Running in tmux session `postprocessing`
- **Progress**: Monitor with `/root/fMRI/scripts/shell/monitor_postprocessing.sh`
- **Safety**: Original data preserved, outputs in separate directory
- **Persistence**: Survives VPN disconnections
- **Next**: ROI time series ready for VAE training

**For questions or issues**, check the log files or refer to the documentation.

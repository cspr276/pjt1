# Schaefer-200 ROI Timeseries Guide

## Overview

The Schaefer-200 atlas is a brain parcellation that divides the cerebral cortex into **200 distinct regions of interest (ROIs)** based on functional connectivity patterns. These ROIs are organized into **7 large-scale functional networks**.

---

## File Information

### Location
```
/root/fMRI/output/postprocessed/roi_timeseries/
```

### Files
- **Format**: NumPy arrays (`.npy`)
- **Naming**: `sub-XX_roi_timeseries.npy`
- **Total files**: 71 (one per subject)
- **File size**: ~533 KB each

---

## Data Structure

### Dimensions
```
Shape: (200, 341)
  - 200 ROIs (rows)
  - 341 timepoints (columns)
```

### Data Type
- **Type**: `float64`
- **Units**: BOLD signal intensity (arbitrary units)
- **Values**: Mean signal intensity for each ROI at each timepoint

---

## What the Values Mean

Each value in the matrix represents the **average BOLD (Blood Oxygen Level-Dependent) signal intensity** for a specific brain region at a specific moment in time.

**Example**:
```python
roi_data[0, 0]  # Signal intensity for ROI 1 at timepoint 1
roi_data[0, 1]  # Signal intensity for ROI 1 at timepoint 2
roi_data[199, 340]  # Signal intensity for ROI 200 at timepoint 341
```

---

## The 7 Functional Networks

The 200 ROIs are organized into 7 large-scale brain networks:

1. **Visual Network (Vis)**: ~30 ROIs
   - Processes visual information
   - Located in occipital lobe

2. **Somatomotor Network (SomMot)**: ~30 ROIs
   - Motor and somatosensory processing
   - Located in pre/post-central gyri

3. **Dorsal Attention Network (DorsAttn)**: ~20 ROIs
   - Top-down attentional control
   - Located in frontal and parietal regions

4. **Ventral Attention Network (Salience/VentAttn)**: ~20 ROIs
   - Bottom-up attention, salience detection
   - Located in insula and frontal operculum

5. **Limbic Network (Limbic)**: ~10 ROIs
   - Emotion and memory
   - Located in medial temporal and orbitofrontal regions

6. **Frontoparietal Network (Cont)**: ~30 ROIs
   - Executive control, working memory
   - Located in frontal and parietal cortex

7. **Default Mode Network (Default)**: ~60 ROIs
   - Self-referential thinking, mind-wandering
   - Located in medial prefrontal, posterior cingulate, and angular gyrus

---

## How to Load the Data

### Python
```python
import numpy as np

# Load ROI timeseries
roi_data = np.load('sub-01_roi_timeseries.npy')

# Check shape
print(roi_data.shape)  # (200, 341)

# Access specific ROI
roi_1 = roi_data[0, :]  # First ROI timeseries

# Access specific timepoint
time_1 = roi_data[:, 0]  # All ROIs at first timepoint
```

### MATLAB
```matlab
roi_data = load('sub-01_roi_timeseries.npy');
[size(roi_data)]  % [200, 341]
```

---

## Common Analyses

### 1. Functional Connectivity Matrix

Calculate correlations between all ROI pairs:

```python
import numpy as np

roi_data = np.load('sub-01_roi_timeseries.npy')
corr_matrix = np.corrcoef(roi_data)

# Shape: (200, 200)
# corr_matrix[i, j] = correlation between ROI i and ROI j
```

### 2. Extract Specific Network

```python
# Default Mode Network (approximately ROIs 140-200)
dmn_data = roi_data[140:200, :]
dmn_connectivity = np.corrcoef(dmn_data)
```

### 3. ROI-to-ROI Correlation

```python
# Correlation between ROI 1 and ROI 2
correlation = np.corrcoef(roi_data[0, :], roi_data[1, :])[0, 1]
```

### 4. Network-Level Analysis

```python
# Calculate mean connectivity within each network
networks = {
    'Visual': (0, 30),
    'Somatomotor': (30, 60),
    'Dorsal Attention': (60, 80),
    'Ventral Attention': (80, 100),
    'Limbic': (100, 110),
    'Frontoparietal': (110, 140),
    'Default Mode': (140, 200)
}

for network, (start, end) in networks.items():
    network_data = roi_data[start:end, :]
    mean_conn = np.mean(np.corrcoef(network_data))
    print(f"{network}: {mean_conn:.3f}")
```

---

## Visualization Examples

### Plot Single ROI Timeseries
```python
import matplotlib.pyplot as plt

roi_data = np.load('sub-01_roi_timeseries.npy')

plt.figure(figsize=(12, 4))
plt.plot(roi_data[0, :])
plt.xlabel('Timepoints')
plt.ylabel('BOLD Signal')
plt.title('ROI 1 Timeseries')
plt.grid(True, alpha=0.3)
plt.show()
```

### Plot Connectivity Matrix
```python
import seaborn as sns

corr_matrix = np.corrcoef(roi_data)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, square=True)
plt.title('Functional Connectivity Matrix')
plt.xlabel('ROI')
plt.ylabel('ROI')
plt.show()
```

---

## Quality Metrics

For each subject, you also have quality metrics:

```python
import json

with open('/root/fMRI/output/postprocessed/quality_metrics/sub-01_qc.json', 'r') as f:
    qc = json.load(f)

print(qc)
# {
#   "subject": "sub-01",
#   "mean_fd": 0.278,
#   "max_fd": 1.703,
#   "pct_high_motion": 4.99,
#   "mean_dvars": 43.02,
#   "n_timepoints": 341,
#   "n_high_motion": 17
# }
```

---

## Statistics (sub-01 Example)

- **Mean correlation**: 0.416
- **Std correlation**: 0.166
- **Min correlation**: -0.251
- **Max correlation**: 0.942

**Top 5 strongest connections**:
1. ROI 116 ↔ ROI 117: r = 0.942
2. ROI 5 ↔ ROI 108: r = 0.930
3. ROI 5 ↔ ROI 9: r = 0.919

---

## Next Steps

With these ROI timeseries, you can:

1. **Functional Connectivity Analysis**
   - Calculate connectivity matrices
   - Compare connectivity between subjects/groups

2. **Network Analysis**
   - Extract network-level metrics
   - Compare network properties

3. **Graph Theory Analysis**
   - Calculate graph metrics (clustering, path length, etc.)
   - Identify hub regions

4. **Statistical Comparisons**
   - Compare connectivity between conditions
   - Correlate with behavioral measures

---

## Summary

✅ **All 71 subjects** have Schaefer-200 ROI timeseries
✅ **200 ROIs** per subject
✅ **341 timepoints** per ROI
✅ **Ready for connectivity analysis**

The data is clean, processed, and ready for your next analysis steps!

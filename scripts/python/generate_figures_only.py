#!/usr/bin/env python3
"""
Generate publication figures from existing second-level z-maps.
"""

import logging
from pathlib import Path
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path('/root/fMRI')
SECOND_LEVEL_DIR = BASE_DIR / 'analysis' / 'second_level'
STAT_MAPS_DIR = SECOND_LEVEL_DIR / 'stat_maps'
FIGURES_DIR = SECOND_LEVEL_DIR / 'figures'

# Create output directory
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def generate_glass_brain_plots():
    """Generate glass brain plots for all z-maps."""
    logger.info("=" * 80)
    logger.info("GENERATING PUBLICATION FIGURES")
    logger.info("=" * 80)
    
    # Get all z-maps
    z_maps = sorted(STAT_MAPS_DIR.glob('*_zmap.nii.gz'))
    logger.info(f"Found {len(z_maps)} z-maps")
    
    for zmap_path in z_maps:
        try:
            logger.info(f"Processing: {zmap_path.name}")
            
            # Load the z-map
            z_map = nib.load(str(zmap_path))
            
            # Generate glass brain plot
            fig = plt.figure(figsize=(12, 6))
            plotting.plot_glass_brain(
                z_map,
                title=zmap_path.name.replace('_zmap.nii.gz', '').replace('_', ' '),
                threshold=2.3,
                display_mode='lyrz',
                plot_abs=False,
                figure=fig,
                cmap='cold_hot'
            )
            
            # Save figure
            output_path = FIGURES_DIR / f"{zmap_path.stem}_glass_brain.png"
            fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate figure for {zmap_path.name}: {e}")
    
    logger.info("=" * 80)
    logger.info("FIGURE GENERATION COMPLETE")
    logger.info("=" * 80)

if __name__ == '__main__':
    generate_glass_brain_plots()

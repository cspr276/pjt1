---
name: nilearn
description: 'Nilearn plotting patterns for first-level GLM statistical maps, including display-object save semantics and surface plotting workflows from the official Nilearn docs.'
---

# Nilearn plotting for first-level GLM images

Use the official Nilearn API patterns from:

- https://nilearn.github.io/stable/modules/generated/nilearn.plotting.plot_stat_map.html
- https://nilearn.github.io/stable/modules/generated/nilearn.plotting.plot_surf_stat_map.html

## Core rule

`plot_stat_map()` and `plot_surf_stat_map()` return Nilearn display objects, not Matplotlib `Figure` objects.

- `plot_stat_map(...)` returns an `OrthoSlicer`, `ZSlicer`, or similar display object.
- `plot_surf_stat_map(...)` returns a surface display object.
- These display objects are not valid inputs to `plt.close(fig)` or `fig.savefig(...)`.
- The official API is to pass `output_file=...` to save the image directly, and then call `display.close()` if the object is not `None`.

## Correct pattern for volume maps (First-Level GLM)

```python
from nilearn.plotting import plot_stat_map

output_path = "sub-01_words_vs_baseline_axial.png"
display = plot_stat_map(
    z_map,
    threshold=2.3,
    display_mode="z",
    cut_coords=[-12, -2, 8, 18, 28, 38, 48],
    title="sub-01: Words vs Baseline",
    cmap="hot",
    black_bg=False,
    draw_cross=False,
)

if display is not None and hasattr(display, "savefig"):
    display.savefig(str(output_path), dpi=300, bbox_inches="tight")

if display is not None and hasattr(display, "close"):
    display.close()
```

If you use `output_file=` instead, do not also pass `dpi=` directly to `plot_stat_map`; the `dpi` argument is not a recognized plotting param and gets forwarded to Matplotlib image rendering.

## Correct pattern for surface maps

```python
from nilearn.surface import vol_to_surf
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting import plot_surf_stat_map

fsaverage = fetch_surf_fsaverage(mesh="fsaverage5")
texture = vol_to_surf(z_map, fsaverage.pial_left)

output_path = "sub-01_words_vs_baseline_surface_left.png"
display = plot_surf_stat_map(
    fsaverage.infl_left,
    texture,
    hemi="left",
    view="lateral",
    threshold=2.3,
    cmap="hot",
    title="sub-01: Words vs Baseline (Left)",
    colorbar=False,
)

if display is not None and hasattr(display, "savefig"):
    display.savefig(str(output_path), dpi=300, bbox_inches="tight")

if display is not None and hasattr(display, "close"):
    display.close()
```

## Avoid

```python
fig = plot_stat_map(...)
fig.savefig("file.png")
plt.close(fig)  # wrong: fig is a Nilearn display object, not a Matplotlib Figure
```

```python
fig = plot_surf_stat_map(...)
fig.savefig("file.png")
plt.close(fig)  # wrong: same API mismatch
```

```python
plot_stat_map(..., dpi=300)  # wrong: dpi is not a recognized argument here
```

## Checklist for first-level GLM images

- Use existing first-level contrast maps instead of recomputing the GLM.
- Use `output_file=` for each generated image.
- Close the display object returned by Nilearn using `display.close()`.
- Keep the plotting threshold, colormap, and titles consistent with prior analysis.
- Preserve all non-visualization outputs and analysis logic.

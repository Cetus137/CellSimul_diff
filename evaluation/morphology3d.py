"""
3D cell morphology analysis from instance segmentation masks.

Computes per-cell morphological properties directly from provided instance
masks (no watershed segmentation — masks must already exist).

Expected input directory layout:
    {base}.tif          — raw fluorescence volume  (any dtype)
    {base}_masks.tif    — instance mask volume     (integer labels, background=0)

Pairs are discovered automatically by matching {base}.tif → {base}_masks.tif.

Per-cell properties computed (all in voxel units):
    volume              — number of voxels in the cell
    equivalent_diameter — diameter of sphere with the same volume
    sphericity          — isoperimetric quotient: π^(1/3)·(6V)^(2/3) / SA
                          (1.0 = perfect sphere; lower = elongated / irregular)
    elongation          — major_axis_length / minor_axis_length  (≥1.0)
    solidity            — volume / convex_hull_volume  (1.0 = fully convex)
    mean_intensity      — mean raw intensity inside the cell (requires image)
    nn_distance         — distance to nearest-neighbour cell centroid

Comparison between two sets uses Wasserstein-1 distance on each per-property
distribution, with optional plots saved to an output directory.

Public API
----------
load_volume_mask_pairs   — discover (image, mask) .tif pairs in a directory
compute_cell_morphology_3d — per-cell stats from a single mask (+ optional image)
compare_morphology_3d    — Wasserstein distances + plots across two sets
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance
from skimage.measure import regionprops, label as sk_label
from skimage.segmentation import find_boundaries

logger = logging.getLogger(__name__)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_volume_mask_pairs(
    data_dir: str,
    require_image: bool = True,
) -> List[Tuple[Optional[np.ndarray], np.ndarray]]:
    """
    Discover and load (image, mask) pairs from a directory.

    Pairs are matched by stem:
        {base}.tif       → image  (loaded as float32, original range preserved)
        {base}_masks.tif → mask   (loaded as-is; typically uint16 or uint32 label volume)

    Args:
        data_dir:      Directory containing .tif + _masks.tif files.
        require_image: If True, skip pairs where the raw image is missing.
                       If False, image is None for unmatched masks.

    Returns:
        List of (image_or_None, mask) tuples.
        image: (D, H, W) float32 — raw intensity in original range.
        mask:  (D, H, W) int-like — integer labels; 0 = background.
    """
    data_path = Path(data_dir)
    mask_files = sorted(data_path.glob("*_masks.tif"))

    if not mask_files:
        raise ValueError(f"No *_masks.tif files found in {data_dir}")

    pairs = []
    for mask_f in mask_files:
        # Derive image path: strip trailing _masks to get base, re-add .tif
        base_stem = mask_f.stem[: -len("_masks")]          # remove "_masks" suffix
        image_f   = mask_f.parent / (base_stem + ".tif")

        mask = tifffile.imread(str(mask_f))

        if image_f.exists():
            image = tifffile.imread(str(image_f)).astype(np.float32)
            # Handle possible extra channel/time dim: keep (D, H, W)
            while image.ndim > 3:
                image = image[0]
            pairs.append((image, mask))
        elif require_image:
            logger.warning("No matching image for %s — skipping", mask_f.name)
        else:
            pairs.append((None, mask))

    logger.info(
        "Loaded %d (image, mask) pairs from %s", len(pairs), data_dir
    )
    return pairs


def _iter_mask_pairs(
    data_dir: str,
    require_image: bool = True,
):
    """
    Generator yielding one (image_or_None, mask) pair at a time.

    Each volume is loaded on demand and released as soon as the caller moves
    to the next iteration — peak RAM is one volume, not all volumes at once.
    """
    data_path = Path(data_dir)
    mask_files = sorted(data_path.glob("*_masks.tif"))

    if not mask_files:
        raise ValueError(f"No *_masks.tif files found in {data_dir}")

    for mask_f in mask_files:
        base_stem = mask_f.stem[: -len("_masks")]
        image_f   = mask_f.parent / (base_stem + ".tif")

        mask = tifffile.imread(str(mask_f))

        if image_f.exists():
            image = tifffile.imread(str(image_f)).astype(np.float32)
            while image.ndim > 3:
                image = image[0]
            yield image, mask
        elif require_image:
            logger.warning("No matching image for %s — skipping", mask_f.name)
        else:
            yield None, mask


def _count_mask_files(data_dir: str) -> int:
    """Return number of *_masks.tif files in a directory."""
    return len(sorted(Path(data_dir).glob("*_masks.tif")))


# ─── Per-cell morphology ──────────────────────────────────────────────────────

def compute_cell_morphology_3d(
    mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    min_volume: int = 500,
    max_volume: int = 500_000,
    voxel_size_um: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute per-cell 3D morphological properties from an instance mask.

    Each cell is analysed in isolation; cells that touch the volume border or
    fail the volume filter are excluded.

    Args:
        mask:          (D, H, W) integer label volume; 0 = background.
        image:         (D, H, W) float32 raw image (required for mean_intensity).
        min_volume:    Reject cells with fewer voxels than this.
        max_volume:    Reject cells with more voxels than this.
        voxel_size_um: If given as (sz, sy, sx) in µm, volumetric properties are
                       converted to µm³ / µm.  None = report in voxel units.

    Returns:
        Dict mapping property name → 1-D float32 array (one value per accepted cell).
        Keys:
            volume              (voxels or µm³)
            equivalent_diameter (voxels or µm)
            sphericity          (dimensionless, 0–1)
            elongation          (dimensionless, ≥1)
            solidity            (dimensionless, 0–1)
            mean_intensity      (same range as image; NaN if image is None)
            nn_distance         (voxels or µm)
            n_cells             single-element array containing accepted cell count
    """
    volumes, eq_diameters, sphericities = [], [], []
    elongations, solidities, mean_intensities = [], [], []
    centroids = []

    # Relabel to ensure contiguous IDs (some masks may have gaps)
    # — but preserve 0 as background
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids != 0]

    for cell_id in unique_ids:
        binary = mask == cell_id   # (D, H, W) bool

        vol = int(binary.sum())
        if vol < min_volume or vol > max_volume:
            continue

        # ── regionprops on isolated cell ──────────────────────────────────────
        # Label the single-cell binary — regionprops expects a label image
        lbl = binary.astype(np.uint8)
        props = regionprops(lbl)
        if not props:
            continue
        p = props[0]

        # Volume
        if voxel_size_um is not None:
            sz, sy, sx = voxel_size_um
            voxel_vol = sz * sy * sx
            vol_eff = vol * voxel_vol
        else:
            vol_eff = float(vol)

        # Equivalent diameter of a sphere with the same volume
        eq_diam = float(2.0 * (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0))
        if voxel_size_um is not None:
            eq_diam *= float(np.cbrt(voxel_vol))

        # Sphericity via surface voxel count (isoperimetric quotient approximation)
        # Surface voxels: voxels adjacent to background in 6-connectivity
        surface_voxels = float(find_boundaries(binary, mode="inner", connectivity=1).sum())
        if surface_voxels > 0:
            sphericity = float(
                (np.pi ** (1.0 / 3.0)) * ((6.0 * vol) ** (2.0 / 3.0)) / surface_voxels
            )
            # Clamp to [0, 1] — approximation can slightly exceed 1 for small cells
            sphericity = min(sphericity, 1.0)
        else:
            sphericity = float("nan")

        # Elongation: major / minor axis of equivalent inertia ellipsoid
        try:
            # regionprops in 3D returns axis_major_length / axis_minor_length directly
            major = float(p.axis_major_length)
            minor = float(p.axis_minor_length)
            elongation = major / minor if minor > 1e-3 else float("nan")
        except AttributeError:
            # Fall back to inertia tensor eigenvalues for older skimage
            try:
                eigvals = p.inertia_tensor_eigvals
                l_max = float(max(eigvals))
                l_min = float(min(e for e in eigvals if e > 1e-6))
                elongation = float(np.sqrt(l_max / l_min)) if l_min > 1e-6 else float("nan")
            except Exception:
                elongation = float("nan")

        # Solidity
        try:
            solidity = float(p.solidity)
        except Exception:
            solidity = float("nan")

        # Mean intensity inside cell (intensity-weighted)
        if image is not None:
            mean_int = float(image[binary].mean())
        else:
            mean_int = float("nan")

        # Centroid for NN distance (stored as (z, y, x))
        centroids.append(list(p.centroid))

        volumes.append(vol_eff)
        eq_diameters.append(eq_diam)
        sphericities.append(sphericity)
        elongations.append(elongation)
        solidities.append(solidity)
        mean_intensities.append(mean_int)

    n = len(volumes)
    if n == 0:
        logger.warning("No accepted cells found in this volume")
        return {k: np.array([], dtype=np.float32) for k in
                ["volume", "equivalent_diameter", "sphericity", "elongation",
                 "solidity", "mean_intensity", "nn_distance"]}

    # ── Nearest-neighbour distances ───────────────────────────────────────────
    cent_arr = np.array(centroids, dtype=np.float32)   # (N, 3)
    if len(cent_arr) >= 2:
        tree = cKDTree(cent_arr)
        nn_dists, _ = tree.query(cent_arr, k=2)        # k=2: self + nearest
        nn_dists = nn_dists[:, 1].astype(np.float32)   # drop self (distance=0)
        if voxel_size_um is not None:
            sz, sy, sx = voxel_size_um
            # Scale NN distances by mean voxel size (anisotropy handled approximately)
            mean_vox = float(np.cbrt(sz * sy * sx))
            nn_dists = nn_dists * mean_vox
    else:
        nn_dists = np.full(n, float("nan"), dtype=np.float32)

    return {
        "volume":              np.array(volumes,        dtype=np.float32),
        "equivalent_diameter": np.array(eq_diameters,   dtype=np.float32),
        "sphericity":          np.array(sphericities,   dtype=np.float32),
        "elongation":          np.array(elongations,    dtype=np.float32),
        "solidity":            np.array(solidities,     dtype=np.float32),
        "mean_intensity":      np.array(mean_intensities, dtype=np.float32),
        "nn_distance":         nn_dists,
        "n_cells":             np.array([n], dtype=np.int32),
    }


# ─── Dataset-level aggregation ────────────────────────────────────────────────

def aggregate_morphology(
    stats_list: List[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """
    Pool per-cell arrays across multiple volumes into single flat arrays.

    Args:
        stats_list: Output of compute_cell_morphology_3d per volume.

    Returns:
        Dict with same keys, values pooled across all volumes (excluding n_cells).
    """
    keys = [k for k in stats_list[0].keys() if k != "n_cells"]
    pooled = {}
    for k in keys:
        arrays = [s[k] for s in stats_list if len(s[k]) > 0]
        pooled[k] = np.concatenate(arrays) if arrays else np.array([], dtype=np.float32)
    pooled["n_cells"] = np.array([s["n_cells"][0] for s in stats_list if len(s["n_cells"]) > 0],
                                  dtype=np.int32)
    return pooled


# ─── Comparison ───────────────────────────────────────────────────────────────

# Properties to compare and their axis labels for plots
_COMPARE_PROPS = {
    "volume":              "Volume (voxels)",
    "equivalent_diameter": "Equivalent diameter (voxels)",
    "sphericity":          "Sphericity",
    "elongation":          "Elongation (major/minor axis)",
    "solidity":            "Solidity",
    "nn_distance":         "Nearest-neighbour distance (voxels)",
}


def _plot_and_compare(
    real_agg: Dict[str, np.ndarray],
    syn_agg:  Dict[str, np.ndarray],
    out_dir: str,
) -> Dict[str, float]:
    """
    Shared comparison + plotting kernel used by both compare_morphology_3d
    (serial) and aggregate_from_npz (parallel array job).
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    real_ncells = int(real_agg["n_cells"].sum())
    syn_ncells  = int(syn_agg["n_cells"].sum())
    logger.info("Accepted cells — real: %d, synthetic: %d", real_ncells, syn_ncells)

    summary: Dict = {
        "n_cells_real":      real_ncells,
        "n_cells_synthetic": syn_ncells,
    }

    n_props = len(_COMPARE_PROPS)
    fig, axes = plt.subplots(
        2, (n_props + 1) // 2,
        figsize=(4 * ((n_props + 1) // 2), 7),
    )
    axes = axes.ravel()

    for ax_idx, (prop_key, prop_label) in enumerate(_COMPARE_PROPS.items()):
        r_vals = real_agg[prop_key]
        s_vals = syn_agg[prop_key]
        r_vals = r_vals[np.isfinite(r_vals)]
        s_vals = s_vals[np.isfinite(s_vals)]

        if len(r_vals) < 2 or len(s_vals) < 2:
            logger.warning("Insufficient valid values for '%s' — skipping W1", prop_key)
            w1 = float("nan")
        else:
            w1 = float(wasserstein_distance(r_vals, s_vals))

        summary[f"wasserstein_{prop_key}"] = round(w1, 6) if np.isfinite(w1) else None

        ax = axes[ax_idx]
        lo = min(r_vals.min() if len(r_vals) else 0, s_vals.min() if len(s_vals) else 0)
        hi = max(r_vals.max() if len(r_vals) else 1, s_vals.max() if len(s_vals) else 1)
        bins = np.linspace(lo, hi, 40)
        ax.hist(r_vals, bins=bins, color="#2196F3", alpha=0.6, density=True,
                label=f"real (n={len(r_vals)})", linewidth=0)
        ax.hist(s_vals, bins=bins, color="#FF5722", alpha=0.6, density=True,
                label=f"syn (n={len(s_vals)})", linewidth=0)
        ax.set_xlabel(prop_label, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(f"W1={w1:.3f}" if np.isfinite(w1) else "W1=n/a", fontsize=9)
        ax.legend(fontsize=7, framealpha=0.8)
        ax.tick_params(labelsize=7)

    for i in range(n_props, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("3D Cell Morphology: Real vs Synthetic", fontsize=11, y=1.01)
    plt.tight_layout()
    plot_path = str(Path(out_dir) / "morphology_3d.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved morphology comparison \u2192 %s", plot_path)

    logger.info("\u2500" * 50)
    logger.info("Morphology Wasserstein distances:")
    for k, v in summary.items():
        if k.startswith("wasserstein_") and v is not None:
            logger.info("  %-38s = %.4f", k, v)
    logger.info("\u2500" * 50)

    return summary


def compare_morphology_3d(
    real_dir: str,
    syn_dir: str,
    out_dir: str = ".",
    min_volume: int = 500,
    max_volume: int = 500_000,
    voxel_size_um: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, float]:
    """
    Compare morphological distributions between real and synthetic datasets.

    Streams files one at a time — peak RAM is a single (image, mask) pair,
    not all volumes simultaneously.

    Args:
        real_dir:  Directory with real {base}.tif + {base}_masks.tif pairs.
        syn_dir:   Directory with synthetic {base}.tif + {base}_masks.tif pairs.
        out_dir:   Directory for output plots (created if absent).
        min_volume, max_volume: Cell volume filter (voxels).
        voxel_size_um: Optional (sz, sy, sx) for unit conversion.

    Returns:
        Flat dict of Wasserstein-1 distances per property, plus cell count summary.
        Keys: wasserstein_{property} for each property in _COMPARE_PROPS.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    def _stream_stats(data_dir, require_image, label):
        n_files = _count_mask_files(data_dir)
        logger.info("Computing %s morphology (%d volumes) ...", label, n_files)
        # Print every 1% of volumes (at least every 1 file)
        log_every = max(1, n_files // 100)
        stats_list = []
        for i, (image, mask) in enumerate(
                _iter_mask_pairs(data_dir, require_image=require_image)):
            stats = compute_cell_morphology_3d(
                mask, image, min_volume, max_volume, voxel_size_um)
            stats_list.append(stats)
            del mask, image   # free volume immediately
            done = i + 1
            if done % log_every == 0 or done == n_files:
                pct = 100.0 * done / n_files
                n_accepted = sum(int(s["n_cells"][0]) for s in stats_list if len(s["n_cells"]) > 0)
                logger.info("  %s: %d / %d (%.0f%%)  —  %d cells accepted so far",
                            label, done, n_files, pct, n_accepted)
        return stats_list

    real_stats_list = _stream_stats(real_dir, require_image=True,  label="real")
    syn_stats_list  = _stream_stats(syn_dir,  require_image=False, label="synthetic")

    real_agg = aggregate_morphology(real_stats_list)
    syn_agg  = aggregate_morphology(syn_stats_list)

    return _plot_and_compare(real_agg, syn_agg, out_dir)


# ─── Convenience: evaluate a directory ────────────────────────────────────────

def evaluate_directory(
    real_dir: str,
    syn_dir: str,
    out_dir: str = ".",
    min_volume: int = 500,
    max_volume: int = 500_000,
    voxel_size_um: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, float]:
    """
    Load pairs from two directories and run the full morphology comparison.

    Args:
        real_dir: Directory with real {base}.tif + {base}_masks.tif pairs.
        syn_dir:  Directory with synthetic {base}.tif + {base}_masks.tif pairs.
        out_dir:  Directory for plots and summary.
        min_volume, max_volume: Voxel volume filter.
        voxel_size_um: Optional (sz, sy, sx) for µm conversion.

    Returns:
        Summary dict (same as compare_morphology_3d).
    """
    return compare_morphology_3d(
        real_dir=real_dir,
        syn_dir=syn_dir,
        out_dir=out_dir,
        min_volume=min_volume,
        max_volume=max_volume,
        voxel_size_um=voxel_size_um,
    )


# ─── Array-job helpers ───────────────────────────────────────────────────────

def make_manifest(
    real_dir: str,
    syn_dir: str,
    manifest_path: str,
) -> int:
    """
    Write a tab-separated manifest of all (mask_file, image_file, label) pairs
    from real_dir and syn_dir.  Each line:

        mask_path <TAB> image_path <TAB> label

    image_path is an empty string when no matching .tif exists.
    Returns the total number of lines written.
    """
    import csv

    entries = []
    for data_dir, lbl in [(real_dir, "real"), (syn_dir, "syn")]:
        for mask_f in sorted(Path(data_dir).glob("*_masks.tif")):
            base    = mask_f.stem[: -len("_masks")]
            image_f = mask_f.parent / (base + ".tif")
            entries.append((
                str(mask_f),
                str(image_f) if image_f.exists() else "",
                lbl,
            ))

    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        csv.writer(f, delimiter="\t").writerows(entries)

    logger.info("Manifest written \u2192 %s  (%d entries)", manifest_path, len(entries))
    return len(entries)


def process_single_file(
    mask_file: str,
    out_file: str,
    image_file: Optional[str] = None,
    label: str = "real",
    min_volume: int = 500,
    max_volume: int = 500_000,
    voxel_size_um: Optional[Tuple[float, float, float]] = None,
    z_start: Optional[int] = None,
    z_end: Optional[int] = None,
    tile_size: Optional[int] = None,
) -> None:
    """
    Process one (mask, optional image) pair and save per-cell stats to a .npz.

    Called by each SLURM array-job worker.  The .npz contains all keys from
    compute_cell_morphology_3d plus a ``_label`` string array.

    Args:
        mask_file:  Path to instance mask .tif.
        out_file:   Destination .npz path (directories created automatically).
        image_file: Path to matching raw image .tif; may be empty / None.
        label:      ``"real"`` or ``"syn"``.
        z_start:    If given (with z_end), crop mask[z_start:z_end] before
                    processing.  When None the full z range is used.
        z_end:      Upper bound of z crop (exclusive).  See z_start.
        tile_size:  If given, tile the volume into tile_size×tile_size XY blocks
                    and evaluate each independently, pooling results.  When None
                    the entire volume is processed as one unit.
    """
    mask  = tifffile.imread(mask_file)
    image = None
    if image_file and Path(image_file).exists():
        image = tifffile.imread(image_file).astype(np.float32)
        while image.ndim > 3:
            image = image[0]

    # ── Optional z-slab crop ──────────────────────────────────────────────────
    if z_start is not None and z_end is not None:
        mask = mask[z_start:z_end]
        if image is not None:
            image = image[z_start:z_end]
        logger.info("Z-crop applied: z=%d:%d  → shape %s", z_start, z_end, mask.shape)

    # ── Tile or process whole volume ──────────────────────────────────────────
    if tile_size is not None:
        _D, H, W = mask.shape
        n_y = max(1, H // tile_size)
        n_x = max(1, W // tile_size)
        tile_stats_list = []
        for iy in range(n_y):
            for ix in range(n_x):
                y0, y1 = iy * tile_size, (iy + 1) * tile_size
                x0, x1 = ix * tile_size, (ix + 1) * tile_size
                t_mask  = mask[:, y0:y1, x0:x1]
                t_image = image[:, y0:y1, x0:x1] if image is not None else None
                tile_stats_list.append(
                    compute_cell_morphology_3d(
                        t_mask, t_image, min_volume, max_volume, voxel_size_um
                    )
                )
        logger.info("Tiled %d\u00d7%d (tile_size=%d)", n_y, n_x, tile_size)
        stats = aggregate_morphology(tile_stats_list)
    else:
        stats = compute_cell_morphology_3d(
            mask, image, min_volume, max_volume, voxel_size_um,
        )
    del mask, image

    n_acc = int(stats["n_cells"].sum()) if len(stats["n_cells"]) > 0 else 0
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_file, _label=np.array([label]), **stats)
    logger.info("Saved %s \u2192 %s  (%d cells)", label, out_file, n_acc)


def aggregate_from_npz(
    stats_dir: str,
    out_dir: str,
) -> Dict[str, float]:
    """
    Aggregate all .npz worker outputs from stats_dir, then compare.

    Expects each .npz to contain a ``_label`` key (``"real"`` or ``"syn"``)
    plus all morphology property arrays as produced by process_single_file.

    Args:
        stats_dir: Directory of .npz files written by the array workers.
        out_dir:   Destination for morphology_3d.png + morphology_summary.yaml.

    Returns:
        Summary dict of Wasserstein distances and cell counts.
    """
    npz_files = sorted(Path(stats_dir).glob("*.npz"))
    if not npz_files:
        raise ValueError(f"No .npz files found in {stats_dir}")

    real_stats_list: List[Dict[str, np.ndarray]] = []
    syn_stats_list:  List[Dict[str, np.ndarray]] = []

    for npz_f in npz_files:
        data  = np.load(npz_f, allow_pickle=False)
        lbl   = str(data["_label"][0])
        stats = {k: data[k] for k in data.files if k != "_label"}
        (real_stats_list if lbl == "real" else syn_stats_list).append(stats)

    logger.info(
        "Aggregating: %d real + %d synthetic stat files",
        len(real_stats_list), len(syn_stats_list),
    )

    real_agg = aggregate_morphology(real_stats_list)
    syn_agg  = aggregate_morphology(syn_stats_list)
    return _plot_and_compare(real_agg, syn_agg, out_dir)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "3D cell morphology — array-job worker.\n"
            "Processes one mask file (--index = $SLURM_ARRAY_TASK_ID), saves a\n"
            "per-file .npz to --stats_dir, then automatically aggregates and\n"
            "plots once all expected .npz files are present."
        )
    )
    parser.add_argument("--real_dir",    required=True,
                        help="Directory of real *_masks.tif files")
    parser.add_argument("--syn_dir",     required=True,
                        help="Directory of synthetic *_masks.tif files")
    parser.add_argument("--index",       type=int, required=True,
                        help="0-based file index (= $SLURM_ARRAY_TASK_ID). "
                             "Real files are listed first, then synthetic.")
    parser.add_argument("--stats_dir",   required=True,
                        help="Directory for per-file .npz outputs, final plot, and summary YAML")
    parser.add_argument("--min_volume",  type=int, default=500,
                        help="Minimum cell volume in voxels (default 500)")
    parser.add_argument("--max_volume",  type=int, default=500_000,
                        help="Maximum cell volume in voxels (default 500000)")
    parser.add_argument("--voxel_size",  type=float, nargs=3, default=None,
                        metavar=("SZ", "SY", "SX"),
                        help="Voxel size in µm; omit for voxel units")
    parser.add_argument("--z_start",     type=int, default=None,
                        help="Start of z-slab crop for real volumes (inclusive, "
                             "0-based). Omit to use the full z range.")
    parser.add_argument("--z_end",       type=int, default=None,
                        help="End of z-slab crop for real volumes (exclusive). "
                             "Omit to use the full z range.")
    parser.add_argument("--tile_size",   type=int, default=None,
                        help="XY tile size. Splits the volume into "
                             "tile_size×tile_size blocks evaluated independently. "
                             "Omit to process the whole volume as one unit.")
    args = parser.parse_args()

    # ── Build full sorted file list (real first, then syn) ────────────────────
    real_masks  = sorted(Path(args.real_dir).glob("*_masks.tif"))
    syn_masks   = sorted(Path(args.syn_dir).glob("*_masks.tif"))
    all_entries = [(mf, "real") for mf in real_masks] + \
                  [(mf, "syn")  for mf in syn_masks]
    n_total = len(all_entries)

    if args.index >= n_total:
        logger.info("Index %d out of range (%d entries) — nothing to do.",
                    args.index, n_total)
        raise SystemExit(0)

    # ── Process this task's file ───────────────────────────────────────────────
    mask_f, label = all_entries[args.index]
    base      = mask_f.stem[: -len("_masks")]
    image_f   = mask_f.parent / (base + ".tif")
    vox       = tuple(args.voxel_size) if args.voxel_size else None
    os.makedirs(args.stats_dir, exist_ok=True)
    out_file  = Path(args.stats_dir) / (mask_f.stem + ".npz")

    process_single_file(
        mask_file     = str(mask_f),
        out_file      = str(out_file),
        image_file    = str(image_f) if image_f.exists() else None,
        label         = label,
        min_volume    = args.min_volume,
        max_volume    = args.max_volume,
        voxel_size_um = vox,
        # Z-crop only for real volumes — synthetic is already the correct z depth
        z_start       = args.z_start if (label == "real") else None,
        z_end         = args.z_end   if (label == "real") else None,
        tile_size     = args.tile_size,
    )

    # ── Auto-aggregate when all files are done ────────────────────────────────
    n_done = len(list(Path(args.stats_dir).glob("*.npz")))
    logger.info("Stats dir: %d / %d .npz files present.", n_done, n_total)

    if n_done >= n_total:
        logger.info("All files processed — running aggregation ...")
        summary = aggregate_from_npz(args.stats_dir, args.stats_dir)

        out_path = Path(args.stats_dir) / "morphology_summary.yaml"

        def _clean(v):
            if isinstance(v, (np.floating, np.integer)):
                return v.item()
            return v

        with open(out_path, "w") as f:
            yaml.dump({k: _clean(v) for k, v in summary.items()},
                      f, default_flow_style=False, sort_keys=False)
        logger.info("Summary saved → %s", out_path)
        logger.info("Plot saved    → %s", Path(args.stats_dir) / "morphology_3d.png")
    else:
        logger.info("Aggregation deferred — waiting for remaining tasks.")


#!/usr/bin/env python
# coding: utf-8
"""
Brightness Checker Time Series: Track brightness changes over time in z-stacks.

This script:
1. Loads frames from ND2 file or folder of TIFF frames
2. Groups frames into z-stacks (e.g., 80 frames per stack)
3. Computes MIP for each z-stack
4. Calculates global background (median across all MIPs)
5. Subtracts background from each MIP
6. Computes top 5% brightest pixels for each MIP
7. Generates time-series plots of brightness vs time
8. Saves results as PNG plots and CSV
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
import nd2
from tqdm.auto import tqdm


# ============================================================================
# FRAME LOADING
# ============================================================================

def load_frames_from_nd2(nd2_path: Path, channel: int = 0) -> np.ndarray:
    """
    Load frames from ND2 file.
    
    Returns:
        Array of shape (num_frames, height, width)
    """
    print(f"Loading from ND2: {nd2_path.name}")
    with nd2.ND2File(nd2_path) as f:
        data = np.asarray(f)
        # Assuming data is (frames, channels, height, width)
        frames = data[:, channel, :, :]
    
    print(f"✓ Loaded {frames.shape[0]} frames\n")
    return frames


def load_frames_from_tiff_folder(folder: Path) -> np.ndarray:
    """
    Load numbered TIFF frames from folder.
    
    Expects files named: frame_0.tif, frame_1.tif, etc.
    
    Returns:
        Array of shape (num_frames, height, width)
    """
    print(f"Loading from TIFF folder: {folder}")
    
    # Try different naming conventions
    tiff_files = sorted(folder.glob("frame_*.tif"), key=lambda x: int(x.stem.split("_")[1]))
    
    if not tiff_files:
        # Try just *.tif with numeric sort
        tiff_files = sorted(
            folder.glob("*.tif"),
            key=lambda x: int(''.join(filter(str.isdigit, x.stem)) or 0)
        )
    
    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files found in {folder}")
    
    print(f"Found {len(tiff_files)} TIFF files")
    
    # Load first frame to get dimensions
    first_frame = tifffile.imread(tiff_files[0])
    frames = np.zeros((len(tiff_files), *first_frame.shape), dtype=first_frame.dtype)
    
    for i, tiff_path in enumerate(tqdm(tiff_files, desc="  Loading frames")):
        frames[i] = tifffile.imread(tiff_path)
    
    print(f"✓ Loaded {frames.shape[0]} frames\n")
    return frames


# ============================================================================
# Z-STACK PROCESSING
# ============================================================================

def group_frames_into_zstacks(
    frames: np.ndarray,
    frames_per_stack: int = 80
) -> np.ndarray:
    """
    Group frames into z-stacks.
    
    Args:
        frames: Array of shape (num_frames, height, width)
        frames_per_stack: Number of frames per z-stack (default 80)
    
    Returns:
        Array of z-stacks with shape (num_stacks, frames_per_stack, height, width)
    """
    num_frames = frames.shape[0]
    num_stacks = num_frames // frames_per_stack
    
    # Trim frames to fit evenly into stacks
    trimmed_frames = frames[:num_stacks * frames_per_stack]
    
    zstacks = trimmed_frames.reshape(num_stacks, frames_per_stack, *frames.shape[1:])
    
    print(f"Grouped {num_frames} frames into {num_stacks} z-stacks")
    print(f"  ({frames_per_stack} frames per stack)")
    
    if num_frames % frames_per_stack != 0:
        discarded = num_frames % frames_per_stack
        print(f"  ({discarded} frames discarded)\n")
    else:
        print()
    
    return zstacks


def compute_mips(zstacks: np.ndarray) -> np.ndarray:
    """
    Compute maximum intensity projection for each z-stack.
    
    Args:
        zstacks: Array of shape (num_stacks, frames_per_stack, height, width)
    
    Returns:
        Array of MIPs with shape (num_stacks, height, width)
    """
    print("Computing MIPs...")
    mips = np.max(zstacks, axis=1)  # Max along z-axis
    print(f"✓ Computed {len(mips)} MIPs\n")
    return mips


# ============================================================================
# BACKGROUND SUBTRACTION
# ============================================================================

def compute_global_background(mips: np.ndarray) -> float:
    """
    Compute global background as median pixel value across all MIPs.
    
    Returns:
        Background value (float)
    """
    bg_val = np.median(mips)
    print(f"Computing global background...")
    print(f"  Median pixel value across all MIPs: {bg_val:.2f}\n")
    return bg_val


def subtract_background(mips: np.ndarray, bg_val: float) -> np.ndarray:
    """
    Subtract background from all MIPs.
    
    Args:
        mips: MIP array
        bg_val: Background value to subtract
    
    Returns:
        Background-subtracted MIPs
    """
    print("Subtracting background from MIPs...")
    mips_bgsub = mips.astype(np.float32) - bg_val
    mips_bgsub[mips_bgsub < 0] = 0  # Clip negatives
    print(f"✓ Background subtracted\n")
    return mips_bgsub


# ============================================================================
# BRIGHTNESS METRICS
# ============================================================================

def compute_top_percent_brightness(
    mips: np.ndarray,
    percentile: int = 95
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and median brightness of top percentile pixels for each MIP.
    
    Args:
        mips: Array of MIP images
        percentile: Percentile threshold (default 95 = top 5%)
    
    Returns:
        (mean_brightness, median_brightness) arrays of length num_mips
    """
    print(f"Computing top {100 - percentile}% brightness metrics...")
    
    mean_brightness = np.zeros(len(mips))
    median_brightness = np.zeros(len(mips))
    
    for i, mip in enumerate(tqdm(mips, desc="  Processing")):
        vals = mip.flatten()
        cut = np.percentile(vals, percentile)
        top_vals = vals[vals >= cut]
        
        mean_brightness[i] = np.mean(top_vals)
        median_brightness[i] = np.median(top_vals)
    
    print(f"✓ Brightness metrics computed\n")
    
    return mean_brightness, median_brightness


# ============================================================================
# SAVING RESULTS
# ============================================================================

def save_mips(mips: np.ndarray, output_folder: Path) -> None:
    """Save background-subtracted MIPs as individual TIFF files."""
    mip_folder = output_folder / "mips"
    mip_folder.mkdir(exist_ok=True, parents=True)
    
    print("Saving background-subtracted MIPs...")
    for i, mip in enumerate(tqdm(mips, desc="  Saving")):
        tiff_path = mip_folder / f"mip_{i:04d}.tif"
        tifffile.imwrite(tiff_path, mip.astype(np.float32))
    
    print(f"✓ Saved {len(mips)} MIPs to {mip_folder}\n")


def save_results_csv(
    time_indices: np.ndarray,
    mean_brightness: np.ndarray,
    median_brightness: np.ndarray,
    output_folder: Path
) -> None:
    """
    Save results as CSV with 3 columns: x-values, y-mean, y-median.
    """
    df_results = pd.DataFrame({
        "z_stack_index": time_indices.astype(int),
        "brightness_mean": mean_brightness,
        "brightness_median": median_brightness,
    })
    
    csv_path = output_folder / "brightness_timeseries.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"✓ Saved: brightness_timeseries.csv\n")


# ============================================================================
# PLOTTING
# ============================================================================

def plot_brightness_time_series(
    time_indices: np.ndarray,
    mean_brightness: np.ndarray,
    median_brightness: np.ndarray,
    output_folder: Path
) -> None:
    """Create two plots: mean and median brightness over time."""
    
    # --- Plot 1: Mean brightness ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        time_indices, mean_brightness,
        marker='o', linewidth=2, markersize=8,
        color='#1f77b4', label='Mean (top 5%)'
    )
    ax.set_xlabel("Z-Stack Index (Time)", fontsize=12)
    ax.set_ylabel("Brightness (Mean)", fontsize=12)
    ax.set_title("Brightness Over Time (Mean of Top 5% Pixels)", fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    
    out_path = output_folder / "brightness_mean_timeseries.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: brightness_mean_timeseries.png")
    plt.close()
    
    # --- Plot 2: Median brightness ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        time_indices, median_brightness,
        marker='s', linewidth=2, markersize=8,
        color='#ff7f0e', label='Median (top 5%)'
    )
    ax.set_xlabel("Z-Stack Index (Time)", fontsize=12)
    ax.set_ylabel("Brightness (Median)", fontsize=12)
    ax.set_title("Brightness Over Time (Median of Top 5% Pixels)", fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    
    out_path = output_folder / "brightness_median_timeseries.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: brightness_median_timeseries.png\n")
    plt.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(
    input_path: Path,
    output_folder: Path = None,
    frames_per_stack: int = 80,
    channel: int = 0
) -> None:
    """
    Main brightness time-series analysis pipeline.
    
    Args:
        input_path: ND2 file or folder with TIFF frames
        output_folder: Where to save results (auto-generated if None)
        frames_per_stack: Frames per z-stack (default 80)
        channel: ND2 channel to use (default 0)
    """
    
    # Auto-generate output folder if not specified
    if output_folder is None:
        if input_path.is_file():
            output_folder = input_path.parent / f"{input_path.stem}_timeseries_analysis"
        else:
            output_folder = input_path / "timeseries_analysis"
    
    output_folder.mkdir(exist_ok=True, parents=True)
    
    print("=" * 70)
    print("BRIGHTNESS CHECKER TIME-SERIES ANALYSIS")
    print("=" * 70 + "\n")
    
    # --- Load frames ---
    print(f"Input: {input_path}\n")
    
    if input_path.is_file() and input_path.suffix.lower() == ".nd2":
        frames = load_frames_from_nd2(input_path, channel=channel)
    else:
        frames = load_frames_from_tiff_folder(input_path)
    
    print(f"Frame shape: {frames.shape}\n")
    
    # --- Group into z-stacks ---
    zstacks = group_frames_into_zstacks(frames, frames_per_stack=frames_per_stack)
    
    # --- Compute MIPs ---
    mips = compute_mips(zstacks)
    
    # --- Compute background ---
    bg_val = compute_global_background(mips)
    
    # --- Subtract background ---
    mips_bgsub = subtract_background(mips, bg_val)
    
    # --- Compute brightness metrics ---
    mean_brightness, median_brightness = compute_top_percent_brightness(mips_bgsub, percentile=95)
    
    # --- Save MIPs ---
    save_mips(mips_bgsub, output_folder)
    
    # --- Generate plots ---
    time_indices = np.arange(len(mips))
    print("Generating plots...")
    plot_brightness_time_series(time_indices, mean_brightness, median_brightness, output_folder)
    
    # --- Save CSV ---
    save_results_csv(time_indices, mean_brightness, median_brightness, output_folder)
    
    print("=" * 70)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_folder}\n")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze brightness changes over time in z-stack time-series.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From ND2 file (all frames)
  uv run brightness_checker_time.py /path/to/timeseries.nd2
  
  # From TIFF folder
  uv run brightness_checker_time.py /path/to/tiff_frames/
  
  # With custom output folder
  uv run brightness_checker_time.py /path/to/timeseries.nd2 --output ./results
  
  # With custom frames per stack
  uv run brightness_checker_time.py /path/to/timeseries.nd2 --frames-per-stack 100
        """
    )
    
    parser.add_argument(
        "input",
        type=Path,
        help="ND2 file or folder with TIFF frames"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output folder (default: auto-generated)"
    )
    
    parser.add_argument(
        "--frames-per-stack",
        type=int,
        default=80,
        help="Frames per z-stack (default: 80)"
    )
    
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="ND2 channel to use (default: 0)"
    )
    
    args = parser.parse_args()
    
    main(
        args.input,
        output_folder=args.output,
        frames_per_stack=args.frames_per_stack,
        channel=args.channel
    )

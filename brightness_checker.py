#!/usr/bin/env python
# coding: utf-8
"""
Brightness Checker: Analyze fluorescence intensity in immobilized sensor animals.

This script:
1. Converts ND2 microscopy files to maximum intensity projections (MIPs)
2. Performs background subtraction
3. Parses metadata from filenames
4. Computes brightness statistics (mean, median, top 5% brightest pixels)
5. Generates comparison plots vs control strain
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile
import nd2
import re
from tqdm.auto import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

CONTROL_STRAIN = "SWF1088"
CONTROL_POWER = 13
CHANNEL_OF_INTEREST = 0
BRIGHTNESS_PERCENTILE = 95  # Top 5% = 95th percentile


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_nd2_files(input_folder: Path) -> list[Path]:
    """Check that all ND2 files have consistent dimensions."""
    nd2_files = sorted(input_folder.glob("*.nd2"))
    
    if not nd2_files:
        raise FileNotFoundError(f"No .nd2 files found in {input_folder}")
    
    # Use first file as reference
    with nd2.ND2File(nd2_files[0]) as f:
        xarr0 = f.to_xarray()
        expected_dims = xarr0.dims
        expected_shape = xarr0.shape
    
    print(f"✓ Found {len(nd2_files)} ND2 files")
    print(f"  Expected dims: {expected_dims}")
    print(f"  Expected shape: {expected_shape}\n")
    
    # Check remaining files
    mismatch_found = False
    for nd2_path in nd2_files[1:]:
        with nd2.ND2File(nd2_path) as f:
            xarr = f.to_xarray()
            dims = xarr.dims
            shape = xarr.shape
        
        if (dims != expected_dims) or (shape != expected_shape):
            if not mismatch_found:
                print("⚠️  Files with differing dimensions:")
                mismatch_found = True
            print(f"   {nd2_path.name}: dims={dims}, shape={shape}")
    
    if not mismatch_found:
        print("✓ All files match expected dimensions.\n")
    
    return nd2_files


def create_mips(nd2_files: list[Path], output_folder: Path, channel: int) -> None:
    """Convert ND2 files to maximum intensity projections."""
    output_folder.mkdir(exist_ok=True, parents=True)
    
    print(f"Creating MIPs (channel {channel})...")
    for nd2_path in tqdm(nd2_files, desc="  Processing"):
        with nd2.ND2File(nd2_path) as f:
            data = np.asarray(f)
            data_ch = data[:, channel, :, :]
            mip = data_ch.max(axis=0).astype(data.dtype, copy=False)
        
        tif_path = output_folder / f"{nd2_path.stem}_ch{channel}_MIP.tif"
        tifffile.imwrite(tif_path, mip, imagej=True)
    
    print("✓ MIPs created.\n")


def background_subtract_mips(mip_files: list[Path], output_folder: Path) -> None:
    """Subtract median background from MIP images."""
    output_folder.mkdir(exist_ok=True, parents=True)
    
    print("Background subtracting MIPs...")
    for mip_path in tqdm(mip_files, desc="  Processing"):
        img = tifffile.imread(mip_path).astype(np.float32)
        bg_val = np.median(img)
        img_bgsub = img - bg_val
        img_bgsub[img_bgsub < 0] = 0
        
        out_path = output_folder / f"{mip_path.stem}_bgsub.tif"
        tifffile.imwrite(out_path, img_bgsub.astype(np.float32))
    
    print("✓ Background subtraction complete.\n")


def parse_filename_metadata(stem: str) -> dict | None:
    """
    Parse metadata from filename using regex.
    
    Expected format: YYYYMMDD_SWFxxxx_Xx_power_animal
    Example: 20251204_SWF1454_X3_13_3
    """
    pattern = re.compile(
        r"(?P<date>\d{8})_(?P<strain>SWF\d{4})_(?P<outcrossed>X\d{1})_(?P<power>\d+?)_(?P<animal>\d+)"
    )
    
    m = pattern.search(stem)
    if not m:
        return None
    
    return {
        "date": m.group("date"),
        "strain": m.group("strain"),
        "outcrossed": m.group("outcrossed"),
        "laser_power": int(m.group("power")),
        "animal": int(m.group("animal")),
    }


def organize_data(data_folder: Path) -> pd.DataFrame:
    """Parse metadata from TIFF files and create metadata DataFrame."""
    files = sorted(data_folder.glob("*.tif"))
    
    if not files:
        raise FileNotFoundError(f"No TIFF files found in {data_folder}")
    
    print(f"Organizing {len(files)} TIFF files...")
    
    records = []
    skipped = 0
    
    for p in files:
        stem = p.stem
        metadata = parse_filename_metadata(stem)
        
        if not metadata:
            print(f"  ⚠️  Skipping (name doesn't match pattern): {p.name}")
            skipped += 1
            continue
        
        records.append({
            "path": p,
            "stem": stem,
            **metadata,
        })
    
    df_meta = pd.DataFrame(records)
    df_meta = df_meta.sort_values(["strain", "laser_power", "animal"]).reset_index(drop=True)
    
    print(f"✓ Parsed {len(records)} files ({skipped} skipped).\n")
    
    return df_meta


def compute_brightness_metrics(df_meta: pd.DataFrame, percentile: int = 95) -> pd.DataFrame:
    """Compute brightness statistics for each animal."""
    print(f"Computing brightness metrics (top {100 - percentile}%)...")
    
    records = []
    for row in tqdm(df_meta.itertuples(index=False), total=len(df_meta), desc="  Processing"):
        img = tifffile.imread(row.path).astype(np.float32)
        vals = img.flatten()
        
        cut = np.percentile(vals, percentile)
        top_vals = vals[vals >= cut]
        
        bright_mean = np.mean(top_vals)
        bright_median = np.median(top_vals)
        
        records.append({
            "strain": row.strain,
            "laser_power": row.laser_power,
            "animal": row.animal,
            "bright_mean": bright_mean,
            "bright_median": bright_median,
        })
    
    df_bright = pd.DataFrame(records)
    print(f"✓ Computed metrics for {len(df_bright)} animals.\n")
    
    return df_bright


def designate_controls(df_meta: pd.DataFrame) -> pd.DataFrame:
    """Add control/experimental designation to metadata."""
    df_meta["is_control"] = (
        (df_meta["strain"] == CONTROL_STRAIN) &
        (df_meta["laser_power"] == CONTROL_POWER)
    )
    
    df_meta["condition_label"] = df_meta.apply(
        lambda row: "control" if row["is_control"] else "experimental",
        axis=1
    )
    
    n_control = df_meta["is_control"].sum()
    print(f"✓ Designated {n_control} control animals.\n")
    
    return df_meta


def plot_mean_brightness(df_bright: pd.DataFrame, output_folder: Path) -> None:
    """Plot mean brightness vs laser power for each strain."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    summary = df_bright.groupby(["strain", "laser_power"])["bright_mean"].agg(
        ["mean", "sem"]
    ).reset_index()
    
    for strain in summary["strain"].unique():
        sub = summary[summary["strain"] == strain]
        ax.errorbar(
            sub["laser_power"], sub["mean"], yerr=sub["sem"],
            marker='o', label=strain, linewidth=2, capsize=5
        )
    
    ax.set_xlabel("Laser Power", fontsize=12)
    ax.set_ylabel("Mean brightness (top 5%)", fontsize=12)
    ax.set_title("Brightness response curve by strain")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    out_path = output_folder / "brightness_mean_by_power.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {out_path}")
    plt.close()


def plot_median_brightness(df_bright: pd.DataFrame, output_folder: Path) -> None:
    """Plot median brightness with quartile error bars."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    summary = df_bright.groupby(["strain", "laser_power"])["bright_mean"].agg(
        median='median',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    ).reset_index()
    
    for strain in summary["strain"].unique():
        sub = summary[summary["strain"] == strain]
        ax.errorbar(
            sub["laser_power"], sub["median"],
            yerr=[sub["median"] - sub["q25"], sub["q75"] - sub["median"]],
            marker='o', label=strain, linewidth=2, capsize=5
        )
    
    ax.set_xlabel("Laser Power", fontsize=12)
    ax.set_ylabel("Median brightness (top 5%)", fontsize=12)
    ax.set_title("Brightness response curve by strain (median ± IQR)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    out_path = output_folder / "brightness_median_by_power.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {out_path}")
    plt.close()


def plot_strain_vs_control(
    df_bright: pd.DataFrame,
    strain_name: str,
    metric: str = "median",
    output_folder: Path = None
) -> None:
    """Plot one experimental strain compared to control."""
    
    # Get control data
    mask_ctrl = (
        (df_bright["strain"] == CONTROL_STRAIN) &
        (df_bright["laser_power"] == CONTROL_POWER)
    )
    ctrl_vals = df_bright[mask_ctrl]["bright_median"].values
    ctrl_metric = np.median(ctrl_vals) if metric == "median" else np.mean(ctrl_vals)
    
    # Build plot dataframe
    plot_data = []
    
    # Add control
    for val in ctrl_vals:
        plot_data.append({
            "condition": f"{CONTROL_STRAIN}\n(P{CONTROL_POWER})",
            "bright_median": val,
            "type": "control"
        })
    
    # Add experimental (all powers for this strain)
    strain_powers = sorted(
        df_bright[df_bright["strain"] == strain_name]["laser_power"].unique()
    )
    
    for p in strain_powers:
        mask_exp = (
            (df_bright["strain"] == strain_name) &
            (df_bright["laser_power"] == p)
        )
        exp_vals = df_bright[mask_exp]["bright_median"].values
        
        if len(exp_vals) == 0:
            continue
        
        condition_metric = np.median(exp_vals) if metric == "median" else np.mean(exp_vals)
        color_type = "below" if condition_metric < ctrl_metric else "above"
        
        for val in exp_vals:
            plot_data.append({
                "condition": f"{strain_name}\n(P{p})",
                "bright_median": val,
                "type": color_type
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create palette
    conditions = plot_df["condition"].unique()
    palette = {}
    for cond in conditions:
        cond_type = plot_df[plot_df["condition"] == cond]["type"].iloc[0]
        if cond_type == "control":
            palette[cond] = "#808080"  # Grey
        elif cond_type == "below":
            palette[cond] = "#1f77b4"  # Blue
        else:
            palette[cond] = "#ff7f0e"  # Orange
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(
        data=plot_df, x="condition", y="bright_median",
        palette=palette, showcaps=True, boxprops={'alpha': 0.5},
        width=0.6, ax=ax
    )
    
    sns.swarmplot(
        data=plot_df, x="condition", y="bright_median",
        color="black", size=6, edgecolor="black", linewidth=0.5,
        zorder=10, ax=ax
    )
    
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel(f"{metric.capitalize()} brightness (top 5%)", fontsize=12)
    ax.set_title(f"{strain_name} vs Control ({metric})", fontsize=14)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, 40)
    
    plt.tight_layout()
    
    if output_folder:
        out_path = output_folder / f"strain_{strain_name}_vs_control_{metric}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {out_path}")
    
    plt.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(input_folder: Path, output_base: Path = None, skip_mip: bool = False) -> None:
    """
    Main brightness analysis pipeline.
    
    Args:
        input_folder: Path to folder containing ND2 files
        output_base: Path for outputs (default: input_folder/analysis)
        skip_mip: If True, skip MIP creation (assume they exist)
    """
    
    if output_base is None:
        output_base = input_folder / "analysis"
    
    output_base.mkdir(exist_ok=True, parents=True)
    
    print("=" * 70)
    print("BRIGHTNESS CHECKER PIPELINE")
    print("=" * 70 + "\n")
    
    # --- Validate ND2 files ---
    nd2_files = validate_nd2_files(input_folder)
    
    # --- Create MIPs ---
    if not skip_mip:
        mip_folder = output_base / "mips"
        create_mips(nd2_files, mip_folder, CHANNEL_OF_INTEREST)
    else:
        mip_folder = output_base / "mips"
        print(f"Skipping MIP creation (using existing files in {mip_folder})\n")
    
    # --- Background subtraction ---
    mip_files = sorted(mip_folder.glob("*_MIP.tif"))
    bgsub_folder = output_base / "bgsub"
    background_subtract_mips(mip_files, bgsub_folder)
    
    # --- Parse metadata ---
    df_meta = organize_data(bgsub_folder)
    
    # --- Designate controls ---
    df_meta = designate_controls(df_meta)
    
    # --- Compute brightness ---
    df_bright = compute_brightness_metrics(df_meta, percentile=BRIGHTNESS_PERCENTILE)
    
    # --- Save results ---
    results_folder = output_base / "results"
    results_folder.mkdir(exist_ok=True, parents=True)
    
    df_bright.to_csv(results_folder / "brightness_metrics.csv", index=False)
    print(f"✓ Saved: brightness_metrics.csv\n")
    
    # --- Generate plots ---
    print("Generating plots...")
    plot_mean_brightness(df_bright, results_folder)
    plot_median_brightness(df_bright, results_folder)
    
    # Plot each strain vs control
    experimental_strains = df_bright[
        df_bright["strain"] != CONTROL_STRAIN
    ]["strain"].unique()
    
    for strain in sorted(experimental_strains):
        plot_strain_vs_control(df_bright, strain, metric="median", output_folder=results_folder)
    
    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {results_folder}\n")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze brightness in immobilized sensor animals.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python brightness_checker.py /path/to/nd2/files
  python brightness_checker.py /path/to/nd2/files --output /custom/output
  python brightness_checker.py /path/to/nd2/files --skip-mip
        """
    )
    
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Folder containing ND2 microscopy files"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output folder (default: input_folder/analysis)"
    )
    
    parser.add_argument(
        "--skip-mip",
        action="store_true",
        help="Skip MIP creation (assume MIPs already exist)"
    )
    
    args = parser.parse_args()
    
    main(args.input_folder, args.output, args.skip_mip)
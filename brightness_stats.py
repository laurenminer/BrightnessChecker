"""
Brightness Statistics Extractor

Analyzes TIF image files and extracts mean and median of top 5% brightest pixels.
Outputs two CSV files with the results.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
from typing import Tuple
import argparse


def load_tif_image(filepath: str) -> np.ndarray:
    """
    Load a TIF image file
    
    Parameters
    ----------
    filepath : str
        Path to the TIF file
        
    Returns
    -------
    np.ndarray
        Image data as numpy array
    """
    img = Image.open(filepath)
    return np.array(img)


def get_top_percentile_pixels(image: np.ndarray, percentile: float = 5.0) -> np.ndarray:
    """
    Extract top percentile brightest pixels from an image
    
    Parameters
    ----------
    image : np.ndarray
        Image data (2D or 3D array)
    percentile : float
        Percentile threshold (default: 5.0 for top 5%)
        
    Returns
    -------
    np.ndarray
        Pixels in the top percentile by brightness
    """
    # Flatten image to 1D
    flat_image = image.flatten()
    
    # Calculate the threshold value for top percentile
    threshold = np.percentile(flat_image, 100 - percentile)
    
    # Get pixels above threshold
    top_pixels = flat_image[flat_image >= threshold]
    
    return top_pixels


def process_image(filepath: str, percentile: float = 5.0) -> Tuple[float, float]:
    """
    Process a single image and return mean and median of top percentile pixels
    
    Parameters
    ----------
    filepath : str
        Path to image file
    percentile : float
        Percentile threshold (default: 5.0)
        
    Returns
    -------
    tuple
        (mean, median) of top percentile pixels
    """
    try:
        # Load image
        image = load_tif_image(filepath)
        
        # Get top percentile pixels
        top_pixels = get_top_percentile_pixels(image, percentile)
        
        # Calculate statistics
        mean_val = np.mean(top_pixels)
        median_val = np.median(top_pixels)
        
        return mean_val, median_val
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return np.nan, np.nan


def process_folder(folder_path: str, output_dir: str = None, percentile: float = 5.0, verbose: bool = True):
    """
    Process all TIF files in a folder and output CSV files
    
    Parameters
    ----------
    folder_path : str
        Path to folder containing TIF files
    output_dir : str, optional
        Directory to save output CSVs (default: same as folder_path)
    percentile : float
        Percentile threshold (default: 5.0 for top 5%)
    verbose : bool
        Print progress information
    """
    
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Set output directory
    if output_dir is None:
        output_dir = folder
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all TIF files
    tif_files = sorted(list(folder.glob("*.tif")) + list(folder.glob("*.tiff")))
    
    if not tif_files:
        raise FileNotFoundError(f"No TIF files found in {folder_path}")
    
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"BRIGHTNESS STATISTICS EXTRACTOR")
        print(f"{'=' * 70}")
        print(f"Processing {len(tif_files)} TIF files from: {folder_path}")
        print(f"Top percentile: {percentile}%")
        print(f"Output directory: {output_dir}\n")
    
    # Process each file
    filenames = []
    means = []
    medians = []
    
    for i, filepath in enumerate(tif_files, 1):
        if verbose:
            print(f"[{i}/{len(tif_files)}] Processing: {filepath.name}")
        
        mean_val, median_val = process_image(str(filepath), percentile)
        
        filenames.append(filepath.name)
        means.append(mean_val)
        medians.append(median_val)
    
    # Create DataFrames
    df_means = pd.DataFrame({
        "filename": filenames,
        f"mean_top_{percentile}pct": means
    })
    
    df_medians = pd.DataFrame({
        "filename": filenames,
        f"median_top_{percentile}pct": medians
    })
    
    # Save CSVs
    means_output = output_dir / f"brightness_means_top{percentile}pct.csv"
    medians_output = output_dir / f"brightness_medians_top{percentile}pct.csv"
    
    df_means.to_csv(means_output, index=False)
    df_medians.to_csv(medians_output, index=False)
    
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Results saved:")
        print(f"  Means:   {means_output}")
        print(f"  Medians: {medians_output}")
        print(f"{'=' * 70}\n")
    
    return df_means, df_medians


def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(
        description="Extract brightness statistics from TIF images"
    )
    parser.add_argument("input_folder", help="Folder containing TIF files")
    parser.add_argument(
        "-o", "--output",
        help="Output directory (default: same as input folder)",
        default=None
    )
    parser.add_argument(
        "-p", "--percentile",
        type=float,
        default=5.0,
        help="Percentile threshold (default: 5.0 for top 5%%)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    process_folder(
        args.input_folder,
        output_dir=args.output,
        percentile=args.percentile,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
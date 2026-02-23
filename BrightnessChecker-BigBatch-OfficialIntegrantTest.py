#!/usr/bin/env python
# coding: utf-8

# ## Brightness checking immobilized sensor animals

# ### Housekeeping

# In[8]:


# Dependencies
import argparse
from pathlib import Path
import numpy as np
import tifffile
import nd2
import xarray as xr
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.patches import Polygon
import pandas as pd
import csv
from tqdm.auto import tqdm
import re
import seaborn as sns


# In[3]:


# Locate the folder with your images and set up where to save
input_folder = Path('/store1/lauren/Tetramisole_Immobilized_Imaging/20260212_OfficialBrightnessCheck')
output_folder = input_folder / "tiffs"
output_folder.mkdir(exist_ok=True)

print("Input folder:", input_folder)
print("Output folder:", output_folder)


# ### Data Checks

# In[4]:


# Check that the dimensions are as expected
nd2_files = sorted(input_folder.glob("*.nd2"))

if not nd2_files:
    raise FileNotFoundError(f"No .nd2 files found in {input_folder}")

# Use the first file as the reference
with nd2.ND2File(nd2_files[0]) as f:
    xarr0 = f.to_xarray()
    expected_dims = xarr0.dims
    expected_shape = xarr0.shape

print("Expected dims (from first file):", expected_dims)
print("Expected shape (from first file):", expected_shape)

# Now check the rest against the expected dims/shape
mismatch_found = False

for nd2_path in nd2_files[1:]:
    with nd2.ND2File(nd2_path) as f:
        xarr = f.to_xarray()
        dims = xarr.dims
        shape = xarr.shape

    if (dims != expected_dims) or (shape != expected_shape):
        if not mismatch_found:
            print("\nFiles with differing dimensions:")
            mismatch_found = True
        print(f"{nd2_path.name}: dims={dims}, shape={shape}")

if not mismatch_found:
    print("\nAll files match the expected dimensions.")


# ### Create MIP

# In[5]:


channel_of_interest = 0

nd2_files = sorted(input_folder.glob("*.nd2"))
print(f"Found {len(nd2_files)} ND2 files.")

for nd2_path in tqdm(nd2_files, desc="Making MIPs"):
    with nd2.ND2File(nd2_path) as f:
        data = np.asarray(f)
        data_ch = data[:, channel_of_interest, :, :]
        mip = data_ch.max(axis=0).astype(data.dtype, copy=False)

    tif_path = output_folder / f"{nd2_path.stem}_ch{channel_of_interest}_MIP.tif"
    tifffile.imwrite(tif_path, mip, imagej=True)


# ### Quality Control - Optional

# In[49]:


# Tool to inspect a single image (if you want to)

get_ipython().run_line_magic('matplotlib', 'inline')

mip_files = sorted(output_folder.glob("*.tif"))
print(f"Found {len(mip_files)} MIP tiffs")

# pick one to inspect
example_tif = mip_files[2] # change this index to flip through the images
print("Showing:", example_tif)

img = tifffile.imread(example_tif)
print("Image shape:", img.shape, "dtype:", img.dtype)

plt.figure(figsize=(6, 6))
plt.imshow(img, cmap='gray')
plt.title(example_tif.name)
plt.axis('off')
plt.show()


# # Background Subtraction

# In[6]:


# Output folder for background-subtracted TIFFs
bgsub_folder = output_folder / "bgsub_simple"
bgsub_folder.mkdir(exist_ok=True)

# Find all MIP TIFFs
mip_files = sorted(output_folder.glob("*_MIP.tif"))
print(f"Found {len(mip_files)} MIP TIFFs.")

for mip_path in tqdm(mip_files, desc="Background subtracting MIPs"):
    # Load image
    img = tifffile.imread(mip_path).astype(np.float32)

    # 1) Compute median pixel intensity
    bg_val = np.median(img)

    # 2) Subtract median background
    img_bgsub = img - bg_val
    img_bgsub[img_bgsub < 0] = 0   # Optional: clip negatives

    # 3) Save as TIFF
    out_path = bgsub_folder / f"{mip_path.stem}_bgsub.tif"
    tifffile.imwrite(out_path, img_bgsub.astype(np.float32))

print("\nDone background-subtracting all MIPs.")


# In[7]:


# Display an example of the background subtracted image for quality control

example =  mip_files[2]
stem = example.stem

raw = tifffile.imread(example)
bgsub = tifffile.imread(bgsub_folder / f"{stem}_bgsub.tif")


fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(raw, cmap='gray')
axes[0].set_title("Raw MIP")
axes[0].axis('off')

axes[1].imshow(bgsub, cmap='gray')
axes[1].set_title("Bg-sub MIP")
axes[1].axis('off')

plt.tight_layout()
plt.show()


# ### Organize Data

# In[9]:


# Choose which folder to index – here I assume bgsub MIPs
data_folder = output_folder / "bgsub_simple"   # change to whatever you want
print("Data folder:", data_folder)

files = sorted(data_folder.glob("*.tif"))
print(f"Found {len(files)} TIFF files in {data_folder}")

records = []
pattern = re.compile(
    r"(?P<date>\d{8})_(?P<strain>SWF\d{4})_(?P<outcrossed>X\d{1})_(?P<power>\d+?)_(?P<animal>\d+)"
)

for p in files:
    stem = p.stem  # e.g. '20251204_SWF1454_13_3_ch0_MIP_bgsub'
    m = pattern.search(stem)
    if not m:
        # If anything doesn't match, report it so you can fix/ignore it
        print(f"⚠️  Skipping (name doesn't match pattern): {p.name}")
        continue

    date      = m.group("date")           # '20251204'
    strain    = m.group("strain")         # 'SWF1454'
    outcrossed =m.group("outcrossed")     # 'X5'
    power     = int(m.group("power"))     # e.g. 13
    animal_id = int(m.group("animal"))    # e.g. 3

    records.append({
        "path": p,
        "stem": stem,
        "date": date,
        "strain": strain,
        "outcrossed": outcrossed,
        "laser_power": power,
        "animal": animal_id,
    })

df_meta = pd.DataFrame(records)

# Sort for readability
df_meta = df_meta.sort_values(["strain", "laser_power", "animal"]).reset_index(drop=True)

print("\nParsed metadata table (first few rows):")
display(df_meta.head())

print("\nCounts by strain and laser_power:")
display(df_meta.groupby(["strain", "laser_power"]).size().unstack(fill_value=0))


# In[12]:


#designate control data
# Define control condition
CONTROL_STRAIN = "SWF1088"
CONTROL_POWER  = 13

df_meta["is_control"] = (
    (df_meta["strain"] == CONTROL_STRAIN) &
    (df_meta["laser_power"] == CONTROL_POWER)
)

# Optional: also create a readable label
df_meta["condition_label"] = df_meta.apply(
    lambda row: "control" if row.is_control else "experimental",
    axis=1
)

display(df_meta.head())


# ### Compute Brightest Pixels

# In[10]:


## Compute the top 5% brightest pixels per animal,
## grouped by (strain, laser_power) condition.

# Dictionary: keys are (strain, laser_power), values are lists of arrays (one per animal)
top5_by_condition = {}

for row in df_meta.itertuples(index=False):
    # row has fields: path, stem, date, strain, laser_power, animal, ...
    strain = row.strain
    power  = row.laser_power
    path   = row.path

    condition = (strain, power)

    # Ensure key exists
    if condition not in top5_by_condition:
        top5_by_condition[condition] = []

    # Load image
    img = tifffile.imread(path).astype(np.float32)
    vals = img.flatten()

    # Compute top 5% for *this animal*
    cut = np.percentile(vals, 95)
    top_vals = vals[vals >= cut]

    top5_by_condition[condition].append(top_vals)

    # Optional light print (comment out if noisy)
    # print(f"{row.stem}: condition={condition}, top5 count={len(top_vals)}")

print("\nDONE computing per-animal top-5% arrays for each (strain, laser_power) condition.")


# ### Evaluate Laser Powers - MEAN

# In[12]:


# Convert top5_by_condition to df_bright
records = []

for (strain, power), top5_arrays in top5_by_condition.items():
    # top5_arrays is a list of arrays, one per animal
    for animal_idx, top5_vals in enumerate(top5_arrays):
        bright_mean = np.mean(top5_vals)
        records.append({
            "strain": strain,
            "laser_power": power,
            "animal": animal_idx,
            "bright_mean": bright_mean,
        })

df_bright = pd.DataFrame(records)

print("df_bright created with shape:", df_bright.shape)
display(df_bright.head())
fig, ax = plt.subplots(figsize=(12, 6))

# Calculate mean and sem for each strain/power combination
summary = df_bright.groupby(["strain","laser_power"])["bright_mean"].agg(["mean", "sem"]).reset_index()

# Plot each strain
for strain in summary["strain"].unique():
    sub = summary[summary["strain"] == strain]
    ax.errorbar(sub["laser_power"], sub["mean"], yerr=sub["sem"], 
                marker='o', label=strain, linewidth=2, capsize=5)

ax.set_xlabel("Laser Power", fontsize=12)
ax.set_ylabel("Mean brightness (top 5%)", fontsize=12)
ax.set_title("Brightness response curve by strain")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ### Evaluate Laser Powers - MEDIAN

# In[13]:


# Convert top5_by_condition to df_bright
records = []

for (strain, power), top5_arrays in top5_by_condition.items():
    # top5_arrays is a list of arrays, one per animal
    for animal_idx, top5_vals in enumerate(top5_arrays):
        bright_mean = np.mean(top5_vals)
        records.append({
            "strain": strain,
            "laser_power": power,
            "animal": animal_idx,
            "bright_mean": bright_mean,
        })

df_bright = pd.DataFrame(records)

print("df_bright created with shape:", df_bright.shape)
display(df_bright.head())

fig, ax = plt.subplots(figsize=(12, 6))

# Calculate MEDIAN and quartiles for each strain/power combination
summary = df_bright.groupby(["strain","laser_power"])["bright_mean"].agg(
    median='median',
    q25=lambda x: x.quantile(0.25),
    q75=lambda x: x.quantile(0.75)
).reset_index()

# Plot each strain
for strain in summary["strain"].unique():
    sub = summary[summary["strain"] == strain]
    # Error bars show 25th to 75th percentile range
    ax.errorbar(sub["laser_power"], sub["median"], 
                yerr=[sub["median"] - sub["q25"], sub["q75"] - sub["median"]], 
                marker='o', label=strain, linewidth=2, capsize=5)

ax.set_xlabel("Laser Power", fontsize=12)
ax.set_ylabel("Median brightness (top 5%)", fontsize=12)
ax.set_title("Brightness response curve by strain")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# In[14]:


import numpy as np
import pandas as pd
import tifffile

records = []

for row in df_meta.itertuples(index=False):
    # row has: path, stem, date, strain, laser_power, animal, ...
    img = tifffile.imread(row.path).astype(np.float32)
    vals = img.flatten()

    # top 5% for this animal
    cut = np.percentile(vals, 95)
    top_vals = vals[vals >= cut]
    bright_mean = top_vals.mean()
    bright_median = np.median(top_vals)  # ← FIXED: Use np.median()

    records.append({
        "strain":      row.strain,
        "laser_power": row.laser_power,
        "animal":      row.animal,
        "bright_mean": bright_mean,
        "bright_median": bright_median,  # ← ADD THIS to the dict!
    })

df_bright = pd.DataFrame(records)

print("Per-animal brightness table (first few rows):")
display(df_bright.head())


# ## Visualize Per Strain

# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

CONTROL_STRAIN = "SWF1088"
CONTROL_POWER  = 13

def plot_strain_vs_control(strain_name, metric="median", title=None):
    """
    Plot one strain with:
      - Control (SWF1088 @ power 13) in GREY
      - All laser powers for the given strain in BLUE (below control) or ORANGE (>= control)
    
    Uses boxplot + swarmplot (seaborn style)
    metric: "mean" or "median"
    """
    
    # --- Get control data and metric ---
    mask_ctrl = (
        (df_bright["strain"] == CONTROL_STRAIN) &
        (df_bright["laser_power"] == CONTROL_POWER)
    )
    ctrl_vals = df_bright[mask_ctrl]["bright_median"].values
    
    if metric == "median":
        ctrl_metric = np.median(ctrl_vals)
    else:  # mean
        ctrl_metric = np.mean(ctrl_vals)

    # --- Build dataframe for seaborn ---
    plot_data = []

    # Add control
    for val in ctrl_vals:
        plot_data.append({
            "condition": f"{CONTROL_STRAIN}\n(P{CONTROL_POWER})",
            "bright_median": val,
            "type": "control"
        })

    # Add experimental (all powers for this strain)
    strain_powers = sorted(df_bright[df_bright["strain"] == strain_name]["laser_power"].unique())
    
    for p in strain_powers:
        mask_exp = (
            (df_bright["strain"] == strain_name) &
            (df_bright["laser_power"] == p)
        )
        exp_vals = df_bright[mask_exp]["bright_median"].values
        
        if len(exp_vals) == 0:
            continue
        
        # Calculate metric
        if metric == "median":
            condition_metric = np.median(exp_vals)
        else:
            condition_metric = np.mean(exp_vals)
        
        # Determine color type
        color_type = "below" if condition_metric < ctrl_metric else "above"
        
        for val in exp_vals:
            plot_data.append({
                "condition": f"{strain_name}\n(P{p})",
                "bright_median": val,
                "type": color_type
            })

    plot_df = pd.DataFrame(plot_data)

    # --- Create palette ---
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

    # --- Create plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Boxplot
    sns.boxplot(
        data=plot_df,
        x="condition",
        y="bright_median",
        palette=palette,
        showcaps=True,
        boxprops={'alpha': 0.5},
        width=0.6,
        ax=ax
    )

    # Swarmplot overlay
    sns.swarmplot(
        data=plot_df,
        x="condition",
        y="bright_median",
        color="black",
        size=6,
        edgecolor="black",
        linewidth=0.5,
        zorder=10,
        ax=ax
    )

    # --- Axes formatting ---
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel(f"{metric.capitalize()} brightness (top 5%)", fontsize=12)
    ax.set_title(
        title if title else f"{strain_name} vs Control ({metric})",
        fontsize=14
    )
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, 40)

    plt.tight_layout()
    plt.show()

# Make plots for each strain with MEAN
for strain in sorted(df_bright[df_bright["strain"] != CONTROL_STRAIN]["strain"].unique()):
    plot_strain_vs_control(strain, metric="median")

# Or individual plot:
# plot_strain_vs_control("SWF1487", metric="median")


# In[86]:


# Plot Images

import matplotlib.pyplot as plt
import tifffile
import numpy as np

# Fixed scale
VMIN = 0
VMAX = 150   # adjust as needed

# Merge brightness info with paths if not already merged
df_bp = df_bright.merge(
    df_meta[["strain", "laser_power", "animal", "path"]],
    on=["strain", "laser_power", "animal"],
    how="left"
)

# 1) Median bright SWF1088 animal (across all powers)
sub_1088 = df_bp[df_bp["strain"] == "SWF1088"].sort_values("bright_mean")
median_idx = len(sub_1088) // 2
row_1088 = sub_1088.iloc[median_idx]

# 2) Second brightest SWF1455 animal at laser power 21
sub_1492 = df_bp[df_bp["strain"] == "SWF149205"].sort_values("bright_mean")
median_idx = len(sub_1492) // 2
row_1492 = sub_1492.iloc[median_idx]

# ---- Plot the two selected animals ----
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

selected = [
    (row_1088, f"SWF1088"),
    (row_1492, f"SWF1492")
]

im_last = None

for ax, (row, title) in zip(axes, selected):
    img = tifffile.imread(row["path"]).astype(np.float32)
        # Rotate SWF1088 by 90 degrees
    if title == "SWF1088":
        img = np.flipud(img)
    im = ax.imshow(img, cmap="inferno", vmin=VMIN, vmax=VMAX, interpolation="none")
    im_last = im
    ax.set_title(title)
    ax.axis("off")

# Tighten layout before adding manual colorbar
plt.tight_layout()

# Shared bottom colorbar
cbar_ax = fig.add_axes([0.25, 0.07, 0.50, 0.03])  # [left, bottom, width, height]
cbar = fig.colorbar(im_last, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Intensity", rotation=0, labelpad=5)

plt.show()


# In[56]:


import matplotlib.pyplot as plt
import tifffile
import numpy as np

# fixed display range
VMIN = 0
VMAX = 1500   # <-- adjust if needed

# same order used before
genotype_order = ["SWF1088_GCaMP", "SWF1454", "SWF1455"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for col, genotype in enumerate(genotype_order):
    
    row = rep_rows[rep_rows.genotype == genotype].iloc[0]
    img = tifffile.imread(row["path"]).astype(np.float32)

    ax = axes[col]
    im = ax.imshow(img, cmap="gray", vmin=VMIN, vmax=VMAX, interpolation="none")
    ax.set_title(genotype)
    ax.axis("off")

plt.tight_layout()
plt.show()


# In[65]:


import matplotlib.pyplot as plt
import tifffile
import numpy as np

# Fixed scale
VMIN = 0
VMAX = 1000   # adjust as needed

genotype_order = ["SWF1088_GCaMP", "SWF1454", "SWF1455"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# We will store the last image object so we can attach a colorbar
im_last = None

for col, genotype in enumerate(genotype_order):
    row = rep_rows[rep_rows.genotype == genotype].iloc[0]
    img = tifffile.imread(row["path"]).astype(np.float32)

    ax = axes[col]
    im = ax.imshow(img, cmap="inferno", vmin=VMIN, vmax=VMAX, interpolation="none")
    im_last = im  # save for colorbar

    ax.set_title(genotype)
    ax.axis("off")

# FIRST: fix all subplot spacing
plt.tight_layout()

# THEN: add the manual colorbar axis at the bottom
cbar_ax = fig.add_axes([0.25, 0.07, 0.50, 0.03])  # [left, bottom, width, height]
cbar = fig.colorbar(im_last, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Intensity", rotation=0, labelpad=5)

plt.show()


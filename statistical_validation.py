from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import f_regression
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

FEATURE_CSV = ROOT / "feature_dataset.csv"

PREPROCESSED_DIR = ROOT / "dataset" / "preprocessed"
REFERENCE_DIR = ROOT / "dataset" / "reference-890"

OUTPUT_DIR = ROOT / "feature_ranking_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXISTING 15 NON-REDUNDANT FEATURES
# ============================================================

SELECTED_FEATURES = [
    "edge_density",
    "energy",
    "std",
    "red_ratio",
    "entropy",
    "mean_blue",
    "keypoint_density",
    "mean_green",
    "mean_red",
    "dynamic_range",
    "mean_saturation",
    "laplacian_variance",
    "colorfulness",
    "correlation",
    "homogeneity"
]


# ============================================================
# LOAD FEATURE DATASET
# ============================================================

print("=" * 70)
print("STATISTICAL FEATURE VALIDATION")
print("=" * 70)

print("\nLoading feature dataset...")

df = pd.read_csv(FEATURE_CSV)

print(f"Dataset shape: {df.shape}")


# ============================================================
# CHECK FEATURE COLUMNS
# ============================================================

missing_features = [
    feature
    for feature in SELECTED_FEATURES
    if feature not in df.columns
]

if missing_features:
    print("\nERROR: Missing feature columns:")

    for feature in missing_features:
        print(f"  - {feature}")

    raise ValueError(
        "Some selected features are missing from feature_dataset.csv"
    )


# ============================================================
# CALCULATE SSIM AND PSNR
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING SSIM AND PSNR")
print("=" * 70)

ssim_values = []
psnr_values = []

valid_rows = []
removed_images = []


for index, row in df.iterrows():

    image_name = str(row["image_name"])

    preprocessed_path = PREPROCESSED_DIR / image_name
    reference_path = REFERENCE_DIR / image_name

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not preprocessed_path.exists():
        removed_images.append(
            (image_name, "preprocessed image not found")
        )
        continue

    if not reference_path.exists():
        removed_images.append(
            (image_name, "reference image not found")
        )
        continue

    # --------------------------------------------------------
    # Read images
    # --------------------------------------------------------

    pred = cv2.imread(
        str(preprocessed_path),
        cv2.IMREAD_COLOR
    )

    ref = cv2.imread(
        str(reference_path),
        cv2.IMREAD_COLOR
    )

    if pred is None or ref is None:
        removed_images.append(
            (image_name, "image could not be read")
        )
        continue

    # --------------------------------------------------------
    # Resize reference if necessary
    # --------------------------------------------------------

    if pred.shape[:2] != ref.shape[:2]:

        ref = cv2.resize(
            ref,
            (pred.shape[1], pred.shape[0]),
            interpolation=cv2.INTER_AREA
        )

    # --------------------------------------------------------
    # Convert BGR → RGB
    # --------------------------------------------------------

    pred_rgb = cv2.cvtColor(
        pred,
        cv2.COLOR_BGR2RGB
    )

    ref_rgb = cv2.cvtColor(
        ref,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # SSIM
    # --------------------------------------------------------

    ssim = structural_similarity(
        pred_rgb,
        ref_rgb,
        channel_axis=2,
        data_range=255
    )

    # --------------------------------------------------------
    # PSNR
    # --------------------------------------------------------

    psnr = peak_signal_noise_ratio(
        ref_rgb,
        pred_rgb,
        data_range=255
    )

    valid_rows.append(index)

    ssim_values.append(ssim)
    psnr_values.append(psnr)


# ============================================================
# KEEP ONLY VALID IMAGES
# ============================================================

df_valid = df.loc[valid_rows].copy()

df_valid = df_valid.reset_index(drop=True)

df_valid["SSIM"] = ssim_values
df_valid["PSNR"] = psnr_values


print(f"\nTotal images        : {len(df)}")
print(f"Valid image pairs   : {len(df_valid)}")
print(f"Removed image pairs : {len(removed_images)}")


if len(df_valid) < 10:
    raise ValueError(
        "Too few valid image pairs for statistical analysis."
    )


# ============================================================
# STATISTICAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("STATISTICAL TESTING")
print("=" * 70)

results = []

ALPHA = 0.05


for feature in SELECTED_FEATURES:

    X = df_valid[feature].astype(float).values

    row = {
        "feature": feature
    }

    # ========================================================
    # SSIM
    # ========================================================

    y_ssim = df_valid["SSIM"].astype(float).values

    pearson_ssim_r, pearson_ssim_p = pearsonr(
        X,
        y_ssim
    )

    f_ssim, f_ssim_p = f_regression(
        X.reshape(-1, 1),
        y_ssim
    )

    # ========================================================
    # PSNR
    # ========================================================

    y_psnr = df_valid["PSNR"].astype(float).values

    pearson_psnr_r, pearson_psnr_p = pearsonr(
        X,
        y_psnr
    )

    f_psnr, f_psnr_p = f_regression(
        X.reshape(-1, 1),
        y_psnr
    )

    # ========================================================
    # STORE
    # ========================================================

    row["ssim_pearson_r"] = pearson_ssim_r
    row["ssim_pearson_p"] = pearson_ssim_p

    row["ssim_f_statistic"] = f_ssim[0]
    row["ssim_f_p"] = f_ssim_p[0]

    row["psnr_pearson_r"] = pearson_psnr_r
    row["psnr_pearson_p"] = pearson_psnr_p

    row["psnr_f_statistic"] = f_psnr[0]
    row["psnr_f_p"] = f_psnr_p[0]

    # ========================================================
    # SIGNIFICANCE
    # ========================================================

    row["ssim_significant"] = (
        "Yes"
        if pearson_ssim_p < ALPHA
        else "No"
    )

    row["psnr_significant"] = (
        "Yes"
        if pearson_psnr_p < ALPHA
        else "No"
    )

    results.append(row)

    # ========================================================
    # PRINT
    # ========================================================

    print(f"\n{feature}")

    print(
        f"  SSIM : "
        f"r = {pearson_ssim_r:.4f}, "
        f"p = {pearson_ssim_p:.6f}, "
        f"F = {f_ssim[0]:.4f}, "
        f"F-p = {f_ssim_p[0]:.6f}"
    )

    print(
        f"  PSNR : "
        f"r = {pearson_psnr_r:.4f}, "
        f"p = {pearson_psnr_p:.6f}, "
        f"F = {f_psnr[0]:.4f}, "
        f"F-p = {f_psnr_p[0]:.6f}"
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# OVERALL SIGNIFICANCE
# ============================================================

results_df["significant_for_both"] = (
    (results_df["ssim_pearson_p"] < ALPHA)
    &
    (results_df["psnr_pearson_p"] < ALPHA)
)


# ============================================================
# SAVE CSV
# ============================================================

csv_path = (
    OUTPUT_DIR /
    "statistical_feature_validation.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# SAVE EXCEL
# ============================================================

excel_path = (
    OUTPUT_DIR /
    "statistical_feature_validation.xlsx"
)

results_df.to_excel(
    excel_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\nSignificance criterion: p < 0.05")

print("\nFeatures significant for SSIM:")

for feature in results_df.loc[
    results_df["ssim_pearson_p"] < ALPHA,
    "feature"
]:
    print(f"  ✓ {feature}")


print("\nFeatures significant for PSNR:")

for feature in results_df.loc[
    results_df["psnr_pearson_p"] < ALPHA,
    "feature"
]:
    print(f"  ✓ {feature}")


print("\nFeatures significant for BOTH SSIM and PSNR:")

for feature in results_df.loc[
    results_df["significant_for_both"],
    "feature"
]:
    print(f"  ✓ {feature}")


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(f"\nCSV   : {csv_path}")
print(f"Excel : {excel_path}")

print("\nStatistical validation completed successfully.")
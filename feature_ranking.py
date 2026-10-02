from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

# Existing feature dataset
FEATURE_DATASET = ROOT / "feature_dataset.csv"

# Image folders
PREPROCESSED_DIR = ROOT / "dataset" / "preprocessed"
REFERENCE_DIR = ROOT / "dataset" / "reference-890"

# Output folder
OUTPUT_DIR = ROOT / "feature_ranking_results"

# Random seed
RANDOM_STATE = 42


# ============================================================
# 2. READ IMAGE AS RGB
# ============================================================

def read_rgb(path):

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(
            f"Could not read image: {path}"
        )

    # OpenCV reads images in BGR format.
    # Convert to RGB for consistency.
    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# 3. CALCULATE SSIM AND PSNR
# ============================================================

def calculate_quality(preprocessed, reference):

    # --------------------------------------------------------
    # Check image dimensions
    # --------------------------------------------------------

    if preprocessed.shape[:2] != reference.shape[:2]:

        reference = cv2.resize(
            reference,
            (
                preprocessed.shape[1],
                preprocessed.shape[0]
            ),
            interpolation=cv2.INTER_AREA
        )

    # --------------------------------------------------------
    # SSIM
    # --------------------------------------------------------

    ssim = structural_similarity(
        preprocessed,
        reference,
        channel_axis=2,
        data_range=255
    )

    # --------------------------------------------------------
    # PSNR
    # --------------------------------------------------------

    psnr = peak_signal_noise_ratio(
        reference,
        preprocessed,
        data_range=255
    )

    return ssim, psnr


# ============================================================
# 4. LOAD EXISTING FEATURE DATASET
# ============================================================

def load_feature_dataset():

    print("=" * 70)
    print("LOADING EXISTING FEATURE DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Check whether feature_dataset.csv exists
    # --------------------------------------------------------

    if not FEATURE_DATASET.exists():

        raise FileNotFoundError(
            f"\nCould not find:\n{FEATURE_DATASET}\n\n"
            "Make sure feature_dataset.csv is inside the "
            "UIE-fyp project folder."
        )

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    data = pd.read_csv(
        FEATURE_DATASET
    )

    print(
        f"Feature dataset loaded successfully."
    )

    print(
        f"Rows    : {data.shape[0]}"
    )

    print(
        f"Columns : {data.shape[1]}"
    )

    print()

    # --------------------------------------------------------
    # Display column names
    # --------------------------------------------------------

    print("Columns found:")

    for column in data.columns:

        print(" -", column)

    print()

    # --------------------------------------------------------
    # Check image_name column
    # --------------------------------------------------------

    if "image_name" not in data.columns:

        raise ValueError(
            "\n'image_name' column is missing from "
            "feature_dataset.csv.\n"
            "The ranking program needs image_name to "
            "match each feature row with its reference image."
        )

    # --------------------------------------------------------
    # Check for missing values
    # --------------------------------------------------------

    missing_values = data.isnull().sum().sum()

    print(
        f"Total missing values : {missing_values}"
    )

    if missing_values > 0:

        print(
            "\nWARNING: Missing values were found."
        )

    print()

    return data


# ============================================================
# 5. ADD SSIM AND PSNR TO EXISTING DATASET
# ============================================================

def add_quality_metrics(data):

    print("=" * 70)
    print("CALCULATING SSIM AND PSNR")
    print("=" * 70)

    # --------------------------------------------------------
    # Check folders
    # --------------------------------------------------------

    if not PREPROCESSED_DIR.exists():

        raise FileNotFoundError(
            f"\nPreprocessed folder not found:\n"
            f"{PREPROCESSED_DIR}"
        )

    if not REFERENCE_DIR.exists():

        raise FileNotFoundError(
            f"\nReference folder not found:\n"
            f"{REFERENCE_DIR}"
        )

    # --------------------------------------------------------
    # Lists for quality values
    # --------------------------------------------------------

    ssim_values = []
    psnr_values = []

    missing = []

    total = len(data)

    # --------------------------------------------------------
    # Process every image in feature_dataset.csv
    # --------------------------------------------------------

    for index, image_name in enumerate(
        data["image_name"],
        start=1
    ):

        preprocessed_path = (
            PREPROCESSED_DIR / image_name
        )

        reference_path = (
            REFERENCE_DIR / image_name
        )

        # ----------------------------------------------------
        # Check files
        # ----------------------------------------------------

        if not preprocessed_path.exists():

            missing.append(
                f"{image_name} -> preprocessed image missing"
            )

            ssim_values.append(np.nan)
            psnr_values.append(np.nan)

            continue

        if not reference_path.exists():

            missing.append(
                f"{image_name} -> reference image missing"
            )

            ssim_values.append(np.nan)
            psnr_values.append(np.nan)

            continue

        # ----------------------------------------------------
        # Read images
        # ----------------------------------------------------

        preprocessed = read_rgb(
            preprocessed_path
        )

        reference = read_rgb(
            reference_path
        )

        # ----------------------------------------------------
        # Calculate SSIM and PSNR
        # ----------------------------------------------------

        ssim, psnr = calculate_quality(
            preprocessed,
            reference
        )

        ssim_values.append(ssim)
        psnr_values.append(psnr)

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            index == 1
            or index % 25 == 0
            or index == total
        ):

            print(
                f"Processed {index}/{total} images"
            )

    # --------------------------------------------------------
    # Add quality metrics to existing dataset
    # --------------------------------------------------------

    data = data.copy()

    data["ssim"] = ssim_values
    data["psnr"] = psnr_values

    # --------------------------------------------------------
    # Remove incomplete rows
    # --------------------------------------------------------

    before = len(data)

    data = data.dropna(
        subset=["ssim", "psnr"]
    ).reset_index(drop=True)

    removed = before - len(data)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()

    print("=" * 70)
    print("QUALITY METRIC CALCULATION COMPLETED")
    print("=" * 70)

    print(
        f"Images with valid metrics : {len(data)}"
    )

    print(
        f"Images removed             : {removed}"
    )

    if missing:

        print()
        print(
            f"Missing image pairs : {len(missing)}"
        )

        print(
            "\nFirst missing files:"
        )

        for item in missing[:20]:

            print(
                " -", item
            )

    print()

    return data


# ============================================================
# 6. RANDOM FOREST FEATURE RANKING
# ============================================================

def rank_features(data, target_name):

    print("=" * 70)
    print(
        f"FEATURE RANKING USING {target_name.upper()}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Select the 25 feature columns
    #
    # Exclude:
    # image_name
    # ssim
    # psnr
    # --------------------------------------------------------

    feature_names = [
        column
        for column in data.columns
        if column not in {
            "image_name",
            "ssim",
            "psnr"
        }
    ]

    # --------------------------------------------------------
    # Verify number of features
    # --------------------------------------------------------

    print(
        f"Number of features : {len(feature_names)}"
    )

    if len(feature_names) != 25:

        print(
            "\nWARNING:"
        )

        print(
            "Expected 25 handcrafted features, "
            f"but found {len(feature_names)}."
        )

        print(
            "Please check feature_dataset.csv."
        )

    # --------------------------------------------------------
    # Input and target
    # --------------------------------------------------------

    X = data[feature_names]

    y = data[target_name]

    # --------------------------------------------------------
    # Train-test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    print(
        f"Total samples : {len(X)}"
    )

    print(
        f"Training      : {len(X_train)}"
    )

    print(
        f"Testing       : {len(X_test)}"
    )

    print()

    # --------------------------------------------------------
    # Random Forest Regressor
    # --------------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    print(
        "Training Random Forest..."
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Random Forest training completed."
    )

    # --------------------------------------------------------
    # R² score
    # --------------------------------------------------------

    r2_score = model.score(
        X_test,
        y_test
    )

    print(
        f"Held-out R² score : {r2_score:.4f}"
    )

    # --------------------------------------------------------
    # Permutation importance
    # --------------------------------------------------------

    print(
        "Calculating permutation importance..."
    )

    permutation = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring="neg_mean_squared_error"
    )

    print(
        "Permutation importance completed."
    )

    # --------------------------------------------------------
    # Create ranking DataFrame
    # --------------------------------------------------------

    ranking = pd.DataFrame({

        "feature": feature_names,

        "random_forest_importance":
            model.feature_importances_,

        "permutation_importance_mean":
            permutation.importances_mean,

        "permutation_importance_std":
            permutation.importances_std
    })

    # --------------------------------------------------------
    # Sort by permutation importance
    # --------------------------------------------------------

    ranking = ranking.sort_values(
        "permutation_importance_mean",
        ascending=False
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Add rank
    # --------------------------------------------------------

    ranking.insert(
        0,
        "rank",
        range(1, len(ranking) + 1)
    )

    # --------------------------------------------------------
    # Add target
    # --------------------------------------------------------

    ranking.insert(
        1,
        "target",
        target_name
    )

    # --------------------------------------------------------
    # Add R² score
    # --------------------------------------------------------

    ranking["held_out_r2"] = r2_score

    return ranking, r2_score


# ============================================================
# 7. NORMALIZE PERMUTATION IMPORTANCE
# ============================================================

def normalize_importance(ranking):

    ranking = ranking.copy()

    values = ranking[
        "permutation_importance_mean"
    ]

    total = values.abs().sum()

    if total == 0:

        ranking[
            "normalized_importance"
        ] = 0.0

    else:

        ranking[
            "normalized_importance"
        ] = values / total

    return ranking


# ============================================================
# 8. SAVE CSV + EXCEL
# ============================================================

def save_table(data, filename):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_DIR /
        f"{filename}.csv"
    )

    data.to_csv(
        csv_path,
        index=False
    )

    print(
        f"Saved CSV   : {csv_path}"
    )

    # --------------------------------------------------------
    # Excel
    # --------------------------------------------------------

    excel_path = (
        OUTPUT_DIR /
        f"{filename}.xlsx"
    )

    try:

        data.to_excel(
            excel_path,
            index=False
        )

        print(
            f"Saved Excel : {excel_path}"
        )

    except ModuleNotFoundError:

        print()
        print(
            "WARNING: openpyxl is not installed."
        )

        print(
            "Excel file was not created."
        )

        print(
            "CSV file was successfully created."
        )

        print(
            "\nTo enable Excel output, run:"
        )

        print(
            "python -m pip install openpyxl"
        )


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def main():

    print()
    print("=" * 70)
    print("UNDERWATER IMAGE ENHANCEMENT")
    print("FEATURE RANKING SYSTEM")
    print("=" * 70)
    print()

    # ========================================================
    # STEP 1
    # Load existing feature_dataset.csv
    #
    # IMPORTANT:
    # NO FEATURE EXTRACTION IS DONE HERE.
    # ========================================================

    data = load_feature_dataset()

    # ========================================================
    # STEP 2
    # Calculate SSIM and PSNR
    # ========================================================

    data = add_quality_metrics(
        data
    )

    # ========================================================
    # STEP 3
    # Save feature + quality dataset
    # ========================================================

    print("=" * 70)
    print("SAVING FEATURE + QUALITY DATASET")
    print("=" * 70)

    save_table(
        data,
        "feature_quality_dataset"
    )

    # ========================================================
    # STEP 4
    # Rank features using SSIM
    # ========================================================

    ranking_ssim, ssim_r2 = rank_features(
        data,
        "ssim"
    )

    ranking_ssim = normalize_importance(
        ranking_ssim
    )

    save_table(
        ranking_ssim,
        "ranking_ssim"
    )

    # --------------------------------------------------------
    # Display top SSIM features
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TOP 10 FEATURES FOR SSIM")
    print("=" * 70)

    print(
        ranking_ssim[
            [
                "rank",
                "feature",
                "permutation_importance_mean",
                "permutation_importance_std",
                "random_forest_importance"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # ========================================================
    # STEP 5
    # Rank features using PSNR
    # ========================================================

    ranking_psnr, psnr_r2 = rank_features(
        data,
        "psnr"
    )

    ranking_psnr = normalize_importance(
        ranking_psnr
    )

    save_table(
        ranking_psnr,
        "ranking_psnr"
    )

    # --------------------------------------------------------
    # Display top PSNR features
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TOP 10 FEATURES FOR PSNR")
    print("=" * 70)

    print(
        ranking_psnr[
            [
                "rank",
                "feature",
                "permutation_importance_mean",
                "permutation_importance_std",
                "random_forest_importance"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # ========================================================
    # STEP 6
    # Combine SSIM + PSNR rankings
    # ========================================================

    print()
    print("=" * 70)
    print("CREATING COMBINED FEATURE RANKING")
    print("=" * 70)

    # --------------------------------------------------------
    # Keep only required columns
    # --------------------------------------------------------

    ssim_combined = ranking_ssim[
        [
            "feature",
            "normalized_importance"
        ]
    ].copy()

    psnr_combined = ranking_psnr[
        [
            "feature",
            "normalized_importance"
        ]
    ].copy()

    # --------------------------------------------------------
    # Rename columns so we can compare them
    # --------------------------------------------------------

    ssim_combined = ssim_combined.rename(
        columns={
            "normalized_importance":
                "ssim_importance"
        }
    )

    psnr_combined = psnr_combined.rename(
        columns={
            "normalized_importance":
                "psnr_importance"
        }
    )

    # --------------------------------------------------------
    # Merge SSIM and PSNR importance
    # --------------------------------------------------------

    combined = pd.merge(
        ssim_combined,
        psnr_combined,
        on="feature",
        how="inner"
    )

    # --------------------------------------------------------
    # Calculate average importance
    # --------------------------------------------------------

    combined[
        "combined_importance"
    ] = (
        combined["ssim_importance"]
        +
        combined["psnr_importance"]
    ) / 2

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    combined = combined.sort_values(
        "combined_importance",
        ascending=False
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Add final rank
    # --------------------------------------------------------

    combined.insert(
        0,
        "rank",
        range(1, len(combined) + 1)
    )

    # ========================================================
    # STEP 7
    # Save combined ranking
    # ========================================================

    save_table(
        combined,
        "ranking_combined"
    )

    # ========================================================
    # STEP 8
    # Display final ranking
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL COMBINED FEATURE RANKING")
    print("=" * 70)

    print(
        combined.to_string(
            index=False
        )
    )

    # ========================================================
    # STEP 9
    # Display top 10 selected features
    # ========================================================

    print()
    print("=" * 70)
    print("TOP 10 FEATURES")
    print("=" * 70)

    top_features = combined.head(10)

    for _, row in top_features.iterrows():

        print(
            f"{int(row['rank']):2d}. "
            f"{row['feature']:<25} "
            f"{row['combined_importance']:.6f}"
        )

    # ========================================================
    # STEP 10
    # Final summary
    # ========================================================

    print()
    print("=" * 70)
    print("FEATURE RANKING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print()
    print(
        f"Images used          : {len(data)}"
    )

    print(
        f"Features ranked      : "
        f"{len([c for c in data.columns if c not in {'image_name', 'ssim', 'psnr'}])}"
    )

    print(
        f"SSIM held-out R²     : {ssim_r2:.4f}"
    )

    print(
        f"PSNR held-out R²     : {psnr_r2:.4f}"
    )

    print()
    print(
        f"Results folder:\n{OUTPUT_DIR}"
    )

    print()
    print("Generated files:")

    print(
        "1. feature_quality_dataset.csv"
    )

    print(
        "2. feature_quality_dataset.xlsx"
    )

    print(
        "3. ranking_ssim.csv"
    )

    print(
        "4. ranking_ssim.xlsx"
    )

    print(
        "5. ranking_psnr.csv"
    )

    print(
        "6. ranking_psnr.xlsx"
    )

    print(
        "7. ranking_combined.csv"
    )

    print(
        "8. ranking_combined.xlsx"
    )

    print()
    print(
        "Next step: use the final ranking to select "
        "the most informative features before CNN modeling."
    )


# ============================================================
# 10. RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()
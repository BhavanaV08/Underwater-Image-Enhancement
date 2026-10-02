import pandas as pd
import numpy as np
from pathlib import Path
import cv2

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from skimage.metrics import (
    structural_similarity,
    peak_signal_noise_ratio
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

FEATURE_FILE = ROOT / "feature_dataset.csv"
RANKING_FILE = ROOT / "feature_ranking_results" / "ranking_combined.csv"

PREPROCESSED_DIR = ROOT / "dataset" / "preprocessed"
REFERENCE_DIR = ROOT / "dataset" / "reference-890"

OUTPUT_DIR = ROOT / "feature_ranking_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("FEATURE SUBSET EVALUATION")
print("=" * 65)

df = pd.read_csv(FEATURE_FILE)
ranking_df = pd.read_csv(RANKING_FILE)

print(f"\nFeature dataset shape : {df.shape}")
print(f"Ranking dataset shape : {ranking_df.shape}")


# ============================================================
# 15 FEATURES AFTER CORRELATION ANALYSIS
# ============================================================

selected_features = [
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
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in selected_features
    if feature not in df.columns
]

if missing_features:

    print("\nERROR: Missing features:")

    for feature in missing_features:
        print(" -", feature)

    raise ValueError("Some selected features are missing.")


# ============================================================
# CALCULATE SSIM AND PSNR
# ============================================================

print("\n" + "=" * 65)
print("CALCULATING SSIM AND PSNR")
print("=" * 65)

ssim_values = []
psnr_values = []
valid_images = []
removed_images = []


for index, row in df.iterrows():

    image_name = row["image_name"]

    preprocessed_path = PREPROCESSED_DIR / image_name
    reference_path = REFERENCE_DIR / image_name

    # Check both images exist
    if not preprocessed_path.exists():
        removed_images.append(
            (image_name, "preprocessed image missing")
        )
        continue

    if not reference_path.exists():
        removed_images.append(
            (image_name, "reference image missing")
        )
        continue


    # Read images
    preprocessed = cv2.imread(
        str(preprocessed_path)
    )

    reference = cv2.imread(
        str(reference_path)
    )


    if preprocessed is None or reference is None:

        removed_images.append(
            (image_name, "image could not be read")
        )

        continue


    # Convert BGR → RGB
    preprocessed = cv2.cvtColor(
        preprocessed,
        cv2.COLOR_BGR2RGB
    )

    reference = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2RGB
    )


    # Make sure dimensions match
    if preprocessed.shape != reference.shape:

        reference = cv2.resize(
            reference,
            (
                preprocessed.shape[1],
                preprocessed.shape[0]
            )
        )


    # --------------------------------------------------------
    # SSIM
    # --------------------------------------------------------

    ssim_score = structural_similarity(
        preprocessed,
        reference,
        channel_axis=2,
        data_range=255
    )


    # --------------------------------------------------------
    # PSNR
    # --------------------------------------------------------

    psnr_score = peak_signal_noise_ratio(
        reference,
        preprocessed,
        data_range=255
    )


    ssim_values.append(ssim_score)
    psnr_values.append(psnr_score)
    valid_images.append(index)


    # Progress
    if len(valid_images) % 100 == 0:

        print(
            f"Processed {len(valid_images)} images..."
        )


print(
    f"\nImages with valid metrics : "
    f"{len(valid_images)}"
)

print(
    f"Images removed            : "
    f"{len(removed_images)}"
)


# ============================================================
# CREATE DATASET WITH METRICS
# ============================================================

metric_df = df.loc[
    valid_images,
    selected_features + ["image_name"]
].copy()

metric_df["ssim"] = ssim_values
metric_df["psnr"] = psnr_values

metric_df = metric_df.reset_index(drop=True)


print(
    f"\nFinal evaluation dataset : "
    f"{metric_df.shape}"
)


# ============================================================
# ORDER FEATURES USING COMBINED IMPORTANCE
# ============================================================

ranking_df = ranking_df.sort_values(
    by="combined_importance",
    ascending=False
).reset_index(drop=True)


ranking_df = ranking_df[
    ranking_df["feature"].isin(selected_features)
].copy()


ranking_df = ranking_df.sort_values(
    by="combined_importance",
    ascending=False
).reset_index(drop=True)


ordered_features = ranking_df[
    "feature"
].tolist()


# ============================================================
# PRINT FEATURE ORDER
# ============================================================

print("\n" + "=" * 65)
print("FEATURE ORDER AFTER RANKING + CORRELATION ANALYSIS")
print("=" * 65)

for i, feature in enumerate(
    ordered_features,
    start=1
):

    importance = ranking_df.loc[
        ranking_df["feature"] == feature,
        "combined_importance"
    ].iloc[0]

    print(
        f"{i:2d}. {feature:22s} "
        f"importance = {importance:.6f}"
    )


# ============================================================
# FEATURE SUBSET SIZES
# ============================================================

subset_sizes = [
    15,
    12,
    10,
    8,
    6
]


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# EVALUATE FEATURE SUBSETS
# ============================================================

for subset_size in subset_sizes:

    print("\n" + "=" * 65)
    print(
        f"EVALUATING TOP {subset_size} FEATURES"
    )
    print("=" * 65)


    current_features = ordered_features[
        :subset_size
    ]


    print("\nFeatures used:")

    for i, feature in enumerate(
        current_features,
        start=1
    ):

        print(
            f"{i:2d}. {feature}"
        )


    X = metric_df[
        current_features
    ]


    # ========================================================
    # SSIM MODEL
    # ========================================================

    y_ssim = metric_df["ssim"]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_ssim,
        test_size=0.20,
        random_state=42
    )


    ssim_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2
    )


    ssim_model.fit(
        X_train,
        y_train
    )


    ssim_prediction = ssim_model.predict(
        X_test
    )


    ssim_r2 = r2_score(
        y_test,
        ssim_prediction
    )


    ssim_mse = mean_squared_error(
        y_test,
        ssim_prediction
    )


    ssim_rmse = np.sqrt(
        ssim_mse
    )


    ssim_mae = mean_absolute_error(
        y_test,
        ssim_prediction
    )


    # ========================================================
    # PSNR MODEL
    # ========================================================

    y_psnr = metric_df["psnr"]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_psnr,
        test_size=0.20,
        random_state=42
    )


    psnr_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2
    )


    psnr_model.fit(
        X_train,
        y_train
    )


    psnr_prediction = psnr_model.predict(
        X_test
    )


    psnr_r2 = r2_score(
        y_test,
        psnr_prediction
    )


    psnr_mse = mean_squared_error(
        y_test,
        psnr_prediction
    )


    psnr_rmse = np.sqrt(
        psnr_mse
    )


    psnr_mae = mean_absolute_error(
        y_test,
        psnr_prediction
    )


    # ========================================================
    # COMBINED SCORE
    # ========================================================

    combined_r2 = (
        ssim_r2 + psnr_r2
    ) / 2


    # ========================================================
    # STORE
    # ========================================================

    results.append({

        "feature_count": subset_size,

        "ssim_r2": ssim_r2,
        "ssim_rmse": ssim_rmse,
        "ssim_mae": ssim_mae,

        "psnr_r2": psnr_r2,
        "psnr_rmse": psnr_rmse,
        "psnr_mae": psnr_mae,

        "combined_r2": combined_r2,

        "features": ", ".join(
            current_features
        )
    })


    # ========================================================
    # PRINT
    # ========================================================

    print("\nSSIM Performance")

    print(
        f"R²   : {ssim_r2:.4f}"
    )

    print(
        f"RMSE : {ssim_rmse:.6f}"
    )

    print(
        f"MAE  : {ssim_mae:.6f}"
    )


    print("\nPSNR Performance")

    print(
        f"R²   : {psnr_r2:.4f}"
    )

    print(
        f"RMSE : {psnr_rmse:.6f}"
    )

    print(
        f"MAE  : {psnr_mae:.6f}"
    )


    print(
        f"\nCombined R² : "
        f"{combined_r2:.4f}"
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_DIR /
    "feature_subset_evaluation.csv",
    index=False
)

results_df.to_excel(
    OUTPUT_DIR /
    "feature_subset_evaluation.xlsx",
    index=False
)


# ============================================================
# BEST FEATURE SUBSET
# ============================================================

best_result = results_df.loc[
    results_df["combined_r2"].idxmax()
]


best_feature_count = int(
    best_result["feature_count"]
)


best_features = ordered_features[
    :best_feature_count
]


# ============================================================
# PRINT COMPARISON
# ============================================================

print("\n" + "=" * 65)
print("FEATURE SUBSET COMPARISON")
print("=" * 65)

print(
    results_df[
        [
            "feature_count",
            "ssim_r2",
            "psnr_r2",
            "combined_r2"
        ]
    ].to_string(index=False)
)


# ============================================================
# PRINT BEST
# ============================================================

print("\n" + "=" * 65)
print("BEST FEATURE SUBSET")
print("=" * 65)

print(
    f"\nBest number of features : "
    f"{best_feature_count}"
)

print(
    f"SSIM R²                 : "
    f"{best_result['ssim_r2']:.4f}"
)

print(
    f"PSNR R²                 : "
    f"{best_result['psnr_r2']:.4f}"
)

print(
    f"Combined R²             : "
    f"{best_result['combined_r2']:.4f}"
)


print("\nBest feature set:")

for i, feature in enumerate(
    best_features,
    start=1
):

    print(
        f"{i:2d}. {feature}"
    )


# ============================================================
# SAVE FINAL FEATURE SET
# ============================================================

best_features_df = pd.DataFrame({

    "final_rank": range(
        1,
        len(best_features) + 1
    ),

    "feature": best_features
})


best_features_df.to_csv(
    OUTPUT_DIR /
    "final_selected_features.csv",
    index=False
)

best_features_df.to_excel(
    OUTPUT_DIR /
    "final_selected_features.xlsx",
    index=False
)


# ============================================================
# SAVE METRIC DATASET
# ============================================================

metric_df.to_csv(
    OUTPUT_DIR /
    "feature_dataset_with_metrics.csv",
    index=False
)

metric_df.to_excel(
    OUTPUT_DIR /
    "feature_dataset_with_metrics.xlsx",
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 65)
print("FILES SAVED")
print("=" * 65)

print(
    "\nfeature_subset_evaluation.csv"
)

print(
    "feature_subset_evaluation.xlsx"
)

print(
    "final_selected_features.csv"
)

print(
    "final_selected_features.xlsx"
)

print(
    "feature_dataset_with_metrics.csv"
)

print(
    "feature_dataset_with_metrics.xlsx"
)

print("\nEvaluation completed successfully.")
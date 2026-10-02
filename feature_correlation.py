import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

FEATURE_FILE = ROOT / "feature_dataset.csv"

OUTPUT_DIR = ROOT / "feature_ranking_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD FEATURE DATASET
# ============================================================

df = pd.read_csv(FEATURE_FILE)

print("=" * 60)
print("FEATURE CORRELATION ANALYSIS")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# SELECT ONLY THE 25 FEATURES
# ============================================================

feature_columns = [
    "mean",
    "std",
    "variance",
    "entropy",
    "dynamic_range",
    "rms_contrast",

    "mean_red",
    "mean_green",
    "mean_blue",
    "colorfulness",
    "red_ratio",
    "mean_saturation",
    "mean_value",

    "contrast",
    "correlation",
    "energy",
    "homogeneity",
    "ASM",
    "dissimilarity",
    "glcm_entropy",
    "glcm_variance",

    "edge_density",
    "gradient",
    "laplacian_variance",
    "keypoint_density"
]

features = df[feature_columns]


# ============================================================
# PEARSON CORRELATION MATRIX
# ============================================================

correlation_matrix = features.corr(method="pearson")


# Save correlation matrix
correlation_matrix.to_csv(
    OUTPUT_DIR / "feature_correlation_matrix.csv"
)

correlation_matrix.to_excel(
    OUTPUT_DIR / "feature_correlation_matrix.xlsx"
)

print("\nCorrelation matrix saved.")


# ============================================================
# FIND HIGHLY CORRELATED FEATURE PAIRS
# ============================================================

threshold = 0.90

high_corr_pairs = []

for i in range(len(feature_columns)):
    for j in range(i + 1, len(feature_columns)):

        feature1 = feature_columns[i]
        feature2 = feature_columns[j]

        corr_value = correlation_matrix.loc[
            feature1, feature2
        ]

        if abs(corr_value) >= threshold:

            high_corr_pairs.append({
                "feature_1": feature1,
                "feature_2": feature2,
                "correlation": corr_value,
                "absolute_correlation": abs(corr_value)
            })


high_corr_df = pd.DataFrame(high_corr_pairs)

if not high_corr_df.empty:

    high_corr_df = high_corr_df.sort_values(
        by="absolute_correlation",
        ascending=False
    )

else:

    high_corr_df = pd.DataFrame(
        columns=[
            "feature_1",
            "feature_2",
            "correlation",
            "absolute_correlation"
        ]
    )


# Save highly correlated pairs
high_corr_df.to_csv(
    OUTPUT_DIR / "highly_correlated_features.csv",
    index=False
)

high_corr_df.to_excel(
    OUTPUT_DIR / "highly_correlated_features.xlsx",
    index=False
)


# ============================================================
# PRINT HIGHLY CORRELATED PAIRS
# ============================================================

print("\n" + "=" * 60)
print("HIGHLY CORRELATED FEATURE PAIRS")
print("=" * 60)

if high_corr_df.empty:

    print("\nNo feature pairs found above the threshold.")

else:

    for _, row in high_corr_df.iterrows():

        print(
            f"{row['feature_1']:20s} <-> "
            f"{row['feature_2']:20s} : "
            f"{row['correlation']:.4f}"
        )


# ============================================================
# REDUNDANCY-AWARE FEATURE SELECTION
# ============================================================

# Load combined feature ranking from previous step
ranking_file = OUTPUT_DIR / "ranking_combined.csv"

ranking_df = pd.read_csv(ranking_file)


# Sort by combined importance
ranking_df = ranking_df.sort_values(
    by="combined_importance",
    ascending=False
).reset_index(drop=True)


selected_features = []
removed_features = []

for feature in ranking_df["feature"]:

    keep = True

    for selected in selected_features:

        corr_value = correlation_matrix.loc[
            feature, selected
        ]

        if abs(corr_value) >= threshold:

            keep = False

            removed_features.append({
                "removed_feature": feature,
                "kept_feature": selected,
                "correlation": corr_value
            })

            break

    if keep:
        selected_features.append(feature)


# ============================================================
# SAVE SELECTED FEATURES
# ============================================================

selected_df = pd.DataFrame({
    "rank_after_redundancy_removal": range(
        1, len(selected_features) + 1
    ),
    "selected_feature": selected_features
})

selected_df.to_csv(
    OUTPUT_DIR / "selected_features.csv",
    index=False
)

selected_df.to_excel(
    OUTPUT_DIR / "selected_features.xlsx",
    index=False
)


# ============================================================
# SAVE REMOVED FEATURES
# ============================================================

removed_df = pd.DataFrame(removed_features)

removed_df.to_csv(
    OUTPUT_DIR / "removed_redundant_features.csv",
    index=False
)

removed_df.to_excel(
    OUTPUT_DIR / "removed_redundant_features.xlsx",
    index=False
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL FEATURE SELECTION")
print("=" * 60)

print(f"\nOriginal number of features : {len(feature_columns)}")
print(f"Selected features           : {len(selected_features)}")
print(
    f"Removed redundant features  : "
    f"{len(feature_columns) - len(selected_features)}"
)

print("\nSelected features:")

for i, feature in enumerate(selected_features, start=1):

    print(f"{i:2d}. {feature}")


print("\nRemoved redundant features:")

if removed_df.empty:

    print("None")

else:

    for _, row in removed_df.iterrows():

        print(
            f"{row['removed_feature']} "
            f"-> kept {row['kept_feature']} "
            f"(correlation = {row['correlation']:.4f})"
        )


# ============================================================
# CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(16, 14))

plt.imshow(
    correlation_matrix,
    interpolation="nearest",
    aspect="auto"
)

plt.colorbar(label="Pearson Correlation")

plt.xticks(
    range(len(feature_columns)),
    feature_columns,
    rotation=90
)

plt.yticks(
    range(len(feature_columns)),
    feature_columns
)

plt.title("Feature Correlation Matrix")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nCorrelation heatmap saved.")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)
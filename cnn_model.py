from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

FEATURE_CSV = ROOT / "feature_dataset.csv"

IMAGE_DIR = ROOT / "dataset" / "preprocessed"
REFERENCE_DIR = ROOT / "dataset" / "reference-890"

MODEL_DIR = ROOT / "cnn_results"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "cnn_quality_model.keras"

# Save target normalization statistics
TARGET_STATS_PATH = MODEL_DIR / "target_normalization_stats.npz"

IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 100
RANDOM_STATE = 42


# Reproducibility
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# 2. FINAL 8 SELECTED FEATURES
# ============================================================

SELECTED_FEATURES = [
    "edge_density",
    "energy",
    "std",
    "red_ratio",
    "entropy",
    "mean_blue",
    "keypoint_density",
    "mean_green"
]


# ============================================================
# 3. LOAD FEATURE DATASET
# ============================================================

print("=" * 70)
print("LOADING FEATURE DATASET")
print("=" * 70)

df = pd.read_csv(FEATURE_CSV)

print("Dataset shape:", df.shape)
print("Number of images:", len(df))


# ------------------------------------------------------------
# Check required feature columns
# ------------------------------------------------------------

required_columns = [
    "image_name"
] + SELECTED_FEATURES


missing_columns = [
    col for col in required_columns
    if col not in df.columns
]


if missing_columns:

    raise ValueError(
        f"Missing columns in feature dataset: "
        f"{missing_columns}"
    )


# Remove rows with missing feature values

df = df.dropna(
    subset=required_columns
).reset_index(drop=True)


print(
    "Dataset after removing missing values:",
    len(df)
)


# ============================================================
# 4. CALCULATE SSIM AND PSNR
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING SSIM AND PSNR")
print("=" * 70)


ssim_values = []
psnr_values = []

valid_rows = []


for index, row in df.iterrows():

    image_name = row["image_name"]

    preprocessed_path = (
        IMAGE_DIR / image_name
    )

    reference_path = (
        REFERENCE_DIR / image_name
    )


    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    processed = cv2.imread(
        str(preprocessed_path)
    )

    reference = cv2.imread(
        str(reference_path)
    )


    # --------------------------------------------------------
    # Check whether images exist
    # --------------------------------------------------------

    if processed is None:

        print(
            "WARNING: Preprocessed image not found:",
            image_name
        )

        continue


    if reference is None:

        print(
            "WARNING: Reference image not found:",
            image_name
        )

        continue


    # --------------------------------------------------------
    # Convert BGR -> RGB
    # --------------------------------------------------------

    processed = cv2.cvtColor(
        processed,
        cv2.COLOR_BGR2RGB
    )

    reference = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Resize reference if necessary
    # --------------------------------------------------------

    if reference.shape[:2] != (
        IMG_SIZE,
        IMG_SIZE
    ):

        reference = cv2.resize(
            reference,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA
        )


    # --------------------------------------------------------
    # Ensure processed image is 256x256
    # --------------------------------------------------------

    if processed.shape[:2] != (
        IMG_SIZE,
        IMG_SIZE
    ):

        processed = cv2.resize(
            processed,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA
        )


    # --------------------------------------------------------
    # Calculate SSIM
    # --------------------------------------------------------

    ssim = structural_similarity(
        processed,
        reference,
        channel_axis=2,
        data_range=255
    )


    # --------------------------------------------------------
    # Calculate PSNR
    # --------------------------------------------------------

    psnr = peak_signal_noise_ratio(
        processed,
        reference,
        data_range=255
    )


    ssim_values.append(ssim)
    psnr_values.append(psnr)

    valid_rows.append(index)


# ============================================================
# 5. KEEP ONLY VALID IMAGE PAIRS
# ============================================================

df = df.iloc[
    valid_rows
].reset_index(drop=True)


df["SSIM"] = np.array(
    ssim_values,
    dtype=np.float32
)

df["PSNR"] = np.array(
    psnr_values,
    dtype=np.float32
)


print(
    "\nSuccessfully calculated metrics for:",
    len(df),
    "images"
)


print("\nSSIM statistics:")
print(
    df["SSIM"].describe()
)


print("\nPSNR statistics:")
print(
    df["PSNR"].describe()
)


# ============================================================
# 6. SAVE COMPLETE CNN DATASET
# ============================================================

cnn_dataset_path = (
    MODEL_DIR / "cnn_dataset.csv"
)


df.to_csv(
    cnn_dataset_path,
    index=False
)


print(
    "\nCNN dataset saved to:"
)

print(cnn_dataset_path)


# ============================================================
# 7. LOAD PREPROCESSED IMAGES
# ============================================================

print("\n" + "=" * 70)
print("LOADING PREPROCESSED IMAGES")
print("=" * 70)


images = []
valid_rows = []


for index, row in df.iterrows():

    image_name = row["image_name"]

    image_path = (
        IMAGE_DIR / image_name
    )


    image = cv2.imread(
        str(image_path)
    )


    if image is None:

        print(
            "WARNING: Could not load:",
            image_name
        )

        continue


    # --------------------------------------------------------
    # Convert BGR -> RGB
    # --------------------------------------------------------

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Resize if necessary
    # --------------------------------------------------------

    if image.shape[:2] != (
        IMG_SIZE,
        IMG_SIZE
    ):

        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA
        )


    # --------------------------------------------------------
    # Normalize image to [0, 1]
    # --------------------------------------------------------

    image = (
        image.astype(np.float32)
        / 255.0
    )


    images.append(image)

    valid_rows.append(index)


# Keep only rows corresponding to successfully loaded images

df = df.iloc[
    valid_rows
].reset_index(drop=True)


images = np.array(
    images,
    dtype=np.float32
)


print(
    "Images loaded:",
    len(images)
)

print(
    "Image shape:",
    images.shape
)


# ============================================================
# 8. PREPARE FINAL 8 HANDCRAFTED FEATURES
# ============================================================

print("\n" + "=" * 70)
print("PREPARING FINAL 8 FEATURES")
print("=" * 70)


X_features = df[
    SELECTED_FEATURES
].values.astype(
    np.float32
)


print(
    "Feature matrix shape:",
    X_features.shape
)


for i, feature in enumerate(
    SELECTED_FEATURES,
    start=1
):

    print(
        f"{i}. {feature}"
    )


# ============================================================
# 9. TARGETS
# ============================================================

y_ssim = df[
    "SSIM"
].values.astype(
    np.float32
)


y_psnr = df[
    "PSNR"
].values.astype(
    np.float32
)


# ============================================================
# 10. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CREATING DATA SPLITS")
print("=" * 70)


indices = np.arange(
    len(df)
)


# ------------------------------------------------------------
# 70% training
# 30% temporary
# ------------------------------------------------------------

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=RANDOM_STATE
)


# ------------------------------------------------------------
# 15% validation
# 15% test
# ------------------------------------------------------------

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=RANDOM_STATE
)


print(
    "Training images   :",
    len(train_idx)
)

print(
    "Validation images :",
    len(val_idx)
)

print(
    "Testing images    :",
    len(test_idx)
)


# ============================================================
# 11. IMAGE SPLIT
# ============================================================

X_img_train = images[
    train_idx
]

X_img_val = images[
    val_idx
]

X_img_test = images[
    test_idx
]


# ============================================================
# 12. FEATURE SPLIT
# ============================================================

X_feat_train = X_features[
    train_idx
]

X_feat_val = X_features[
    val_idx
]

X_feat_test = X_features[
    test_idx
]


# ============================================================
# 13. TARGET SPLIT
# ============================================================

y_ssim_train = y_ssim[
    train_idx
]

y_ssim_val = y_ssim[
    val_idx
]

y_ssim_test = y_ssim[
    test_idx
]


y_psnr_train = y_psnr[
    train_idx
]

y_psnr_val = y_psnr[
    val_idx
]

y_psnr_test = y_psnr[
    test_idx
]


# ============================================================
# 14. STANDARDIZE HANDCRAFTED FEATURES
# ============================================================

print("\n" + "=" * 70)
print("STANDARDIZING HANDCRAFTED FEATURES")
print("=" * 70)


feature_scaler = StandardScaler()


# IMPORTANT:
# Fit only on training data

X_feat_train = feature_scaler.fit_transform(
    X_feat_train
).astype(
    np.float32
)


X_feat_val = feature_scaler.transform(
    X_feat_val
).astype(
    np.float32
)


X_feat_test = feature_scaler.transform(
    X_feat_test
).astype(
    np.float32
)


print(
    "Feature standardization completed."
)


# ============================================================
# 15. NORMALIZE SSIM AND PSNR TARGETS
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZING SSIM AND PSNR TARGETS")
print("=" * 70)


# IMPORTANT:
# Target statistics are calculated ONLY from training data.
#
# This prevents information from the validation/test sets
# leaking into the training process.


ssim_mean = np.mean(
    y_ssim_train
)

ssim_std = np.std(
    y_ssim_train
)


psnr_mean = np.mean(
    y_psnr_train
)

psnr_std = np.std(
    y_psnr_train
)


# Safety check

if ssim_std == 0:
    ssim_std = 1.0


if psnr_std == 0:
    psnr_std = 1.0


# ------------------------------------------------------------
# Standardized training targets
# ------------------------------------------------------------

y_ssim_train_scaled = (
    (y_ssim_train - ssim_mean)
    / ssim_std
).astype(
    np.float32
)


y_ssim_val_scaled = (
    (y_ssim_val - ssim_mean)
    / ssim_std
).astype(
    np.float32
)


y_ssim_test_scaled = (
    (y_ssim_test - ssim_mean)
    / ssim_std
).astype(
    np.float32
)


y_psnr_train_scaled = (
    (y_psnr_train - psnr_mean)
    / psnr_std
).astype(
    np.float32
)


y_psnr_val_scaled = (
    (y_psnr_val - psnr_mean)
    / psnr_std
).astype(
    np.float32
)


y_psnr_test_scaled = (
    (y_psnr_test - psnr_mean)
    / psnr_std
).astype(
    np.float32
)


print("\nSSIM target normalization:")
print(
    f"Training mean : {ssim_mean:.6f}"
)

print(
    f"Training std  : {ssim_std:.6f}"
)


print("\nPSNR target normalization:")
print(
    f"Training mean : {psnr_mean:.6f}"
)

print(
    f"Training std  : {psnr_std:.6f}"
)


# ------------------------------------------------------------
# Save target normalization statistics
# ------------------------------------------------------------

np.savez(
    TARGET_STATS_PATH,
    ssim_mean=ssim_mean,
    ssim_std=ssim_std,
    psnr_mean=psnr_mean,
    psnr_std=psnr_std
)


print(
    "\nTarget normalization statistics saved to:"
)

print(TARGET_STATS_PATH)


# ============================================================
# 16. BUILD DATA AUGMENTATION
# ============================================================

print("\n" + "=" * 70)
print("BUILDING DATA AUGMENTATION")
print("=" * 70)


# Moderate geometric augmentation.
#
# We intentionally avoid strong color augmentation because
# underwater color information is important to this project.


data_augmentation = tf.keras.Sequential(
    [

        tf.keras.layers.RandomFlip(
            mode="horizontal"
        ),

        tf.keras.layers.RandomRotation(
            factor=0.05
        ),

        tf.keras.layers.RandomZoom(
            height_factor=0.10,
            width_factor=0.10
        ),

        tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        )

    ],
    name="data_augmentation"
)


print(
    "Data augmentation enabled."
)

print(
    "- Horizontal flip"
)

print(
    "- Small rotation"
)

print(
    "- Small zoom"
)

print(
    "- Small translation"
)


# ============================================================
# 17. BUILD CNN + FEATURE FUSION MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING CNN + FEATURE FUSION MODEL")
print("=" * 70)


# ------------------------------------------------------------
# IMAGE INPUT
# ------------------------------------------------------------

image_input = tf.keras.Input(
    shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    ),
    name="image_input"
)


# ------------------------------------------------------------
# DATA AUGMENTATION
# ------------------------------------------------------------

x = data_augmentation(
    image_input
)


# ============================================================
# CNN BLOCK 1
# ============================================================

x = tf.keras.layers.Conv2D(
    32,
    (3, 3),
    padding="same"
)(x)


x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.MaxPooling2D(
    (2, 2)
)(x)


# ============================================================
# CNN BLOCK 2
# ============================================================

x = tf.keras.layers.Conv2D(
    64,
    (3, 3),
    padding="same"
)(x)


x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.MaxPooling2D(
    (2, 2)
)(x)


x = tf.keras.layers.Dropout(
    0.20
)(x)


# ============================================================
# CNN BLOCK 3
# ============================================================

x = tf.keras.layers.Conv2D(
    128,
    (3, 3),
    padding="same"
)(x)


x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.MaxPooling2D(
    (2, 2)
)(x)


x = tf.keras.layers.Dropout(
    0.25
)(x)


# ============================================================
# CNN BLOCK 4
# ============================================================

x = tf.keras.layers.Conv2D(
    256,
    (3, 3),
    padding="same"
)(x)


x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.ReLU()(x)


# ============================================================
# GLOBAL AVERAGE POOLING
# ============================================================

x = tf.keras.layers.GlobalAveragePooling2D()(
    x
)


# ============================================================
# CNN DENSE REPRESENTATION
# ============================================================

x = tf.keras.layers.Dense(
    128,
    activation="relu"
)(x)


x = tf.keras.layers.Dropout(
    0.30
)(x)


cnn_features = x


# ============================================================
# 18. HANDCRAFTED FEATURE BRANCH
# ============================================================

feature_input = tf.keras.Input(
    shape=(
        len(SELECTED_FEATURES),
    ),
    name="feature_input"
)


# ------------------------------------------------------------
# Dense layer 1
# ------------------------------------------------------------

f = tf.keras.layers.Dense(
    32,
    activation="relu"
)(feature_input)


f = tf.keras.layers.BatchNormalization()(
    f
)


f = tf.keras.layers.Dropout(
    0.20
)(f)


# ------------------------------------------------------------
# Dense layer 2
# ------------------------------------------------------------

f = tf.keras.layers.Dense(
    16,
    activation="relu"
)(f)


handcrafted_features = f


# ============================================================
# 19. FEATURE FUSION
# ============================================================

combined = tf.keras.layers.Concatenate(
    name="feature_fusion"
)([
    cnn_features,
    handcrafted_features
])


# ============================================================
# 20. COMMON DENSE LAYERS
# ============================================================

combined = tf.keras.layers.Dense(
    128,
    activation="relu"
)(combined)


combined = tf.keras.layers.BatchNormalization()(
    combined
)


combined = tf.keras.layers.Dropout(
    0.30
)(combined)


combined = tf.keras.layers.Dense(
    64,
    activation="relu"
)(combined)


# ============================================================
# 21. SSIM OUTPUT
# ============================================================

ssim_output = tf.keras.layers.Dense(
    1,
    activation="linear",
    name="ssim_output"
)(combined)


# ============================================================
# 22. PSNR OUTPUT
# ============================================================

psnr_output = tf.keras.layers.Dense(
    1,
    activation="linear",
    name="psnr_output"
)(combined)


# ============================================================
# 23. CREATE MODEL
# ============================================================

model = tf.keras.Model(
    inputs=[
        image_input,
        feature_input
    ],
    outputs=[
        ssim_output,
        psnr_output
    ]
)


# ============================================================
# 24. COMPILE MODEL
# ============================================================

print("\n" + "=" * 70)
print("COMPILING MODEL")
print("=" * 70)


model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    # Both targets are standardized,
    # therefore equal loss weights are appropriate.

    loss={
        "ssim_output": tf.keras.losses.Huber(),
        "psnr_output": tf.keras.losses.Huber()
    },

    loss_weights={
        "ssim_output": 1.0,
        "psnr_output": 1.0
    },

    metrics={

        "ssim_output": [
            tf.keras.metrics.MeanAbsoluteError(
                name="mae"
            )
        ],

        "psnr_output": [
            tf.keras.metrics.MeanAbsoluteError(
                name="mae"
            )
        ]

    }
)


# ============================================================
# 25. MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# 26. CALLBACKS
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=12,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=5,

    min_lr=1e-6,

    verbose=1
)


checkpoint = tf.keras.callbacks.ModelCheckpoint(

    filepath=str(MODEL_PATH),

    monitor="val_loss",

    save_best_only=True,

    verbose=1
)


# ============================================================
# 27. TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING CNN TRAINING")
print("=" * 70)


history = model.fit(

    {
        "image_input": X_img_train,
        "feature_input": X_feat_train
    },

    {
        # IMPORTANT:
        # Train using standardized targets

        "ssim_output": y_ssim_train_scaled,
        "psnr_output": y_psnr_train_scaled
    },

    validation_data=(

        {
            "image_input": X_img_val,
            "feature_input": X_feat_val
        },

        {
            "ssim_output": y_ssim_val_scaled,
            "psnr_output": y_psnr_val_scaled
        }

    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ],

    verbose=1
)


# ============================================================
# 28. SAVE MODEL
# ============================================================

model.save(
    MODEL_PATH
)


print(
    "\nModel saved to:"
)

print(
    MODEL_PATH
)


# ============================================================
# 29. TEST SET PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("TEST SET PREDICTION")
print("=" * 70)


predictions = model.predict(

    {
        "image_input": X_img_test,
        "feature_input": X_feat_test
    },

    verbose=1
)


# ------------------------------------------------------------
# Predictions are currently in standardized scale
# ------------------------------------------------------------

pred_ssim_scaled = predictions[
    0
].flatten()


pred_psnr_scaled = predictions[
    1
].flatten()


# ============================================================
# 30. INVERSE TRANSFORM PREDICTIONS
# ============================================================

# Convert predictions back to original SSIM scale

pred_ssim = (
    pred_ssim_scaled * ssim_std
    + ssim_mean
)


# Convert predictions back to original PSNR scale

pred_psnr = (
    pred_psnr_scaled * psnr_std
    + psnr_mean
)


pred_ssim = pred_ssim.astype(
    np.float32
)

pred_psnr = pred_psnr.astype(
    np.float32
)


# ============================================================
# 31. EVALUATION
# ============================================================

ssim_r2 = r2_score(
    y_ssim_test,
    pred_ssim
)


ssim_rmse = np.sqrt(
    mean_squared_error(
        y_ssim_test,
        pred_ssim
    )
)


ssim_mae = mean_absolute_error(
    y_ssim_test,
    pred_ssim
)


psnr_r2 = r2_score(
    y_psnr_test,
    pred_psnr
)


psnr_rmse = np.sqrt(
    mean_squared_error(
        y_psnr_test,
        pred_psnr
    )
)


psnr_mae = mean_absolute_error(
    y_psnr_test,
    pred_psnr
)


combined_r2 = (
    ssim_r2 +
    psnr_r2
) / 2


# ============================================================
# 32. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL CNN RESULTS")
print("=" * 70)


print("\nSSIM")
print("-" * 30)

print(
    f"R²   : {ssim_r2:.4f}"
)

print(
    f"RMSE : {ssim_rmse:.4f}"
)

print(
    f"MAE  : {ssim_mae:.4f}"
)


print("\nPSNR")
print("-" * 30)

print(
    f"R²   : {psnr_r2:.4f}"
)

print(
    f"RMSE : {psnr_rmse:.4f}"
)

print(
    f"MAE  : {psnr_mae:.4f}"
)


print("\nCombined R²")
print("-" * 30)

print(
    f"{combined_r2:.4f}"
)


# ============================================================
# 33. SAVE EVALUATION RESULTS
# ============================================================

results = pd.DataFrame({

    "Metric": [
        "SSIM",
        "PSNR",
        "Combined"
    ],

    "R2": [
        ssim_r2,
        psnr_r2,
        combined_r2
    ],

    "RMSE": [
        ssim_rmse,
        psnr_rmse,
        np.nan
    ],

    "MAE": [
        ssim_mae,
        psnr_mae,
        np.nan
    ]

})


results_path = (
    MODEL_DIR /
    "cnn_evaluation_results.csv"
)


results.to_csv(
    results_path,
    index=False
)


print(
    "\nEvaluation results saved to:"
)

print(
    results_path
)


# ============================================================
# 34. TRAINING AND VALIDATION LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    history.history["loss"],
    label="Training Loss"
)


plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Normalized Loss"
)

plt.title(
    "CNN Training and Validation Loss"
)


plt.legend()

plt.grid(True)

plt.tight_layout()


plt.savefig(
    MODEL_DIR /
    "training_validation_loss.png",
    dpi=300
)


plt.show()


# ============================================================
# 35. SSIM ACTUAL VS PREDICTED
# ============================================================

plt.figure(
    figsize=(7, 6)
)


plt.scatter(
    y_ssim_test,
    pred_ssim,
    alpha=0.7
)


min_value = min(
    np.min(y_ssim_test),
    np.min(pred_ssim)
)


max_value = max(
    np.max(y_ssim_test),
    np.max(pred_ssim)
)


plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)


plt.xlabel(
    "Actual SSIM"
)

plt.ylabel(
    "Predicted SSIM"
)


plt.title(
    f"Actual vs Predicted SSIM "
    f"(R² = {ssim_r2:.4f})"
)


plt.grid(True)

plt.tight_layout()


plt.savefig(
    MODEL_DIR /
    "ssim_actual_vs_predicted.png",
    dpi=300
)


plt.show()


# ============================================================
# 36. PSNR ACTUAL VS PREDICTED
# ============================================================

plt.figure(
    figsize=(7, 6)
)


plt.scatter(
    y_psnr_test,
    pred_psnr,
    alpha=0.7
)


min_value = min(
    np.min(y_psnr_test),
    np.min(pred_psnr)
)


max_value = max(
    np.max(y_psnr_test),
    np.max(pred_psnr)
)


plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)


plt.xlabel(
    "Actual PSNR"
)

plt.ylabel(
    "Predicted PSNR"
)


plt.title(
    f"Actual vs Predicted PSNR "
    f"(R² = {psnr_r2:.4f})"
)


plt.grid(True)

plt.tight_layout()


plt.savefig(
    MODEL_DIR /
    "psnr_actual_vs_predicted.png",
    dpi=300
)


plt.show()


# ============================================================
# 37. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CNN IMPLEMENTATION COMPLETED")
print("=" * 70)


print("\nOutput folder:")
print(
    MODEL_DIR
)


print("\nGenerated files:")

print(
    "- cnn_dataset.csv"
)

print(
    "- cnn_quality_model.keras"
)

print(
    "- target_normalization_stats.npz"
)

print(
    "- cnn_evaluation_results.csv"
)

print(
    "- training_validation_loss.png"
)

print(
    "- ssim_actual_vs_predicted.png"
)

print(
    "- psnr_actual_vs_predicted.png"
)


print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"SSIM R²     : {ssim_r2:.4f}"
)

print(
    f"SSIM RMSE   : {ssim_rmse:.4f}"
)

print(
    f"SSIM MAE    : {ssim_mae:.4f}"
)

print()

print(
    f"PSNR R²     : {psnr_r2:.4f}"
)

print(
    f"PSNR RMSE   : {psnr_rmse:.4f}"
)

print(
    f"PSNR MAE    : {psnr_mae:.4f}"
)

print()

print(
    f"Combined R² : {combined_r2:.4f}"
)

print("=" * 70)
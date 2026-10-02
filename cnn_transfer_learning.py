from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from skimage.metrics import structural_similarity, peak_signal_noise_ratio


# ============================================================
# TRANSFER LEARNING CNN + 8 HANDCRAFTED FEATURES
# Experiment 4
#
# Changes from the previous CNN:
#   1. MobileNetV2 pretrained on ImageNet
#   2. Target normalization for SSIM and PSNR
#   3. Moderate geometric augmentation
#   4. Huber loss
#   5. Equal loss weights
#   6. Feature fusion with the final 8 handcrafted features
#
# IMPORTANT:
# This model predicts image quality (SSIM and PSNR).
# It does NOT generate an enhanced image.
# ============================================================


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

FEATURE_CSV = ROOT / "feature_dataset.csv"
IMAGE_DIR = ROOT / "dataset" / "preprocessed"
REFERENCE_DIR = ROOT / "dataset" / "reference-890"

MODEL_DIR = ROOT / "cnn_results"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "cnn_transfer_learning_model.keras"

IMG_SIZE = 256
BATCH_SIZE = 16

# Stage 1: frozen MobileNetV2
FROZEN_EPOCHS = 30

# Stage 2: fine-tune the last part of MobileNetV2
FINETUNE_EPOCHS = 40

RANDOM_STATE = 42

# Number of MobileNetV2 layers to unfreeze during fine-tuning.
# BatchNormalization layers will remain frozen.
UNFREEZE_LAST_LAYERS = 30


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
# 3. REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# 4. LOAD FEATURE DATASET
# ============================================================

print("=" * 70)
print("TRANSFER LEARNING CNN")
print("=" * 70)

print("\n" + "=" * 70)
print("LOADING FEATURE DATASET")
print("=" * 70)

df = pd.read_csv(FEATURE_CSV)

print("Dataset shape:", df.shape)
print("Number of images:", len(df))

required_columns = ["image_name"] + SELECTED_FEATURES

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in feature dataset: {missing_columns}"
    )

df = df.dropna(subset=required_columns).reset_index(drop=True)

print("Dataset after removing missing values:", len(df))


# ============================================================
# 5. CALCULATE SSIM AND PSNR
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING SSIM AND PSNR")
print("=" * 70)

ssim_values = []
psnr_values = []
valid_rows = []

for index, row in df.iterrows():

    image_name = row["image_name"]

    processed_path = IMAGE_DIR / image_name
    reference_path = REFERENCE_DIR / image_name

    processed = cv2.imread(str(processed_path))
    reference = cv2.imread(str(reference_path))

    if processed is None:
        print("WARNING: Preprocessed image not found:", image_name)
        continue

    if reference is None:
        print("WARNING: Reference image not found:", image_name)
        continue

    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)

    if processed.shape[:2] != (IMG_SIZE, IMG_SIZE):
        processed = cv2.resize(
            processed,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA
        )

    if reference.shape[:2] != (IMG_SIZE, IMG_SIZE):
        reference = cv2.resize(
            reference,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA
        )

    ssim = structural_similarity(
        processed,
        reference,
        channel_axis=2,
        data_range=255
    )

    psnr = peak_signal_noise_ratio(
        processed,
        reference,
        data_range=255
    )

    ssim_values.append(ssim)
    psnr_values.append(psnr)
    valid_rows.append(index)


df = df.iloc[valid_rows].reset_index(drop=True)

df["SSIM"] = np.asarray(ssim_values, dtype=np.float32)
df["PSNR"] = np.asarray(psnr_values, dtype=np.float32)

print("\nSuccessfully calculated metrics for:", len(df), "images")

print("\nSSIM statistics:")
print(df["SSIM"].describe())

print("\nPSNR statistics:")
print(df["PSNR"].describe())


# ============================================================
# 6. SAVE DATASET USED BY THIS EXPERIMENT
# ============================================================

cnn_dataset_path = MODEL_DIR / "cnn_transfer_dataset.csv"

df.to_csv(cnn_dataset_path, index=False)

print("\nTransfer-learning dataset saved to:")
print(cnn_dataset_path)


# ============================================================
# 7. LOAD PREPROCESSED IMAGES
# ============================================================

print("\n" + "=" * 70)
print("LOADING PREPROCESSED IMAGES")
print("=" * 70)

images = []
image_valid_rows = []

for index, row in df.iterrows():

    image_name = row["image_name"]
    image_path = IMAGE_DIR / image_name

    image = cv2.imread(str(image_path))

    if image is None:
        print("WARNING: Could not load:", image_name)
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.shape[:2] != (IMG_SIZE, IMG_SIZE):
        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA
        )

    # Keep image in [0, 1].
    # MobileNetV2 preprocessing is applied inside the model.
    image = image.astype(np.float32) / 255.0

    images.append(image)
    image_valid_rows.append(index)


df = df.iloc[image_valid_rows].reset_index(drop=True)

images = np.asarray(images, dtype=np.float32)

print("Images loaded:", len(images))
print("Image shape:", images.shape)


# ============================================================
# 8. PREPARE FINAL 8 HANDCRAFTED FEATURES
# ============================================================

print("\n" + "=" * 70)
print("PREPARING FINAL 8 FEATURES")
print("=" * 70)

X_features = df[SELECTED_FEATURES].values.astype(np.float32)

print("Feature matrix shape:", X_features.shape)

for i, feature in enumerate(SELECTED_FEATURES, start=1):
    print(f"{i}. {feature}")


# ============================================================
# 9. TARGETS
# ============================================================

y_ssim = df["SSIM"].values.astype(np.float32)
y_psnr = df["PSNR"].values.astype(np.float32)


# ============================================================
# 10. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CREATING DATA SPLITS")
print("=" * 70)

indices = np.arange(len(df))

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=RANDOM_STATE
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=RANDOM_STATE
)

print("Training images   :", len(train_idx))
print("Validation images :", len(val_idx))
print("Testing images    :", len(test_idx))


# ============================================================
# 11. SPLIT IMAGES
# ============================================================

X_img_train = images[train_idx]
X_img_val = images[val_idx]
X_img_test = images[test_idx]


# ============================================================
# 12. SPLIT HANDCRAFTED FEATURES
# ============================================================

X_feat_train = X_features[train_idx]
X_feat_val = X_features[val_idx]
X_feat_test = X_features[test_idx]


# ============================================================
# 13. SPLIT TARGETS
# ============================================================

y_ssim_train = y_ssim[train_idx]
y_ssim_val = y_ssim[val_idx]
y_ssim_test = y_ssim[test_idx]

y_psnr_train = y_psnr[train_idx]
y_psnr_val = y_psnr[val_idx]
y_psnr_test = y_psnr[test_idx]


# ============================================================
# 14. STANDARDIZE HANDCRAFTED FEATURES
# ============================================================

print("\n" + "=" * 70)
print("STANDARDIZING HANDCRAFTED FEATURES")
print("=" * 70)

scaler = StandardScaler()

X_feat_train = scaler.fit_transform(X_feat_train).astype(np.float32)
X_feat_val = scaler.transform(X_feat_val).astype(np.float32)
X_feat_test = scaler.transform(X_feat_test).astype(np.float32)

np.savez(
    MODEL_DIR / "transfer_feature_scaler_stats.npz",
    mean=scaler.mean_,
    scale=scaler.scale_
)

print("Feature standardization completed.")


# ============================================================
# 15. NORMALIZE TARGETS
#
# Statistics are calculated ONLY from the training set.
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZING SSIM AND PSNR TARGETS")
print("=" * 70)

ssim_mean = float(np.mean(y_ssim_train))
ssim_std = float(np.std(y_ssim_train))

psnr_mean = float(np.mean(y_psnr_train))
psnr_std = float(np.std(y_psnr_train))

if ssim_std < 1e-8:
    ssim_std = 1.0

if psnr_std < 1e-8:
    psnr_std = 1.0

y_ssim_train_norm = (
    (y_ssim_train - ssim_mean) / ssim_std
).astype(np.float32)

y_ssim_val_norm = (
    (y_ssim_val - ssim_mean) / ssim_std
).astype(np.float32)

y_ssim_test_norm = (
    (y_ssim_test - ssim_mean) / ssim_std
).astype(np.float32)

y_psnr_train_norm = (
    (y_psnr_train - psnr_mean) / psnr_std
).astype(np.float32)

y_psnr_val_norm = (
    (y_psnr_val - psnr_mean) / psnr_std
).astype(np.float32)

y_psnr_test_norm = (
    (y_psnr_test - psnr_mean) / psnr_std
).astype(np.float32)

np.savez(
    MODEL_DIR / "transfer_target_normalization_stats.npz",
    ssim_mean=ssim_mean,
    ssim_std=ssim_std,
    psnr_mean=psnr_mean,
    psnr_std=psnr_std
)

print(f"SSIM mean = {ssim_mean:.6f}")
print(f"SSIM std  = {ssim_std:.6f}")
print(f"PSNR mean = {psnr_mean:.6f}")
print(f"PSNR std  = {psnr_std:.6f}")


# ============================================================
# 16. DATA AUGMENTATION
#
# Only geometric augmentation is used.
# No color augmentation because underwater color
# information is important for the selected features.
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.05
        ),

        tf.keras.layers.RandomZoom(
            0.10
        ),

        tf.keras.layers.RandomTranslation(
            0.05,
            0.05
        )
    ],
    name="geometric_augmentation"
)


# ============================================================
# 17. BUILD TRANSFER LEARNING MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING MOBILENETV2 TRANSFER-LEARNING MODEL")
print("=" * 70)

image_input = tf.keras.Input(
    shape=(IMG_SIZE, IMG_SIZE, 3),
    name="image_input"
)


# ------------------------------------------------------------
# IMAGE BRANCH
# ------------------------------------------------------------

augmented_image = data_augmentation(image_input)

# Input currently has values in [0, 1].
# MobileNetV2 preprocess_input expects values in [0, 255]
# and converts them approximately to [-1, 1].
mobilenet_input = tf.keras.layers.Lambda(
    lambda z: tf.keras.applications.mobilenet_v2.preprocess_input(
        z * 255.0
    ),
    name="mobilenet_preprocessing"
)(augmented_image)


base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Stage 1: freeze the entire pretrained backbone.
base_model.trainable = False

deep_features = base_model(
    mobilenet_input,
    training=False
)

deep_features = tf.keras.layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(deep_features)

deep_features = tf.keras.layers.Dense(
    128,
    activation="relu",
    name="deep_feature_dense"
)(deep_features)

deep_features = tf.keras.layers.Dropout(
    0.30
)(deep_features)


# ------------------------------------------------------------
# HANDCRAFTED FEATURE BRANCH
# ------------------------------------------------------------

feature_input = tf.keras.Input(
    shape=(len(SELECTED_FEATURES),),
    name="feature_input"
)

hand_features = tf.keras.layers.Dense(
    32,
    activation="relu"
)(feature_input)

hand_features = tf.keras.layers.BatchNormalization()(
    hand_features
)

hand_features = tf.keras.layers.Dropout(
    0.20
)(hand_features)

hand_features = tf.keras.layers.Dense(
    16,
    activation="relu"
)(hand_features)


# ------------------------------------------------------------
# FEATURE FUSION
# ------------------------------------------------------------

combined = tf.keras.layers.Concatenate(
    name="feature_fusion"
)([
    deep_features,
    hand_features
])

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


# ------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------

ssim_output = tf.keras.layers.Dense(
    1,
    activation="linear",
    name="ssim_output"
)(combined)

psnr_output = tf.keras.layers.Dense(
    1,
    activation="linear",
    name="psnr_output"
)(combined)


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
# 18. COMPILE - STAGE 1
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

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

model.summary()


# ============================================================
# 19. STAGE 1 CALLBACKS
# ============================================================

stage1_checkpoint = MODEL_DIR / "mobilenet_stage1_best.keras"

stage1_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(stage1_checkpoint),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]


# ============================================================
# 20. STAGE 1 TRAINING - FROZEN MOBILENETV2
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1: TRAINING WITH FROZEN MOBILENETV2")
print("=" * 70)

history_stage1 = model.fit(
    {
        "image_input": X_img_train,
        "feature_input": X_feat_train
    },

    {
        "ssim_output": y_ssim_train_norm,
        "psnr_output": y_psnr_train_norm
    },

    validation_data=(
        {
            "image_input": X_img_val,
            "feature_input": X_feat_val
        },

        {
            "ssim_output": y_ssim_val_norm,
            "psnr_output": y_psnr_val_norm
        }
    ),

    epochs=FROZEN_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=stage1_callbacks,
    verbose=1
)


# ============================================================
# 21. STAGE 2 - FINE-TUNING
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2: FINE-TUNING LAST MOBILENETV2 LAYERS")
print("=" * 70)

base_model.trainable = True

# First freeze every layer.
for layer in base_model.layers:
    layer.trainable = False

# Then unfreeze only the last N layers.
for layer in base_model.layers[-UNFREEZE_LAST_LAYERS:]:
    if not isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = True

trainable_count = sum(
    int(layer.trainable)
    for layer in base_model.layers
)

print(
    "Trainable MobileNetV2 layers:",
    trainable_count
)

print(
    "Total MobileNetV2 layers:",
    len(base_model.layers)
)


# Recompile with a much smaller learning rate.
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

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


stage2_checkpoint = MODEL_DIR / "mobilenet_stage2_best.keras"

stage2_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(stage2_checkpoint),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]


history_stage2 = model.fit(
    {
        "image_input": X_img_train,
        "feature_input": X_feat_train
    },

    {
        "ssim_output": y_ssim_train_norm,
        "psnr_output": y_psnr_train_norm
    },

    validation_data=(
        {
            "image_input": X_img_val,
            "feature_input": X_feat_val
        },

        {
            "ssim_output": y_ssim_val_norm,
            "psnr_output": y_psnr_val_norm
        }
    ),

    epochs=FINETUNE_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=stage2_callbacks,
    verbose=1
)


# ============================================================
# 22. SAVE FINAL MODEL
# ============================================================

model.save(MODEL_PATH)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(MODEL_PATH)


# ============================================================
# 23. TEST SET PREDICTION
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

pred_ssim_norm = predictions[0].reshape(-1)
pred_psnr_norm = predictions[1].reshape(-1)


# Convert predictions back to original units.
pred_ssim = (
    pred_ssim_norm * ssim_std + ssim_mean
)

pred_psnr = (
    pred_psnr_norm * psnr_std + psnr_mean
)


# ============================================================
# 24. CALCULATE FINAL METRICS
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
    ssim_r2 + psnr_r2
) / 2


# ============================================================
# 25. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TRANSFER-LEARNING CNN RESULTS")
print("=" * 70)

print("\nSSIM")
print("-" * 30)
print(f"R²   : {ssim_r2:.4f}")
print(f"RMSE : {ssim_rmse:.4f}")
print(f"MAE  : {ssim_mae:.4f}")

print("\nPSNR")
print("-" * 30)
print(f"R²   : {psnr_r2:.4f}")
print(f"RMSE : {psnr_rmse:.4f}")
print(f"MAE  : {psnr_mae:.4f}")

print("\nCombined R²")
print("-" * 30)
print(f"{combined_r2:.4f}")


# ============================================================
# 26. SAVE EVALUATION RESULTS
# ============================================================

evaluation_results = pd.DataFrame({
    "metric": [
        "SSIM_R2",
        "SSIM_RMSE",
        "SSIM_MAE",
        "PSNR_R2",
        "PSNR_RMSE",
        "PSNR_MAE",
        "Combined_R2"
    ],

    "value": [
        ssim_r2,
        ssim_rmse,
        ssim_mae,
        psnr_r2,
        psnr_rmse,
        psnr_mae,
        combined_r2
    ]
})

evaluation_path = (
    MODEL_DIR /
    "cnn_transfer_learning_evaluation_results.csv"
)

evaluation_results.to_csv(
    evaluation_path,
    index=False
)

print("\nEvaluation results saved to:")
print(evaluation_path)


# ============================================================
# 27. SAVE TEST PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "image_name": df.iloc[test_idx]["image_name"].values,
    "actual_ssim": y_ssim_test,
    "predicted_ssim": pred_ssim,
    "actual_psnr": y_psnr_test,
    "predicted_psnr": pred_psnr
})

prediction_path = (
    MODEL_DIR /
    "cnn_transfer_learning_test_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)

print("\nTest predictions saved to:")
print(prediction_path)


# ============================================================
# 28. COMBINE TRAINING HISTORIES
# ============================================================

stage1_loss = history_stage1.history.get(
    "loss",
    []
)

stage1_val_loss = history_stage1.history.get(
    "val_loss",
    []
)

stage2_loss = history_stage2.history.get(
    "loss",
    []
)

stage2_val_loss = history_stage2.history.get(
    "val_loss",
    []
)

all_loss = stage1_loss + stage2_loss
all_val_loss = stage1_val_loss + stage2_val_loss


# ============================================================
# 29. TRAINING / VALIDATION LOSS PLOT
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    all_loss,
    label="Training Loss"
)

plt.plot(
    all_val_loss,
    label="Validation Loss"
)

plt.axvline(
    len(stage1_loss) - 1,
    linestyle="--",
    label="Fine-tuning starts"
)

plt.xlabel("Epoch")
plt.ylabel("Huber Loss")
plt.title("MobileNetV2 Transfer-Learning Training")
plt.legend()
plt.tight_layout()

loss_plot_path = (
    MODEL_DIR /
    "cnn_transfer_training_validation_loss.png"
)

plt.savefig(
    loss_plot_path,
    dpi=300
)

plt.close()

print("\nLoss plot saved to:")
print(loss_plot_path)


# ============================================================
# 30. SSIM ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    y_ssim_test,
    pred_ssim,
    alpha=0.7
)

minimum = min(
    y_ssim_test.min(),
    pred_ssim.min()
)

maximum = max(
    y_ssim_test.max(),
    pred_ssim.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual SSIM")
plt.ylabel("Predicted SSIM")
plt.title(
    f"SSIM Actual vs Predicted (R² = {ssim_r2:.4f})"
)

plt.tight_layout()

ssim_plot_path = (
    MODEL_DIR /
    "cnn_transfer_ssim_actual_vs_predicted.png"
)

plt.savefig(
    ssim_plot_path,
    dpi=300
)

plt.close()

print("SSIM plot saved to:")
print(ssim_plot_path)


# ============================================================
# 31. PSNR ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    y_psnr_test,
    pred_psnr,
    alpha=0.7
)

minimum = min(
    y_psnr_test.min(),
    pred_psnr.min()
)

maximum = max(
    y_psnr_test.max(),
    pred_psnr.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual PSNR")
plt.ylabel("Predicted PSNR")
plt.title(
    f"PSNR Actual vs Predicted (R² = {psnr_r2:.4f})"
)

plt.tight_layout()

psnr_plot_path = (
    MODEL_DIR /
    "cnn_transfer_psnr_actual_vs_predicted.png"
)

plt.savefig(
    psnr_plot_path,
    dpi=300
)

plt.close()

print("PSNR plot saved to:")
print(psnr_plot_path)


# ============================================================
# 32. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRANSFER-LEARNING EXPERIMENT COMPLETED")
print("=" * 70)

print("\nFinal model:")
print(MODEL_PATH)

print("\nResults:")
print(f"SSIM R²     : {ssim_r2:.4f}")
print(f"PSNR R²     : {psnr_r2:.4f}")
print(f"Combined R² : {combined_r2:.4f}")

print("\nCompare this experiment with:")
print("Previous Huber CNN:")
print("  SSIM R²     = 0.7313")
print("  PSNR R²     = 0.5047")
print("  Combined R² = 0.6180")

print("\nRF 8-feature baseline:")
print("  SSIM R²     = 0.7827")
print("  PSNR R²     = 0.5168")
print("  Combined R² = 0.6498")

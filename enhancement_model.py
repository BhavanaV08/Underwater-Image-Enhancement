from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


# ============================================================
# FEATURE-GUIDED UNDERWATER IMAGE ENHANCEMENT
#
# Input  : RAW underwater image
# Target : Reference/enhanced image
# Guidance: 8 selected handcrafted features
# Output : Enhanced image
# ============================================================


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent

# IMPORTANT:
# The enhancement network now uses RAW images as input.
FEATURE_CSV = ROOT / "feature_dataset.csv"
IMAGE_DIR = ROOT / "dataset" / "raw-890"
REFERENCE_DIR = ROOT / "dataset" / "reference-890"

RESULT_DIR = ROOT / "enhancement_results"
RESULT_DIR.mkdir(exist_ok=True)

MODEL_PATH = RESULT_DIR / "feature_guided_enhancement_model.keras"

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 80
RANDOM_STATE = 42


# ============================================================
# 8 FINAL SELECTED FEATURES
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
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


print("=" * 70)
print("FEATURE-GUIDED UNDERWATER IMAGE ENHANCEMENT")
print("=" * 70)


# ============================================================
# 2. IMAGE PREPROCESSING
# ============================================================

def resize_with_reflect_padding(image, size=256):
    """
    Resize image while preserving aspect ratio.

    The longer dimension is resized to 'size'.
    The shorter dimension is padded using reflection.

    This avoids:
    - stretching
    - black bars
    - severe geometric distortion

    Output:
        size x size x 3
    """

    h, w = image.shape[:2]

    if h == 0 or w == 0:
        raise ValueError("Invalid image dimensions.")

    # Scale so that the longer side becomes 'size'
    scale = size / max(h, w)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    image = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Calculate required padding
    pad_h = size - new_h
    pad_w = size - new_w

    top = pad_h // 2
    bottom = pad_h - top

    left = pad_w // 2
    right = pad_w - left

    # Reflection padding instead of black padding
    #
    # If the image is very small, REFLECT_101 can fail.
    # In normal dataset images this should not happen, but
    # REPLICATE is used as a safe fallback.
    try:
        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_REFLECT_101
        )
    except cv2.error:
        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_REPLICATE
        )

    # Safety check
    if image.shape[:2] != (size, size):
        image = cv2.resize(
            image,
            (size, size),
            interpolation=cv2.INTER_AREA
        )

    return image


# ============================================================
# 3. LOAD DATA
# ============================================================

print()
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(FEATURE_CSV)

required = ["image_name"] + SELECTED_FEATURES

missing = [
    column for column in required
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns in feature_dataset.csv: {missing}"
    )

df = df.dropna(
    subset=required
).reset_index(drop=True)


images = []
references = []
features = []
names = []

aspect_ratio_differences = []


print("Loading raw image/reference pairs...")


for _, row in df.iterrows():

    name = row["image_name"]

    image_path = IMAGE_DIR / name
    reference_path = REFERENCE_DIR / name

    image = cv2.imread(str(image_path))
    reference = cv2.imread(str(reference_path))

    if image is None:
        print(f"WARNING: Could not read input image: {image_path}")
        continue

    if reference is None:
        print(f"WARNING: Could not read reference image: {reference_path}")
        continue


    # --------------------------------------------------------
    # Store original dimensions for checking
    # --------------------------------------------------------

    image_h, image_w = image.shape[:2]
    reference_h, reference_w = reference.shape[:2]

    image_ratio = image_w / image_h
    reference_ratio = reference_w / reference_h

    ratio_difference = abs(
        image_ratio - reference_ratio
    )

    aspect_ratio_differences.append(
        ratio_difference
    )


    # --------------------------------------------------------
    # Convert BGR -> RGB
    # --------------------------------------------------------

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    reference = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # SAME TYPE OF RESIZING FOR INPUT AND REFERENCE
    # --------------------------------------------------------

    image = resize_with_reflect_padding(
        image,
        IMG_SIZE
    )

    reference = resize_with_reflect_padding(
        reference,
        IMG_SIZE
    )


    # --------------------------------------------------------
    # Normalize to [0, 1]
    # --------------------------------------------------------

    image = image.astype(np.float32) / 255.0

    reference = reference.astype(np.float32) / 255.0


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    images.append(image)

    references.append(reference)

    features.append(
        row[SELECTED_FEATURES]
        .values
        .astype(np.float32)
    )

    names.append(name)


# Convert to NumPy arrays

images = np.asarray(
    images,
    dtype=np.float32
)

references = np.asarray(
    references,
    dtype=np.float32
)

features = np.asarray(
    features,
    dtype=np.float32
)

names = np.asarray(names)


# ============================================================
# DATASET INFORMATION
# ============================================================

print()
print("Images     :", images.shape)
print("References :", references.shape)
print("Features   :", features.shape)

print()

if len(aspect_ratio_differences) > 0:

    max_ratio_difference = max(
        aspect_ratio_differences
    )

    mean_ratio_difference = np.mean(
        aspect_ratio_differences
    )

    print(
        f"Maximum input/reference aspect-ratio difference : "
        f"{max_ratio_difference:.6f}"
    )

    print(
        f"Mean input/reference aspect-ratio difference    : "
        f"{mean_ratio_difference:.6f}"
    )

    if max_ratio_difference > 0.02:

        print()
        print(
            "WARNING:"
        )

        print(
            "Some input/reference image pairs have noticeably "
            "different aspect ratios."
        )

        print(
            "Please inspect the corresponding image pairs before "
            "using the final model."
        )


# ============================================================
# 4. SPLIT DATA
# ============================================================

print()
print("=" * 70)
print("SPLITTING DATA")
print("=" * 70)


indices = np.arange(
    len(images)
)


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
# CREATE SPLITS
# ============================================================

X_img_train = images[train_idx]
X_img_val = images[val_idx]
X_img_test = images[test_idx]


X_ref_train = references[train_idx]
X_ref_val = references[val_idx]
X_ref_test = references[test_idx]


X_feat_train = features[train_idx]
X_feat_val = features[val_idx]
X_feat_test = features[test_idx]


# ============================================================
# 5. STANDARDIZE HANDCRAFTED FEATURES
# ============================================================

print()
print("=" * 70)
print("STANDARDIZING HANDCRAFTED FEATURES")
print("=" * 70)


feature_mean = np.mean(
    X_feat_train,
    axis=0
)

feature_std = np.std(
    X_feat_train,
    axis=0
)

feature_std = np.where(
    feature_std < 1e-8,
    1.0,
    feature_std
)


X_feat_train = (
    (X_feat_train - feature_mean)
    / feature_std
).astype(np.float32)


X_feat_val = (
    (X_feat_val - feature_mean)
    / feature_std
).astype(np.float32)


X_feat_test = (
    (X_feat_test - feature_mean)
    / feature_std
).astype(np.float32)


# Save feature scaling statistics

np.savez(
    RESULT_DIR / "feature_scaler_stats.npz",
    mean=feature_mean,
    std=feature_std
)


print("Selected features:")

for feature in SELECTED_FEATURES:
    print("  -", feature)


# ============================================================
# 6. PAIRED AUGMENTATION
# ============================================================

print()
print("=" * 70)
print("SETTING UP PAIRED AUGMENTATION")
print("=" * 70)


def augment_pair(image, reference):
    """
    Apply exactly the same geometric transformation
    to both input and reference.
    """

    # Horizontal flip
    if tf.random.uniform(()) > 0.5:

        image = tf.image.flip_left_right(
            image
        )

        reference = tf.image.flip_left_right(
            reference
        )


    # Vertical flip
    if tf.random.uniform(()) > 0.5:

        image = tf.image.flip_up_down(
            image
        )

        reference = tf.image.flip_up_down(
            reference
        )


    # Random 90-degree rotation
    k = tf.random.uniform(
        (),
        minval=0,
        maxval=4,
        dtype=tf.int32
    )

    image = tf.image.rot90(
        image,
        k
    )

    reference = tf.image.rot90(
        reference,
        k
    )


    return image, reference


# ============================================================
# DATASET CREATION
# ============================================================

def make_dataset(
    image_data,
    feature_data,
    target_data,
    training=False
):

    ds = tf.data.Dataset.from_tensor_slices(
        (
            image_data,
            feature_data,
            target_data
        )
    )


    if training:

        ds = ds.shuffle(
            len(image_data),
            seed=RANDOM_STATE,
            reshuffle_each_iteration=True
        )


    def prepare(
        image,
        feature,
        target
    ):

        if training:

            image, target = augment_pair(
                image,
                target
            )


        inputs = {
            "image_input": image,
            "feature_input": feature
        }


        return inputs, target


    ds = ds.map(
        prepare,
        num_parallel_calls=tf.data.AUTOTUNE
    )


    ds = ds.batch(
        BATCH_SIZE
    )


    ds = ds.prefetch(
        tf.data.AUTOTUNE
    )


    return ds


# ============================================================
# CREATE TF DATASETS
# ============================================================

train_dataset = make_dataset(
    X_img_train,
    X_feat_train,
    X_ref_train,
    training=True
)


val_dataset = make_dataset(
    X_img_val,
    X_feat_val,
    X_ref_val,
    training=False
)


test_dataset = make_dataset(
    X_img_test,
    X_feat_test,
    X_ref_test,
    training=False
)


# ============================================================
# 7. BUILD FEATURE-GUIDED U-NET
# ============================================================

print()
print("=" * 70)
print("BUILDING FEATURE-GUIDED U-NET")
print("=" * 70)


# ============================================================
# IMAGE INPUT
# ============================================================

image_input = tf.keras.Input(
    shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    ),
    name="image_input"
)


# ============================================================
# ENCODER BLOCK 1
# ============================================================

x = tf.keras.layers.Conv2D(
    32,
    3,
    padding="same",
    activation="relu"
)(image_input)


x = tf.keras.layers.Conv2D(
    32,
    3,
    padding="same",
    activation="relu"
)(x)


skip1 = x


x = tf.keras.layers.MaxPooling2D(
    2
)(x)


# ============================================================
# ENCODER BLOCK 2
# ============================================================

x = tf.keras.layers.Conv2D(
    64,
    3,
    padding="same",
    activation="relu"
)(x)


x = tf.keras.layers.Conv2D(
    64,
    3,
    padding="same",
    activation="relu"
)(x)


skip2 = x


x = tf.keras.layers.MaxPooling2D(
    2
)(x)


# ============================================================
# BOTTLENECK
# ============================================================

x = tf.keras.layers.Conv2D(
    128,
    3,
    padding="same",
    activation="relu"
)(x)


x = tf.keras.layers.Conv2D(
    128,
    3,
    padding="same",
    activation="relu"
)(x)


# ============================================================
# HANDCRAFTED FEATURE BRANCH
# ============================================================

feature_input = tf.keras.Input(
    shape=(
        len(SELECTED_FEATURES),
    ),
    name="feature_input"
)


f = tf.keras.layers.Dense(
    32,
    activation="relu"
)(feature_input)


f = tf.keras.layers.BatchNormalization()(
    f
)


f = tf.keras.layers.Dense(
    64,
    activation="relu"
)(f)


# Convert feature vector to spatial guidance map

f = tf.keras.layers.Dense(
    64 * 64 * 16,
    activation="relu"
)(f)


f = tf.keras.layers.Reshape(
    (64, 64, 16)
)(f)


# ============================================================
# FEATURE FUSION
# ============================================================

x = tf.keras.layers.Concatenate(
    axis=-1,
    name="feature_guided_fusion"
)(
    [x, f]
)


x = tf.keras.layers.Conv2D(
    128,
    3,
    padding="same",
    activation="relu"
)(x)


# ============================================================
# DECODER BLOCK 1
# ============================================================

x = tf.keras.layers.UpSampling2D(
    2,
    interpolation="bilinear"
)(x)


x = tf.keras.layers.Concatenate()(
    [x, skip2]
)


x = tf.keras.layers.Conv2D(
    64,
    3,
    padding="same",
    activation="relu"
)(x)


x = tf.keras.layers.Conv2D(
    64,
    3,
    padding="same",
    activation="relu"
)(x)


# ============================================================
# DECODER BLOCK 2
# ============================================================

x = tf.keras.layers.UpSampling2D(
    2,
    interpolation="bilinear"
)(x)


x = tf.keras.layers.Concatenate()(
    [x, skip1]
)


x = tf.keras.layers.Conv2D(
    32,
    3,
    padding="same",
    activation="relu"
)(x)


x = tf.keras.layers.Conv2D(
    32,
    3,
    padding="same",
    activation="relu"
)(x)


# ============================================================
# OUTPUT
# ============================================================

output = tf.keras.layers.Conv2D(
    3,
    3,
    padding="same",
    activation="sigmoid",
    name="enhanced_image"
)(x)


# ============================================================
# CREATE MODEL
# ============================================================

model = tf.keras.Model(
    inputs=[
        image_input,
        feature_input
    ],
    outputs=output
)


# ============================================================
# 8. LOSS FUNCTION
# ============================================================

print()
print("=" * 70)
print("SETTING UP LOSS FUNCTION")
print("=" * 70)


def enhancement_loss(
    y_true,
    y_pred
):

    # Pixel-level L1 loss
    l1 = tf.reduce_mean(
        tf.abs(
            y_true - y_pred
        )
    )


    # Structural similarity loss
    ssim_loss = 1.0 - tf.reduce_mean(
        tf.image.ssim(
            y_true,
            y_pred,
            max_val=1.0
        )
    )


    # Combined loss
    return (
        0.8 * l1
        +
        0.2 * ssim_loss
    )


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss=enhancement_loss
)


model.summary()


# ============================================================
# 9. TRAINING CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),


    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        verbose=1
    ),


    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

]


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print(
    "Maximum epochs :",
    EPOCHS
)

print(
    "Batch size      :",
    BATCH_SIZE
)

print(
    "Input size      :",
    f"{IMG_SIZE} x {IMG_SIZE}"
)


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# 11. LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)


model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "enhancement_loss": enhancement_loss
    }
)


print(
    "Best model loaded from:",
    MODEL_PATH
)


# ============================================================
# 12. GENERATE ENHANCED TEST IMAGES
# ============================================================

print()
print("=" * 70)
print("GENERATING ENHANCED TEST IMAGES")
print("=" * 70)


predicted = model.predict(
    {
        "image_input": X_img_test,
        "feature_input": X_feat_test
    },
    batch_size=BATCH_SIZE,
    verbose=1
)


# Ensure valid image range

predicted = np.clip(
    predicted,
    0.0,
    1.0
)


# ============================================================
# 13. EVALUATE SSIM AND PSNR
# ============================================================

print()
print("=" * 70)
print("CALCULATING SSIM AND PSNR")
print("=" * 70)


ssim_scores = []
psnr_scores = []


for i in range(
    len(predicted)
):

    pred = (
        predicted[i] * 255.0
    ).astype(np.uint8)


    ref = (
        X_ref_test[i] * 255.0
    ).astype(np.uint8)


    # SSIM

    ssim_value = structural_similarity(
        pred,
        ref,
        channel_axis=2,
        data_range=255
    )


    # PSNR

    psnr_value = peak_signal_noise_ratio(
        pred,
        ref,
        data_range=255
    )


    ssim_scores.append(
        ssim_value
    )

    psnr_scores.append(
        psnr_value
    )


ssim_scores = np.asarray(
    ssim_scores
)

psnr_scores = np.asarray(
    psnr_scores
)


# ============================================================
# 14. PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL ENHANCEMENT RESULTS")
print("=" * 70)


print(
    f"Mean SSIM : {np.mean(ssim_scores):.4f}"
)

print(
    f"Mean PSNR : {np.mean(psnr_scores):.4f} dB"
)

print(
    f"Best SSIM : {np.max(ssim_scores):.4f}"
)

print(
    f"Best PSNR : {np.max(psnr_scores):.4f} dB"
)

print(
    f"Worst SSIM: {np.min(ssim_scores):.4f}"
)

print(
    f"Worst PSNR: {np.min(psnr_scores):.4f} dB"
)


# ============================================================
# 15. SAVE ENHANCED IMAGES
# ============================================================

enhanced_dir = (
    RESULT_DIR /
    "enhanced_test_images"
)

enhanced_dir.mkdir(
    exist_ok=True
)


print()
print(
    "Saving enhanced images..."
)


for i, original_index in enumerate(
    test_idx
):

    name = names[
        original_index
    ]


    output_image = (
        predicted[i] * 255.0
    ).astype(np.uint8)


    output_bgr = cv2.cvtColor(
        output_image,
        cv2.COLOR_RGB2BGR
    )


    cv2.imwrite(
        str(
            enhanced_dir / name
        ),
        output_bgr
    )


# ============================================================
# 16. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({

    "image_name":
        names[test_idx],

    "SSIM":
        ssim_scores,

    "PSNR":
        psnr_scores

})


metrics_df.to_csv(
    RESULT_DIR /
    "enhancement_test_metrics.csv",
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame({

    "metric": [

        "Mean SSIM",
        "Mean PSNR",
        "Best SSIM",
        "Best PSNR",
        "Worst SSIM",
        "Worst PSNR"

    ],

    "value": [

        np.mean(ssim_scores),
        np.mean(psnr_scores),
        np.max(ssim_scores),
        np.max(psnr_scores),
        np.min(ssim_scores),
        np.min(psnr_scores)

    ]

})


summary_df.to_csv(
    RESULT_DIR /
    "enhancement_summary.csv",
    index=False
)


# ============================================================
# 17. SAVE TRAINING LOSS PLOT
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
    "Enhancement Loss"
)


plt.title(
    "Feature-Guided Enhancement Training"
)


plt.legend()

plt.tight_layout()


plt.savefig(
    RESULT_DIR /
    "enhancement_training_loss.png",
    dpi=300
)


plt.close()


# ============================================================
# 18. SAVE INPUT / REFERENCE / ENHANCED COMPARISONS
# ============================================================

comparison_dir = (
    RESULT_DIR /
    "comparison_images"
)

comparison_dir.mkdir(
    exist_ok=True
)


print()
print(
    "Saving comparison images..."
)


for i in range(
    min(5, len(X_img_test))
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )


    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    axes[0].imshow(
        X_img_test[i]
    )

    axes[0].set_title(
        "Raw Input"
    )


    # --------------------------------------------------------
    # REFERENCE
    # --------------------------------------------------------

    axes[1].imshow(
        X_ref_test[i]
    )

    axes[1].set_title(
        "Reference"
    )


    # --------------------------------------------------------
    # ENHANCED
    # --------------------------------------------------------

    axes[2].imshow(
        predicted[i]
    )

    axes[2].set_title(
        "Enhanced\n"
        "SSIM={:.3f}, PSNR={:.2f}".format(
            ssim_scores[i],
            psnr_scores[i]
        )
    )


    # --------------------------------------------------------
    # REMOVE AXES
    # --------------------------------------------------------

    for ax in axes:
        ax.axis("off")


    plt.tight_layout()


    plt.savefig(
        comparison_dir /
        f"comparison_{i + 1}.png",
        dpi=200
    )


    plt.close()


# ============================================================
# 19. FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("FEATURE-GUIDED ENHANCEMENT COMPLETED")
print("=" * 70)


print(
    "Model:",
    MODEL_PATH
)


print(
    "Enhanced images:",
    enhanced_dir
)


print(
    "Metrics:",
    RESULT_DIR /
    "enhancement_test_metrics.csv"
)


print(
    "Summary:",
    RESULT_DIR /
    "enhancement_summary.csv"
)


print(
    "Training plot:",
    RESULT_DIR /
    "enhancement_training_loss.png"
)


print(
    "Comparison images:",
    comparison_dir
)


print()
print(
    "Next evaluation: UIQM and UCIQE"
)

print("=" * 70)
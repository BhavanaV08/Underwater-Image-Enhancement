import cv2
import matplotlib.pyplot as plt
import pandas as pd


from feature_extraction import extract_features



# ============================
# Load Images
# ============================

original = cv2.imread(
    r"C:\Users\vbhav\OneDrive\Desktop\UIE-fyp\dataset\raw-890\9567.png"
)

original = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2RGB
)



preprocessed = cv2.imread(
    "enhanced_output.jpg"
)


preprocessed = cv2.cvtColor(
    preprocessed,
    cv2.COLOR_BGR2RGB
)



# Extract features

original_features = extract_features(original)

preprocessed_features = extract_features(preprocessed)



# ============================
# Create DataFrame
# ============================


df = pd.DataFrame({

    "Feature":
    original_features.keys(),

    "Original":
    original_features.values(),

    "Preprocessed":
    preprocessed_features.values()

})



# Round values

df["Original"] = df["Original"].apply(
    lambda x: f"{x:.3f}"
)

df["Preprocessed"] = df["Preprocessed"].apply(
    lambda x: f"{x:.3f}"
)



# ============================
# Feature Categories
# ============================


categories = {

"Brightness & Statistical Features":
[
"mean",
"std",
"variance",
"entropy",
"dynamic_range",
"rms_contrast"
],


"Color Features":
[
"mean_red",
"mean_green",
"mean_blue",
"colorfulness",
"red_ratio",
"mean_saturation",
"mean_value"
],


"Texture Features":
[
"contrast",
"correlation",
"energy",
"homogeneity",
"ASM",
"dissimilarity",
"glcm_entropy",
"glcm_variance"
],


"Edge & Sharpness Features":
[
"edge_density",
"gradient",
"laplacian_variance",
"keypoint_density"
]

}



# Add category column

feature_category=[]


for feature in df["Feature"]:

    for cat,features in categories.items():

        if feature in features:
            feature_category.append(cat)


df.insert(
    0,
    "Category",
    feature_category
)



# ============================
# Plot Table
# ============================


fig, ax = plt.subplots(
    figsize=(14,14)
)


ax.axis("off")



table = ax.table(

    cellText=df.values,

    colLabels=df.columns,

    cellLoc="center",

    loc="center"

)



# Font settings

table.auto_set_font_size(False)

table.set_fontsize(10)


# Adjust cell size

table.scale(
    1,
    2
)



# Title

plt.title(
    "Comparison of Extracted Features Before and After Preprocessing",
    fontsize=16,
    weight="bold",
    pad=25
)



plt.savefig(
    "feature_comparison_visualization.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print("Table generated successfully")
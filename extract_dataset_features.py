import cv2
import os
import pandas as pd

from feature_extraction import extract_features


# ============================
# Dataset path
# ============================

image_folder = r"C:\Users\vbhav\OneDrive\Desktop\UIE-fyp\dataset\preprocessed"


# Output CSV

output_csv = "feature_dataset.csv"



# Store all feature values

all_features = []



# ============================
# Read every image
# ============================

for filename in os.listdir(image_folder):


    if filename.lower().endswith(
        (".jpg",".png",".jpeg")
    ):


        path = os.path.join(
            image_folder,
            filename
        )


        img = cv2.imread(path)


        if img is None:
            continue



        # BGR → RGB

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )


        # Extract 25 features

        features = extract_features(img)



        # Add image name

        features["image_name"] = filename



        all_features.append(features)



        print(
            filename,
            "completed"
        )



# ============================
# Convert to dataframe
# ============================


df = pd.DataFrame(
    all_features
)



# Move image name first

cols = [
    "image_name"
] + [
    c for c in df.columns
    if c!="image_name"
]


df = df[cols]



# Save CSV

df.to_csv(
    output_csv,
    index=False
)



print("\nFeature extraction completed")

print(
    "Total images:",
    len(df)
)

print(
    "Saved:",
    output_csv
)

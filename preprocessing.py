import cv2
import numpy as np
import os


# ============================
# Dataset Paths
# ============================

input_folder = r"C:\Users\vbhav\OneDrive\Desktop\UIE-fyp\dataset\raw-890"

output_folder = r"C:\Users\vbhav\OneDrive\Desktop\UIE-fyp\dataset\preprocessed"


if not os.path.exists(output_folder):
    os.makedirs(output_folder)



# Fixed size for CNN

WIDTH = 256
HEIGHT = 256



# ============================
# Color Correction
# Modified Gray World Algorithm
# ============================

def color_correction(img):

    img = img.astype(np.float32)


    avg_b = np.mean(img[:,:,0])
    avg_g = np.mean(img[:,:,1])
    avg_r = np.mean(img[:,:,2])


    avg = (avg_b + avg_g + avg_r) / 3



    # Correction strength
    # 0 = no correction
    # 1 = full correction

    strength = 0.35



    img[:,:,0] = img[:,:,0] * (
        1 + strength*((avg/avg_b)-1)
    )


    img[:,:,1] = img[:,:,1] * (
        1 + strength*((avg/avg_g)-1)
    )


    img[:,:,2] = img[:,:,2] * (
        1 + strength*((avg/avg_r)-1)
    )



    img = np.clip(
        img,
        0,
        255
    )


    return img.astype(np.uint8)




# ============================
# Contrast Enhancement
# CLAHE
# ============================

def contrast_adjustment(img):


    # Convert BGR to LAB

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB
    )


    l,a,b = cv2.split(lab)



    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(16,16)
    )



    l = clahe.apply(l)



    lab = cv2.merge(
        [l,a,b]
    )



    output = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )


    return output





# ============================
# Normalization
# Pixel range 0-255 → 0-1
# ============================

def normalize(img):

    return img.astype(
        np.float32
    ) / 255.0





# ============================
# Main Processing
# ============================


count = 0


for file in os.listdir(input_folder):


    if file.lower().endswith(
        (".jpg",".png",".jpeg")
    ):


        image_path = os.path.join(
            input_folder,
            file
        )


        img = cv2.imread(
            image_path
        )


        if img is None:
            continue



        original = img.copy()



        # ------------------------
        # 1. Color Correction
        # ------------------------

        img = color_correction(img)



        # ------------------------
        # 2. Contrast Adjustment
        # ------------------------

        img = contrast_adjustment(img)



        # ------------------------
        # 3. Resize with aspect ratio preservation
        # ------------------------

        def resize_with_padding(image, target_size=(256,256)):

            target_w, target_h = target_size

            h,w = image.shape[:2]


            # Calculate scaling factor

            scale = min(
                target_w/w,
                target_h/h
            )


            new_w = int(w*scale)
            new_h = int(h*scale)


            # Resize maintaining ratio

            resized = cv2.resize(
                image,
                (new_w,new_h),
                interpolation=cv2.INTER_CUBIC
            )


            # Create black canvas

            canvas = np.zeros(
                (target_h,target_w,3),
                dtype=np.uint8
            )


            # Center placement

            x_offset = (target_w-new_w)//2
            y_offset = (target_h-new_h)//2


            canvas[
                y_offset:y_offset+new_h,
                x_offset:x_offset+new_w
            ] = resized


            return canvas



        img = resize_with_padding(
            img,
            (WIDTH,HEIGHT)
        )



        # ------------------------
        # 4. Normalization
        # ------------------------

        img = normalize(img)



        # Convert back to uint8
        # for saving

        img_save = (
            img*255
        ).astype(np.uint8)



        save_path = os.path.join(
            output_folder,
            file
        )



        cv2.imwrite(
            save_path,
            img_save
        )



        print(
            file,
            "processed"
        )


        count += 1



print("\nTotal images processed :",count)

print("Preprocessing Completed")



# ============================
# Display sample result
# ============================

before = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2RGB
)


after = cv2.cvtColor(
    img_save,
    cv2.COLOR_BGR2RGB
)



cv2.imshow(
    "Original",
    cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )
)


cv2.imshow(
    "Preprocessed",
    cv2.cvtColor(
        img_save,
        cv2.COLOR_BGR2RGB
    )
)


cv2.waitKey(0)

cv2.destroyAllWindows()
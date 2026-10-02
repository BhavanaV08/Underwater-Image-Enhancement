import cv2
import numpy as np

from skimage.feature import (
    graycomatrix,
    graycoprops
)

from skimage.measure import shannon_entropy


def extract_features(image):

    features = {}

    # --------------------------
    # Convert Images
    # --------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )


    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )


    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]


    # ==========================
    # CATEGORY 1
    # Statistical Features
    # ==========================


    features["mean"] = np.mean(gray)

    features["std"] = np.std(gray)

    features["variance"] = np.var(gray)

    features["entropy"] = shannon_entropy(gray)

    features["dynamic_range"] = (
        np.max(gray)-np.min(gray)
    )


    features["rms_contrast"] = np.sqrt(
        np.mean(
            (gray-np.mean(gray))**2
        )
    )


    # ==========================
    # CATEGORY 2
    # Color Features
    # ==========================


    features["mean_red"] = np.mean(R)

    features["mean_green"] = np.mean(G)

    features["mean_blue"] = np.mean(B)



    rg = R-G

    yb = 0.5*(R+G)-B


    features["colorfulness"] = np.sqrt(
        np.std(rg)**2 +
        np.std(yb)**2
    )


    features["red_ratio"] = (
        np.mean(R) /
        (np.mean(G)+np.mean(B)+1e-10)
    )


    features["mean_saturation"] = np.mean(
        hsv[:,:,1]
    )


    features["mean_value"] = np.mean(
        hsv[:,:,2]
    )


    # ==========================
    # CATEGORY 3
    # GLCM Features
    # ==========================


    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )


    features["contrast"] = graycoprops(
        glcm,
        "contrast"
    )[0,0]


    features["correlation"] = graycoprops(
        glcm,
        "correlation"
    )[0,0]


    features["energy"] = graycoprops(
        glcm,
        "energy"
    )[0,0]


    features["homogeneity"] = graycoprops(
        glcm,
        "homogeneity"
    )[0,0]


    features["ASM"] = (
        features["energy"]**2
    )


    features["dissimilarity"] = graycoprops(
        glcm,
        "dissimilarity"
    )[0,0]


    features["glcm_entropy"] = -np.sum(
        glcm*np.log2(glcm+1e-10)
    )


    features["glcm_variance"] = np.var(glcm)



    # ==========================
    # CATEGORY 4
    # Edge Features
    # ==========================


    edges = cv2.Canny(
        gray,
        100,
        200
    )


    features["edge_density"] = np.mean(
        edges>0
    )


    gx=cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0
    )

    gy=cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1
    )


    gradient=np.sqrt(
        gx**2+gy**2
    )


    features["gradient"] = np.mean(
        gradient
    )


    laplacian=cv2.Laplacian(
        gray,
        cv2.CV_64F
    )


    features["laplacian_variance"] = np.var(
        laplacian
    )


    sift=cv2.SIFT_create()


    keypoints,des=sift.detectAndCompute(
        gray,
        None
    )


    features["keypoint_density"] = (
        len(keypoints) /
        (gray.shape[0]*gray.shape[1])
    )


    return features
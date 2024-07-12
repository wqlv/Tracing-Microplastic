import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
def is_image_file(filename):
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".gif"]
    return any(filename.lower().endswith(ext) for ext in valid_extensions)
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to load image at {image_path}")

    if len(image.shape) != 2:
        raise ValueError("Image must be a 2-dimensional array")

    gcomat = graycomatrix(image, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], symmetric=True, normed=True)
    contrast = graycoprops(gcomat, 'contrast')
    return contrast.flatten()
image_folder = "  "

for subdir, dirs, files in os.walk(image_folder):
    category_name = os.path.basename(subdir)
    if not category_name or category_name.startswith('.'):
        continue

    features = []
    for file in files:
        if not is_image_file(file):
            continue
        file_path = os.path.join(subdir, file)
        try:
            feature = extract_features(file_path)
            features.append(feature)
        except ValueError as e:
            print(e)
            continue

    if features:
        feature_matrix = np.array(features)
        distances = squareform(pdist(feature_matrix, 'euclidean'))

        plt.figure(figsize=(10, 8))
        sns.heatmap(distances, annot=False, cmap='coolwarm')
        plt.title(f"Heatmap for Category: {category_name}")
        plt.show()
    else:
        print(f"No valid images found in category: {category_name}")

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import os
from skimage.feature import graycomatrix, graycoprops
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
def read_and_preprocess_image(image_path, target_size):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Image cannot be read.")
        if image.shape != target_size:
            image = cv2.resize(image, target_size)
        return image
    except Exception as e:
        print(f"Error in processing {image_path}: {e}")
        return None

def extract_texture_features(image):
    glcm = graycomatrix(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    ASM = graycoprops(glcm, 'ASM')
    return np.concatenate([contrast.flatten(), dissimilarity.flatten(),
                           homogeneity.flatten(), energy.flatten(),
                           correlation.flatten(), ASM.flatten()])

def extract_shape_features(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.hypot(sobelx, sobely)
    return np.array([sobel.mean()])

def extract_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten()
    hist_normalized = hist / sum(hist)  # normalize the histogram
    return hist_normalized

def extract_features(folder_path, target_size):
    texture_features = []
    shape_features = []
    histograms = []
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        image = read_and_preprocess_image(image_path, target_size)
        if image is not None:
            texture = extract_texture_features(image)
            texture_features.append(texture)

            shape = extract_shape_features(image)
            shape_features.append(shape)

            hist = extract_histogram(image)
            histograms.append(hist)
    return texture_features, shape_features, histograms

target_size = (  )

folders = [' ']
all_features = []
labels = []
for i, folder_path in enumerate(folders):
    texture_f, shape_f, hist_f = extract_features(folder_path, target_size)
    features = np.hstack((texture_f, shape_f, hist_f))
    label = np.full((features.shape[0],), i)
    all_features.append(features)
    labels.append(label)
all_features = np.vstack(all_features)
labels = np.concatenate(labels)umpa
scaler = StandardScaler()
all_features_normalized = scaler.fit_transform(all_features)
tsne = TSNE(n_components=2, init='random', learning_rate='auto', perplexity= , n_iter=  , random_state=  )
tsne_features = tsne.fit_transform(all_features_normalized)

colors = ['']
folder_labels = ['']
plt.figure(figsize=(10, 8))
for i, folder_name in enumerate(folders):
    plt.scatter(tsne_features[labels == i, 0], tsne_features[labels == i, 1],
                c=colors[i], label=folder_labels[i], edgecolor='k', s=60)

plt.title('t-SNE of Combined Image Features')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.legend()

plt.grid(False)

plt.show()
tsne_df = pd.DataFrame(tsne_features, columns=['t-SNE Feature 1', 't-SNE Feature 2'])
tsne_df['Label'] = [folder_labels[label] for label in labels]

csv_filename = "tsne_features.csv"
tsne_df.to_csv(csv_filename, index=False)

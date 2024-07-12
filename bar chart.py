import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
folder_path = ''
texture_features = []
histograms = []
target_size = (   )
for image_name in os.listdir(folder_path):
    image_path = os.path.join(folder_path, image_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image.shape != target_size:
        image = cv2.resize(image, target_size)
    glcm = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    texture = graycoprops(glcm, 'contrast')
    texture_features.append(texture.flatten())
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    histograms.append(hist.flatten())
texture_df = pd.DataFrame(texture_features)
histogram_df = pd.DataFrame(histograms)
texture_df.to_excel('', index=False)
histogram_df.to_excel('', index=False)
average_texture = np.mean(texture_features, axis=0)
average_histogram = np.mean(histograms, axis=0)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(range(len(average_texture)), average_texture)
plt.title('')
plt.subplot(1, 2, 2)
plt.plot(average_histogram)
plt.title('Average Gray Histogram')
plt.tight_layout()
plt.show()

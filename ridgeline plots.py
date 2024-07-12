import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
import seaborn as sns

folders = ['']

texture_features = []
histograms = []
categories = []

target_size = (1536, 1025)

for folder in folders:
    folder_path = folder
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

        categories.append(folder)

data = pd.DataFrame(texture_features)
data['Category'] = categories

melted_data = data.melt(id_vars='Category', var_name='Feature', value_name='Value')

plt.figure(figsize=(12, 6))
sns.kdeplot(data=melted_data, x='Value', hue='Category', multiple='stack', common_norm=False)
plt.title('Ridge Plot of Texture Features')
plt.xlabel('Feature Value')
plt.ylabel('Density')
plt.show()

data.to_excel("texture_features.xlsx", index=False)

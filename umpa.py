import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from umap import UMAP
from sklearn.cluster import KMeans

model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def load_images_from_folder(folder):
    images = []
    labels = []
    label_map = {}
    for label, subfolder in enumerate(os.listdir(folder)):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            label_map[subfolder] = label  # 将子文件夹名称映射到标签
            for file in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, file)
                img = load_img(img_path, target_size=(224, 224))  # 调整图像大小
                img = img_to_array(img)
                img = preprocess_input(img)
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels), label_map

folder_path = '  '
images, numeric_labels, label_map = load_images_from_folder(folder_path)

features = model.predict(images)

umap = UMAP(n_neighbors=5, min_dist=0.3, n_components=2)
reduced_data = umap.fit_transform(features)

string_labels = np.array([folder_name for folder_name in os.listdir(folder_path) for _ in os.listdir(os.path.join(folder_path, folder_name))])
string_labels = np.array([label_map[label] for label in string_labels])

kmeans = KMeans(n_clusters=len(label_map))
kmeans.fit(np.concatenate([reduced_data, string_labels[:, np.newaxis]], axis=1))
cluster_labels = kmeans.labels_

cluster_names = {label: folder_name for folder_name, label in label_map.items()}
named_labels = np.array([cluster_names[label] for label in cluster_labels])

for name in label_map.keys():
    plt.scatter(reduced_data[named_labels == name, 0],
                reduced_data[named_labels == name, 1],
                label=name)
plt.legend()
plt.colorbar()
plt.title('UMAP projection of the SEM Images dataset')
plt.show()

df = pd.DataFrame(reduced_data, columns=['UMAP_1', 'UMAP_2'])
df['Cluster'] = named_labels

df.to_excel('umap_coordinates2.xlsx', index=False)

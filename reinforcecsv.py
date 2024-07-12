import os
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter


def augment_csv(file_path, output_dir, num_augmented_files=20):

    df = pd.read_csv(file_path, header=None)


    x_values = df.iloc[:, 0]

    for i in range(num_augmented_files):
        y_values = df.iloc[:, 1]

        noise = np.random.normal(0, 0.01, size=y_values.shape)
        y_noisy = y_values + noise


        x_interpolated = np.linspace(x_values.min(), x_values.max(), len(x_values))
        y_interpolated = np.interp(x_interpolated, x_values, y_noisy)


        y_smoothed = gaussian_filter(y_interpolated, sigma=1)


        df_augmented = pd.DataFrame({
            df.columns[0]: x_interpolated,
            df.columns[1]: y_smoothed
        })

        augmented_file_name = f'augmented_{i + 1}_' + os.path.basename(file_path)
        df_augmented.to_csv(os.path.join(output_dir, augmented_file_name), index=False, header=False)


def augment_spectral_data_in_directory(directory_path):
    for subdir, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.csv'):
                file_path = os.path.join(subdir, file)
                output_dir = os.path.join(subdir, 'augmented')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                augment_csv(file_path, output_dir)


# 调用函数
augment_spectral_data_in_directory('   ')


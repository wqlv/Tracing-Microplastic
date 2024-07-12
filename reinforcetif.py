import os
import random
from PIL import Image, ImageEnhance

def augment_image(image, augmentation_count=10):
    augmented_images = []
    for _ in range(augmentation_count):
        transformed = image.copy()


        if random.choice([True, False]):
            transformed = transformed.transpose(Image.FLIP_LEFT_RIGHT)
        if random.choice([True, False]):
            transformed = transformed.transpose(Image.FLIP_TOP_BOTTOM)

        enhancer = ImageEnhance.Brightness(transformed)
        transformed = enhancer.enhance(random.uniform(0.7, 1.3))
        enhancer = ImageEnhance.Contrast(transformed)
        transformed = enhancer.enhance(random.uniform(0.7, 1.3))

        augmented_images.append(transformed)

    return augmented_images

def augment_images_in_folder(folder_path, augmentation_count=10):
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            augmented_images = augment_image(image, augmentation_count)

            # Save augmented images
            for i, aug_image in enumerate(augmented_images):
                aug_image.save(os.path.join(folder_path, f"{filename}_aug_{i}.tif"))

folder_path =
augment_images_in_folder(folder_path, augmentation_count=10)

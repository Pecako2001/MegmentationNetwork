import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

class PolygonSegmentationDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None, target_size=(320, 240), use_cache=True, use_augmentation=True, augmentation_probability=0.02):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.target_size = target_size
        self.use_cache = use_cache
        self.use_augmentation = use_augmentation
        self.augmentation_probability = augmentation_probability
        self.cache_file = os.path.join(image_dir, 'dataset_cache.pkl')

        if self.use_cache and os.path.exists(self.cache_file):
            print(f"Loading dataset from cache: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                self.image_filenames, self.annotations = pickle.load(f)
        else:
            self.image_filenames, self.annotations = self._load_and_cache_data()
            if self.use_cache:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump((self.image_filenames, self.annotations), f)
                print(f"Dataset cached at: {self.cache_file}")

        # Define default transformations based on the use_augmentation flag
        if transform is None:
            if use_augmentation:
                self.augmentation_transform = A.Compose([
                    A.OneOf([
                        A.HorizontalFlip(p=1.0),
                        A.VerticalFlip(p=1.0),
                        A.RandomRotate90(p=1.0),
                    ], p=1.0),
                    A.OneOf([
                        A.Blur(p=0.05),
                        A.GaussianBlur(p=0.05),
                        A.MedianBlur(blur_limit=5, p=0.05),
                    ], p=1.0),
                    A.OneOf([
                        A.RandomBrightnessContrast(p=0.05),
                        A.CLAHE(p=0.05),
                        A.RGBShift(p=0.05),
                    ], p=1.0),
                    A.OneOf([
                        A.ChannelShuffle(p=0.05),
                        A.HueSaturationValue(p=0.05),
                        A.ToGray(p=0.05),
                    ], p=1.0),
                    A.Resize(height=self.target_size[1], width=self.target_size[0]),  # Ensure final size is consistent
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ], additional_targets={'mask': 'mask'})
            
            self.normal_transform = A.Compose([
                A.Resize(height=self.target_size[1], width=self.target_size[0]),  # Ensure consistent size without augmentation
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})

    def _load_and_cache_data(self):
        image_filenames = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        annotations = []

        print(f"Caching dataset: {len(image_filenames)} images")
        for filename in tqdm(image_filenames, total=len(image_filenames), unit='image'):
            image_path = os.path.join(self.image_dir, filename)
            annotation_path = os.path.join(self.annotation_dir, filename.replace('.jpg', '.txt').replace('.png', '.txt'))

            mask_data = None
            if os.path.exists(annotation_path):
                mask_data = []
                with open(annotation_path, 'r') as file:
                    for line in file:
                        line = line.strip().split()
                        class_idx = int(line[0])  # Class index
                        polygon_points = list(map(float, line[1:]))
                        mask_data.append((class_idx, polygon_points))

            annotations.append((image_path, mask_data))

        return image_filenames, annotations

    def _load_image_and_mask(self, image_path, annotation):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        if annotation:
            for class_idx, polygon_points in annotation:
                polygon_points = np.array(polygon_points, dtype=np.float32).reshape(-1, 2)
                polygon_points[:, 0] *= width  # Convert x-coordinates
                polygon_points[:, 1] *= height  # Convert y-coordinates
                polygon_points = polygon_points.astype(np.int32)  # Convert to integer
                cv2.fillPoly(mask, [polygon_points], class_idx)

        return image, mask

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path, annotation = self.annotations[idx]
        image, mask = self._load_image_and_mask(image_path, annotation)

        #original_image = image.copy()

        # Randomly decide whether to apply augmentation
        if self.use_augmentation and random.random() < self.augmentation_probability:
            augmented = self.augmentation_transform(image=image, mask=mask)
            #print(f"Augmented: {image_path}")

            # #Convert augmented image and mask back to NumPy for saving
            # augmented_image_np = augmented['image'].permute(1, 2, 0).cpu().numpy()
            # augmented_image_np = (augmented_image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255.0
            # augmented_image_np = augmented_image_np.astype(np.uint8)

            # # Convert original image to match size and type
            # original_image_np = cv2.resize(original_image, (self.target_size[0], self.target_size[1]))
            # original_image_np = cv2.cvtColor(original_image_np, cv2.COLOR_RGB2BGR)

            # # Concatenate original and augmented images
            # comparison_image = np.concatenate((original_image_np, cv2.cvtColor(augmented_image_np, cv2.COLOR_RGB2BGR)), axis=1)
            # save_path = f'comparison_{os.path.basename(image_path)}'
            # cv2.imwrite(save_path, comparison_image)
            #print(f"Saved comparison image to: {save_path}")
        else:
            augmented = self.normal_transform(image=image, mask=mask)
        
        image = augmented['image']
        mask = augmented['mask']

        return image.float() / 255.0, mask.long()  # Normalize image to [0, 1] and ensure mask is a long tensor

import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PolygonSegmentationDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None, target_size=(320, 240), use_cache=True, use_augmentation=True):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.target_size = target_size
        self.use_cache = use_cache
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
                self.transform = A.Compose([
                    A.SmallestMaxSize(max_size=min(self.target_size)),  # Resize smaller side to target size
                    A.PadIfNeeded(min_height=self.target_size[1], min_width=self.target_size[0], border_mode=cv2.BORDER_CONSTANT, value=0),  # Pad to target size
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ])
            else:
                self.transform = A.Compose([
                    A.SmallestMaxSize(max_size=min(self.target_size)),  # Resize smaller side to target size
                    A.PadIfNeeded(min_height=self.target_size[1], min_width=self.target_size[0], border_mode=cv2.BORDER_CONSTANT, value=0),  # Pad to target size
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ])

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

        # Resize image and mask to the target size
        image = cv2.resize(image, self.target_size)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        return image, mask

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path, annotation = self.annotations[idx]
        image, mask = self._load_image_and_mask(image_path, annotation)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image.float() / 255.0, mask.long()  # Normalize image to [0, 1] and ensure mask is a long tensor

import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from tqdm import tqdm

class PolygonSegmentationDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None, target_size=(320, 240), use_cache=True):
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

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
        mask = torch.from_numpy(mask).long()

        return image, mask

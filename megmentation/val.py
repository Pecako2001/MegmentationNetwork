import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex, Precision
import megmentation.utils as utils
from megmentation.dataset import PolygonSegmentationDataset
from megmentation.model import BasicSegmentationModel
import numpy as np
import cv2
from prettytable import PrettyTable

def main():
    parser = argparse.ArgumentParser(description="Validate the MegmentationNetwork model.")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for validation.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model file.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_dir = os.path.join(args.dataset, 'images')
    annotation_dir = os.path.join(args.dataset, 'labels')
    
    val_dataset = PolygonSegmentationDataset(image_dir, annotation_dir, use_cache=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = BasicSegmentationModel(num_classes=23).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    mean_iou = JaccardIndex(task='multiclass', num_classes=23).to(device)
    precision = Precision(task='multiclass', num_classes=23).to(device)
    
    # Determine the next available validation run folder
    run_folder = utils.find_next_run_folder(base_dir='runs', prefix='val')

    class_counts = np.zeros(23, dtype=int)
    class_ious = np.zeros(23, dtype=float)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_dataloader):
            images = images.to(device).float()
            masks = masks.to(device).long()

            outputs = model(images)
            loss = criterion(outputs, masks)

            _, predictions = torch.max(outputs, 1)
            mean_iou.update(predictions, masks)
            precision.update(predictions, masks)

            # Calculate IOU for each class and update class-wise IOU and counts
            for i in range(args.batch_size):
                mask = masks[i].cpu().numpy()
                pred = predictions[i].cpu().numpy()

                for cls in range(23):
                    cls_mask = (mask == cls)
                    cls_pred = (pred == cls)
                    intersection = np.logical_and(cls_mask, cls_pred).sum()
                    union = np.logical_or(cls_mask, cls_pred).sum()
                    if union > 0:
                        class_ious[cls] += intersection / union
                        class_counts[cls] += cls_mask.sum()

            # Save images with predictions as a grid
            utils.save_annotated_images(images.cpu(), masks.cpu(), predictions.cpu(), save_dir=run_folder, epoch=batch_idx)

            # Print current metrics
            print(f'\rBatch [{batch_idx+1}/{len(val_dataloader)}] - IOU: {mean_iou.compute().item():.4f} - Precision: {precision.compute().item():.4f}', end='')

    # Calculate final IOU per class
    class_ious /= (class_counts + 1e-10)  # Prevent division by zero

    # Print final metrics
    print(f'\nFinal IOU: {mean_iou.compute().item():.4f} - Final Precision: {precision.compute().item():.4f}')

    # Create and print table of IOU per class
    table = PrettyTable(["Class", "Instances", "IoU"])
    class_names = ["class1", "class2", "class3", "cardoor", "window"]  # Replace with actual class names
    for i, name in enumerate(class_names):
        table.add_row([name, class_counts[i], class_ious[i]])

    print("\nClass-wise IOU and Instances:")
    print(table)

    # Reset metrics
    mean_iou.reset()
    precision.reset()

if __name__ == "__main__":
    main()

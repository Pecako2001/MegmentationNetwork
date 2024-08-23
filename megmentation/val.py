import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex, Precision
import megmentation.utils as utils
from megmentation.dataset import PolygonSegmentationDataset
from megmentation.model import TransferLearningSegmentationModel
import numpy as np
import cv2
from prettytable import PrettyTable
import random
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def load_class_names(yaml_file):
    with open(yaml_file, 'r') as stream:
        data = yaml.safe_load(stream)
        class_names = [v for k, v in sorted(data['names'].items())]
    return class_names

def apply_color_map(predictions, num_classes, class_names):
    # Define a fixed color palette
    random.seed(42)  # Ensure consistent colors
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_classes)]
    
    colored_masks = np.zeros((*predictions.shape, 3), dtype=np.uint8)
    for cls in range(num_classes):
        mask = predictions == cls
        colored_masks[mask] = colors[cls]
        
    return colored_masks, colors

def draw_borders_and_labels(image, predictions, class_names, colors):
    # Ensure the image is in the correct format (uint8 and 3 channels)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    # If the image has only 1 channel (grayscale), convert it to 3-channel BGR
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Convert the image to a format that OpenCV can definitely work with
    image = np.ascontiguousarray(image)
    
    # Create a blank canvas to draw the segmentation masks
    mask_overlay = np.zeros_like(image)
    
    # Keep track of which classes are present in the image
    present_classes = set()
    
    # Draw the segmentation masks on the canvas with transparency
    alpha = 0.5  # Transparency factor
    for cls in range(len(class_names)):
        mask = (predictions == cls).astype(np.uint8)
        if mask.sum() > 0:
            present_classes.add(cls)
            color = colors[cls]
            mask_overlay[mask == 1] = color

    # Overlay the mask on the original image
    overlaid_image = cv2.addWeighted(image, 1, mask_overlay, alpha, 0)

    # Plot the image with the mask overlay
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.imshow(cv2.cvtColor(overlaid_image, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    
    # Create a legend only for the present classes
    legend_patches = [mpatches.Patch(color=np.array(colors[cls]) / 255, label=class_names[cls]) for cls in present_classes]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    
    # Convert the Matplotlib figure to an image
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    # Return the final image
    return image_from_plot

def main():
    parser = argparse.ArgumentParser(description="Validate the MegmentationNetwork model.")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for validation.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model file.")
    parser.add_argument('--save_full_images', action='store_true', help="Save full images with predictions.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_dir = os.path.join(args.dataset, 'valid/images')
    annotation_dir = os.path.join(args.dataset, 'valid/labels')
    
    # Load class names from the data.yaml file
    yaml_file = os.path.join(args.dataset, 'data.yaml')
    class_names = load_class_names(yaml_file)
    
    val_dataset = PolygonSegmentationDataset(image_dir, annotation_dir, use_cache=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = TransferLearningSegmentationModel(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    mean_iou = JaccardIndex(task='multiclass', num_classes=len(class_names)).to(device)
    precision = Precision(task='multiclass', num_classes=len(class_names)).to(device)
    
    # Determine the next available validation run folder
    run_folder = utils.find_next_run_folder(base_dir='runs', prefix='val')

    class_counts = np.zeros(len(class_names), dtype=int)
    class_ious = np.zeros(len(class_names), dtype=float)

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
            batch_size = masks.size(0)
            for i in range(batch_size):
                mask = masks[i].cpu().numpy()
                pred = predictions[i].cpu().numpy()

                for cls in range(len(class_names)):
                    cls_mask = (mask == cls)
                    cls_pred = (pred == cls)
                    intersection = np.logical_and(cls_mask, cls_pred).sum()
                    union = np.logical_or(cls_mask, cls_pred).sum()
                    if union > 0:
                        class_ious[cls] += intersection / union
                        class_counts[cls] += cls_mask.sum()

                # Apply the color map to the prediction
                colored_pred, colors = apply_color_map(pred, num_classes=len(class_names), class_names=class_names)
                if args.save_full_images:
                    # Overlay the prediction with borders and labels
                    result_image = draw_borders_and_labels(images[i].cpu().numpy().transpose(1, 2, 0) * 255, pred, class_names, colors)
                    
                    # Save the result image
                    result_image_path = os.path.join(run_folder, f"result_{batch_idx}_{i}.png")
                    cv2.imwrite(result_image_path, result_image)
            
            # Save images with predictions as a grid (Optional if you want to save as a grid)
            utils.save_annotated_images(images.cpu(), masks.cpu(), predictions.cpu(), save_dir=run_folder, epoch=batch_idx)

            # Print current metrics
            print(f'\rBatch [{batch_idx+1}/{len(val_dataloader)}] - IOU: {mean_iou.compute().item():.4f} - Precision: {precision.compute().item():.4f}', end='')

    # Calculate final IOU per class
    class_ious /= (class_counts + 1e-10)  # Prevent division by zero

    # Convert class-wise IOU to percentages
    class_ious = class_ious * 100

    # Print final metrics
    print(f'\nFinal IOU: {mean_iou.compute().item() * 100:.2f}% - Final Precision: {precision.compute().item() * 100:.2f}%')

    # Create and print table of IOU per class
    table = PrettyTable(["Class", "Instances", "IoU (%)"])
    for i, name in enumerate(class_names):
        table.add_row([name, class_counts[i], f'{class_ious[i]:.4f}%'])

    print("\nClass-wise IOU and Instances:")
    print(table)

    # Reset metrics
    mean_iou.reset()
    precision.reset()

if __name__ == "__main__":
    main()

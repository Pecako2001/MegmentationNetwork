import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from tqdm import tqdm
from scipy.ndimage import label

def find_next_run_folder(base_dir='runs', prefix='train'):
    os.makedirs(base_dir, exist_ok=True)
    existing_runs = [int(folder[len(prefix):]) for folder in os.listdir(base_dir) if folder.startswith(prefix) and folder[len(prefix):].isdigit()]
    next_run_number = max(existing_runs, default=0) + 1
    os.makedirs(os.path.join(base_dir, f'{prefix}{next_run_number}'), exist_ok=True)
    return os.path.join(base_dir, f'{prefix}{next_run_number}')

def save_annotated_images(images, masks, predictions, save_dir='runs', epoch=0):
    os.makedirs(save_dir, exist_ok=True)
    
    images = images.permute(0, 2, 3, 1).numpy() * 255  # Convert to [0, 255] range
    images = images.astype(np.uint8)
    masks = masks.numpy()
    predictions = predictions.numpy()

    grid_size = 4
    grid_height = grid_size
    grid_width = grid_size
    img_height, img_width, _ = images[0].shape
    grid_image = np.zeros((img_height * grid_height, img_width * grid_width, 3), dtype=np.uint8)
    original_grid_image = np.zeros((img_height * grid_height, img_width * grid_width, 3), dtype=np.uint8)

    for i in range(min(images.shape[0], grid_size * grid_size)):
        img = images[i]

        # Place original image in the grid for comparison
        row = i // grid_width
        col = i % grid_width
        original_grid_image[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width] = img

        # Apply color map to the mask (ensure masks are valid)
        if masks[i].max() > 0:
            mask = (masks[i] * 255 / masks[i].max()).astype(np.uint8)
            mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        else:
            mask = np.zeros_like(img, dtype=np.uint8)

        if predictions[i].max() > 0:
            pred = (predictions[i] * 255 / predictions[i].max()).astype(np.uint8)
            pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
        else:
            pred = np.zeros_like(img, dtype=np.uint8)

        # Overlay prediction on the original image
        combined_img = cv2.addWeighted(img, 0.7, pred, 0.3, 0)

        # Place combined image in the grid
        grid_image[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width] = combined_img

    # Save the original images grid
    original_save_path = os.path.join(save_dir, f'original_Epoch_{epoch}.png')
    cv2.imwrite(original_save_path, original_grid_image)
    print(f"Saved original image grid to {original_save_path}")

    # Save the annotated images grid
    save_path = os.path.join(save_dir, f'validation_Epoch_{epoch}.png')
    cv2.imwrite(save_path, grid_image)
    print(f"Saved annotated image grid to {save_path}")

def save_training_results(run_folder, epoch, epoch_loss, validation_loss, avg_iou, avg_precision, avg_recall, learning_rate):
    csv_path = os.path.join(run_folder, 'results.csv')
    new_row = {
        'epoch': epoch,
        'epoch_loss': epoch_loss,
        'validation_loss': validation_loss,
        'avg_iou': avg_iou,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'learning_rate': learning_rate
    }
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=new_row.keys())
    else:
        df = pd.read_csv(csv_path)

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)


def plot_training_results(run_folder):
    csv_path = os.path.join(run_folder, 'results.csv')
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['epoch_loss'], label='Training Loss')
    plt.plot(df['epoch'], df['validation_loss'], label='Validation Loss')
    plt.plot(df['epoch'], df['avg_iou'], label='Average IoU')
    plt.plot(df['epoch'], df['avg_precision'], label='Average Precision')
    plt.plot(df['epoch'], df['avg_recall'], label='Average Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend()
    plt.grid(True)
    plt.title('Training and Validation Metrics Over Epochs')
    plt.savefig(os.path.join(run_folder, 'training_results.png'))
    plt.show()

def visualize_dataset_sample(dataset, run_folder, idx=0):
    image, mask = dataset[idx]
    image = image.permute(1, 2, 0).numpy()
    cv2.imwrite(os.path.join(run_folder, 'sample_image.jpg'), cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

def plot_class_distribution(dataset, run_folder, num_classes=23):
    class_counts = np.zeros(num_classes, dtype=int)

    for _, mask in tqdm(dataset, desc="Processing masks", unit="mask"):
        for i in range(num_classes):
            _, num_features = label(mask == i)
            class_counts[i] += num_features  # Count contiguous regions (instances)

    plt.figure(figsize=(12, 8))

    # Generate a color for each bar
    colors = plt.cm.get_cmap('tab20', num_classes).colors

    bars = plt.bar(range(num_classes), class_counts, color=colors)

    # No scientific notation
    plt.ticklabel_format(style='plain', axis='y')

    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.title('Class Distribution in Dataset')

    # Add text on top of each bar
    for bar, count in zip(bars, class_counts):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{count}', ha='center', va='bottom', rotation=90)

    # Save the figure without displaying it
    save_path = os.path.join(run_folder, 'classes.jpg')
    plt.savefig(save_path)
    plt.close()

    print(f"Class distribution saved to {save_path}")

def visualize_random_sample_grid(dataset, run_folder, grid_size=4, batch_num=1):
    indices = np.random.choice(len(dataset), grid_size * grid_size, replace=False)
    images, masks = [], []

    target_size = (320, 240)  # Define your consistent target size

    for i in indices:
        image, mask = dataset[i]

        # Resize the image to the target size
        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
        
        # Resize the mask to the target size (convert to float before interpolation)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=target_size, mode='nearest').long().squeeze(0).squeeze(0)

        images.append(image)
        masks.append(mask)

    images = torch.stack(images).permute(0, 2, 3, 1).numpy() * 255
    masks = torch.stack(masks).numpy()

    img_height, img_width, _ = images[0].shape
    grid_image = np.zeros((img_height * grid_size, img_width * grid_size, 3), dtype=np.uint8)

    for i in range(grid_size * grid_size):
        img = images[i].astype(np.uint8)
        mask = (masks[i] * 255 / masks[i].max()).astype(np.uint8)
        mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        combined_img = cv2.addWeighted(img, 0.6, mask, 0.4, 0)

        row = i // grid_size
        col = i % grid_size
        grid_image[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width] = combined_img

    save_path = os.path.join(run_folder, f'random_sample_grid_batch_{batch_num}.jpg')
    cv2.imwrite(save_path, grid_image)
    print(f"Saved random sample grid to {save_path}")

def save_input_images(images, masks, save_dir, batch_idx):
    os.makedirs(save_dir, exist_ok=True)
    images = images.permute(0, 2, 3, 1).cpu().numpy() * 255  # Convert to [0, 255] range
    images = images.astype(np.uint8)
    masks = masks.cpu().numpy()

    grid_size = 4
    grid_height = grid_size
    grid_width = grid_size
    img_height, img_width, _ = images[0].shape
    grid_image = np.zeros((img_height * grid_height, img_width * grid_width, 3), dtype=np.uint8)
    mask_grid_image = np.zeros((img_height * grid_height, img_width * grid_width, 3), dtype=np.uint8)

    for i in range(min(images.shape[0], grid_size * grid_size)):
        img = images[i].astype(np.uint8)
        mask = masks[i]

        row = i // grid_width
        col = i % grid_width

        # Place original image in the grid
        grid_image[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width] = img

        # Convert mask to single-channel 8-bit format for applyColorMap
        if mask.max() > 0:
            mask = (mask * 255 / mask.max()).astype(np.uint8)
        else:
            mask = np.zeros_like(mask, dtype=np.uint8)

        # Ensure the mask is a 2D single-channel image
        if len(mask.shape) == 3 and mask.shape[2] == 1:
            mask = mask.squeeze(axis=2)

        # Apply color map to the mask
        mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        mask_grid_image[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width] = mask

    # Save the original images grid
    original_save_path = os.path.join(save_dir, f'input_images_batch_{batch_idx}.png')
    cv2.imwrite(original_save_path, grid_image)
    print(f"Saved input images grid to {original_save_path}")

    # Save the masks grid
    mask_save_path = os.path.join(save_dir, f'input_masks_batch_{batch_idx}.png')
    cv2.imwrite(mask_save_path, mask_grid_image)
    print(f"Saved input masks grid to {mask_save_path}")
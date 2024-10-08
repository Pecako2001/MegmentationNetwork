import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchmetrics import JaccardIndex, Precision, Recall
from megmentation.dataset import PolygonSegmentationDataset
from megmentation.model import TransferLearningSegmentationModel
import megmentation.utils as utils
import os, yaml
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
plt.use('Agg')
import torchvision.transforms as T

def get_args():
    parser = argparse.ArgumentParser(description="Training Segmentation Network")
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset folder')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=12, help='Input batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer to use: adam, adamw or sgd')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')

    return parser.parse_args()

def load_yaml_config(yaml_file):
    with open(yaml_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def validate_model(model, dataloader, criterion, device, run_folder, epoch=0):
    model.eval()
    validation_loss = 0.0
    mean_iou = JaccardIndex(task='multiclass', num_classes=23).to(device)
    precision = Precision(task='multiclass', num_classes=23).to(device)
    recall = Recall(task='multiclass', num_classes=23).to(device)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device).float()
            masks = masks.to(device).long()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            validation_loss += loss.item() * images.size(0)
            
            _, predictions = torch.max(outputs, 1)
            mean_iou.update(predictions, masks)
            precision.update(predictions, masks)
            recall.update(predictions, masks)

            if batch_idx == 0 and epoch % 10 == 0:
                utils.save_annotated_images(images.cpu(), masks.cpu(), predictions.cpu(), save_dir=run_folder, epoch=epoch)
    
    avg_iou = mean_iou.compute().item()
    avg_precision = precision.compute().item()
    avg_recall = recall.compute().item()
    mean_iou.reset()
    precision.reset()
    recall.reset()
    return validation_loss / len(dataloader.dataset), avg_iou, avg_precision, avg_recall

def train_model(num_epochs, model, train_dataloader, validation_dataloader, criterion, optimizer, device, run_folder):
    best_validation_loss = float('inf')
    scaler = GradScaler()
    print(model)
    # Initialize lists to store metrics for real-time plotting
    epoch_losses = []
    validation_losses = []
    ious = []
    precisions = []
    recalls = []

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()

        for batch_idx, (images, masks) in enumerate(train_dataloader):
            images = images.to(device).float()
            masks = masks.to(device).long()
            
            #utils.save_input_images(images.cpu(), masks.cpu(), save_dir=run_folder, batch_idx=batch_idx)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

            # Calculate elapsed time and estimated time remaining
            elapsed_time = time.time() - epoch_start_time
            batches_left = len(train_dataloader) - (batch_idx + 1)
            estimated_time_left = (elapsed_time / (batch_idx + 1)) * batches_left

            # Print progress for the current batch
            print(f'\rEpoch [{epoch+1}/{num_epochs}] - Batch [{batch_idx+1}/{len(train_dataloader)}] - '
                  f'Loss: {loss.item():.4f} - '
                  f'Elapsed Time: {elapsed_time:.2f}s - Est. Time Left: {estimated_time_left:.2f}s', end='')

        # Compute epoch-level metrics
        epoch_loss = running_loss / len(train_dataloader.dataset)
        validation_loss, avg_iou, avg_precision, avg_recall = validate_model(
            model, validation_dataloader, criterion, device, run_folder, epoch=epoch
        )

        # Store metrics for plotting
        epoch_losses.append(epoch_loss)
        validation_losses.append(validation_loss)
        ious.append(avg_iou)
        precisions.append(avg_precision)
        recalls.append(avg_recall)

        # Update the best model if necessary
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), os.path.join(run_folder, 'best_model.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(run_folder, 'last_model.pth'))

        # Save metrics to CSV
        utils.save_training_results(run_folder, epoch+1, epoch_loss, validation_loss, avg_iou, avg_precision, avg_recall, optimizer.param_groups[0]['lr'])

        # Calculate total elapsed time and time remaining
        total_elapsed_time = time.time() - start_time
        estimated_total_time = (total_elapsed_time / (epoch + 1)) * num_epochs
        estimated_time_remaining = estimated_total_time - total_elapsed_time

        # Print epoch summary
        print(f'\nEpoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Validation Loss: {validation_loss:.4f}, '
              f'IoU: {avg_iou:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f} - '
              f'Elapsed Time: {total_elapsed_time:.2f}s, Est. Time Remaining: {estimated_time_remaining:.2f}s')

    # Save the final model
    torch.save(model.state_dict(), os.path.join(run_folder, 'last_model.pth'))
    utils.plot_training_results(run_folder)

def main():
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data from data.yaml
    yaml_path = os.path.join(args.dataset, 'data.yaml')
    config = load_yaml_config(yaml_path)

    train_image_dir = os.path.join(args.dataset, config['train'])
    train_annotation_dir = os.path.join(args.dataset, 'train/labels')
    validation_image_dir = os.path.join(args.dataset, config['val'])
    validation_annotation_dir = os.path.join(args.dataset, 'valid/labels')
    #test_image_dir = os.path.join(args.dataset, config['test'])
    #test_annotation_dir = os.path.join(args.dataset, 'test/labels')
    class_names = config['names']

    train_dataset = PolygonSegmentationDataset(train_image_dir, train_annotation_dir, use_cache=True, use_augmentation=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # Create an iterator over the DataLoader
    data_iter = iter(train_dataloader)

    # Fetch the first batch of images and masks
    images, masks = next(data_iter)

    run_folder = utils.find_next_run_folder()
    # If you want to visualize a few images and masks
    for i in range(min(len(images), 4)):  # Visualize the first 4 images
        img = images[i].permute(1, 2, 0).numpy()  # Convert to HWC format
        mask = masks[i].numpy()

        # Reverse normalization if applied
        img = (img * 255).astype(np.uint8)

        # Plot the image and its corresponding mask
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Image")

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Mask")
        plt.savefig(f"{run_folder}/sample_{i+1}.png")

    validation_dataset = PolygonSegmentationDataset(validation_image_dir, validation_annotation_dir, use_cache=True, use_augmentation=False)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Plot and save class distribution
    #utils.plot_class_distribution(train_dataset, run_folder, num_classes=len(class_names))

    # Visualize and save three random grids of samples at the start
    for i in range(3):
        utils.visualize_random_sample_grid(train_dataset, run_folder, grid_size=4, batch_num=i+1)


    model = TransferLearningSegmentationModel(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimzier
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    num_epochs = args.epochs

    train_model(num_epochs, model, train_dataloader, validation_dataloader, criterion, optimizer, device, run_folder)

if __name__ == "__main__":
    main()
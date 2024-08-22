import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchmetrics import JaccardIndex, Precision, Recall
from dataset import PolygonSegmentationDataset
from model import BasicSegmentationModel
import utils as utils
import os
import time

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

        # Save metrics to CSV
        utils.save_training_results(run_folder, epoch+1, epoch_loss, validation_loss, avg_iou, avg_precision, avg_recall)

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

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_dir = 'basic_network/dataset/train/images'
    annotation_dir = 'basic_network/dataset/train/labels'
    validation_image_dir = 'basic_network/dataset/valid/images'
    validation_annotation_dir = 'basic_network/dataset/valid/labels'

    train_dataset = PolygonSegmentationDataset(image_dir, annotation_dir, use_cache=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=4)

    validation_dataset = PolygonSegmentationDataset(validation_image_dir, validation_annotation_dir, use_cache=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=12, shuffle=False, num_workers=4)

    run_folder = utils.find_next_run_folder()

    # Plot and save class distribution
    utils.plot_class_distribution(train_dataset, run_folder)

    # Visualize and save a random grid of samples
    utils.visualize_random_sample_grid(train_dataset, run_folder, grid_size=4)

    model = BasicSegmentationModel(num_classes=23).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 101

    train_model(num_epochs, model, train_dataloader, validation_dataloader, criterion, optimizer, device, run_folder)
import torch
import cv2
import numpy as np
from basic import BasicSegmentationModel  # Assuming the model class is in model.py

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 23  # Set this to match the model used during training
model = BasicSegmentationModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load('runs/train30/last_model.pth'))
model.eval()

# Load and preprocess the image
image_path = 'basic_network/dataset/valid/images/car76_jpg.rf.7457f85f9c943b5f7ebf929901a31cb1.jpg'  # Replace with your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (320, 230))  # Resize to match the model's input size
image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0  # Normalize and add batch dimension
image_tensor = image_tensor.to(device)

# Run inference
with torch.no_grad():
    output = model(image_tensor)
    prediction = torch.argmax(output, dim=1).cpu().numpy()[0]  # Get the predicted class for each pixel

# Convert prediction to a color map (for 23 classes, example with random color assignments)
color_map = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
for cls in range(23):  # Assuming a max of 23 classes
    color = np.random.randint(0, 255, size=3, dtype=np.uint8)  # Generate random color for each class
    color_map[prediction == cls] = color

# Save the output image
output_path = 'basic_network/inferenced.jpg'  # Replace with your desired output path
cv2.imwrite(output_path, color_map)

print(f'Saved the segmented image at {output_path}')

from torchviz import make_dot
from megmentation.model import TransferLearningSegmentationModel
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TransferLearningSegmentationModel(num_classes=23).to(device)
model.load_state_dict(torch.load('runs/train7/best_model.pth'))

# Assuming 'model' is your model and 'x' is a sample input tensor
x = torch.randn(1, 3, 240, 320).to(device)  # Replace with the appropriate input size for your model
y = model(x)  # Forward pass

# Visualize the model
make_dot(y, params=dict(model.named_parameters())).render("model_architecture", format="png")

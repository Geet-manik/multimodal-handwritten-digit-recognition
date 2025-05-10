# Import required libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torchvision.datasets import MNIST

# Configure device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess UCI Pen-Based dataset for stroke features
pen_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra', header=None)
pen_features = pen_data.iloc[:, :-1].values
pen_labels = pen_data.iloc[:, -1].values

# Normalize pen features using StandardScaler
scaler = StandardScaler()
pen_features = scaler.fit_transform(pen_features)

# Define image transformations for MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Custom dataset class to align MNIST images with pen data
class MultimodalDataset(Dataset):
    def __init__(self, mnist_data, pen_features, pen_labels):
        self.mnist_data = mnist_data
        self.pen_features = pen_features
        self.pen_labels = pen_labels
        
        # Ensure same number of samples for both modalities
        self.length = min(len(mnist_data), len(pen_features))
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        image, _ = self.mnist_data[idx]  # Ignore MNIST label
        pen_feat = torch.FloatTensor(self.pen_features[idx])
        label = self.pen_labels[idx] % 10  # Convert UCI labels to 0-9 range
        
        return (image, pen_feat), label

# Create training datasets and dataloaders
mnist_train = MNIST(root='./data', train=True, download=True, transform=transform)
train_dataset = MultimodalDataset(mnist_train, pen_features[:60000], pen_labels[:60000])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create testing datasets and dataloaders
mnist_test = MNIST(root='./data', train=False, download=True, transform=transform)
test_dataset = MultimodalDataset(mnist_test, pen_features[60000:], pen_labels[60000:])
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define multimodal neural network architecture
class MultimodalDigitClassifier(nn.Module):
    def __init__(self):
        super(MultimodalDigitClassifier, self).__init__()
        
        # Image processing branch (CNN)
        self.image_net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*5*5, 128)
        )
        
        # Pen stroke features branch (MLP)
        self.pen_net = nn.Sequential(
            nn.Linear(16, 64),  # 16 pen features from UCI dataset
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Combined classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
    def forward(self, x_img, x_pen):
        # Process image through CNN branch
        img_features = self.image_net(x_img)
        
        # Process pen features through MLP branch
        pen_features = self.pen_net(x_pen)
        
        # Concatenate features from both modalities
        combined = torch.cat((img_features, pen_features), dim=1)
        
        # Final classification
        return self.classifier(combined)

# Initialize model, loss function, and optimizer
model = MultimodalDigitClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for (images, pen_features), labels in train_loader:
        # Move data to appropriate device
        images = images.to(device)
        pen_features = pen_features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images, pen_features)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Model evaluation on test set
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for (images, pen_features), labels in test_loader:
        # Move data to appropriate device
        images = images.to(device)
        pen_features = pen_features.to(device)
        labels = labels.to(device)
        
        # Forward pass and prediction
        outputs = model(images, pen_features)
        _, predicted = torch.max(outputs.data, 1)
        
        # Calculate accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Function to visualize model predictions
def visualize_predictions(num_samples=8):
    model.eval()
    (images, pen_features), labels = next(iter(test_loader))
    images, pen_features = images.to(device), pen_features.to(device)
    
    with torch.no_grad():
        outputs = model(images, pen_features)
        _, predicted = torch.max(outputs, 1)
    
    # Move tensors to CPU for visualization
    images = images.cpu()
    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 2))
    for i in range(num_samples):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f'True: {labels[i]}\nPred: {predicted[i]}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# Generate prediction visualizations
visualize_predictions()
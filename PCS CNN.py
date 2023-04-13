import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import numpy as np
import scipy.io as sio
import pandas as pd
import os 
from matplotlib import pyplot as plt

folder_path = ''
csv_path = ''
train_plot = []
test_plot = []

# Load the data from the .mat files
PCS = []

for filename in os.listdir(folder_path):
    if filename.endswith('.mat'):
        mat_data = sio.loadmat(os.path.join(folder_path, filename))
        mat_data = mat_data['Cavity'][7:18,7:18]
        PCS.append(mat_data)

PCS = np.array(PCS)

y_df = pd.read_csv(csv_path, header=None, skiprows=1, usecols=[1, 2])
amps = np.array(y_df)

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=2, dilation = 1, stride = 2, padding = 2)
        self.pool = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(128, 32, 3, padding=1)
        self.fc1 = nn.Linear(8 * 2 * 2, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 8 * 2 * 2)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
   
PCS_tensor = torch.tensor(PCS, dtype=torch.float32)
amps_tensor = torch.tensor(amps, dtype=torch.float32)

# Combine PCS and amps tensors into a PyTorch dataset
dataset = TensorDataset(PCS_tensor, amps_tensor)

# Define the sizes of the training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Use random_split to split the dataset into training and testing sets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size = 1, shuffle=True)

x_train = []
y_train = []
for data in train_loader:
    traininputs, trainlabels = data
    x_train.append(traininputs)
    y_train.append(trainlabels)
x_train = torch.cat(x_train, dim=0)
y_train = torch.cat(y_train, dim=0)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

x_test = []
y_test = []

for data in test_loader:
    testinputs, testlabels = data
    x_test.append(testinputs)
    y_test.append(testlabels)
x_test = torch.cat(x_test, dim=0)
y_test = torch.cat(y_test, dim=0)


model = MyCNN()

optimizer = optim.Adagrad(model.parameters(), lr=0.0001)
epochs = 50

def custom_loss(output, target):
    # Calculate mean absolute error
    mae = torch.mean(torch.abs(output - target))
    # Check if absolute difference between values is less than or equal to 1
    within_range = torch.abs(output - target) <= 2.5
    # Compute mean of binary within_range values (1 if within range, 0 if not)
    accuracy = torch.mean(within_range.float())
    # Return combined loss and accuracy as a tuple
    return mae, accuracy

for epoch in range(epochs):
    # Set model to train mode
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss, acc = custom_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        train_acc += acc.item() * inputs.size(0)
    
        
    # Set model to eval mode
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    
    with torch.no_grad():
        for val_inputs, val_targets in test_loader:
            val_outputs = model(val_inputs)
            loss, acc = custom_loss(val_outputs, val_targets)
            val_loss += loss.item() * val_inputs.size(0)
            val_acc += acc.item() * val_inputs.size(0)
    
    # Calculate average losses and accuracies for epoch
    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_acc / len(train_loader.dataset)
    val_loss = val_loss / len(test_loader.dataset)
    val_acc = val_acc / len(test_loader.dataset)
    
    # Print results for epoch
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")


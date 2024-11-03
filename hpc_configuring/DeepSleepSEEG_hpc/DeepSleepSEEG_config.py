import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
from model_DeepSleepSEEG import DeepSleepSEEG  # Assuming this is your model class

# Paths to the dataset and output directory
train_data_path = '/gpfsnyu/scratch/rw3045/train_data_noCoordinates.npz'
test_data_path = '/gpfsnyu/scratch/rw3045/test_data_noCoordinates.npz'
output_dir = '/gpfsnyu/home/rw3045/DeepSleepSEEG/Baseline'

# Load the datasets
train_data = np.load(train_data_path)
test_data = np.load(test_data_path)

X_train = train_data['X']
y_train = train_data['y']
X_test = test_data['X']
y_test = test_data['y']
X_train = X_train[:,0:3000]
X_test = X_test[:,0:3000]
X_train = X_train.reshape(4576, 1, 3000)
X_test = X_test.reshape(1144, 1, 3000)

# Initialize the model
model = DeepSleepSEEG()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()  # Use appropriate loss function for your task

# Training loop (simplified)
num_epochs = 10000
loss_overtime = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(torch.tensor(X_train,dtype=torch.float).to(device))
    loss = criterion(outputs, torch.tensor(y_train).to(device))
    loss_overtime.append(loss)
    loss.backward()
    optimizer.step()

    # Optionally print progress
    if epoch % 1 == 100:
        print('Epoch {}, Loss: {}'.format(epoch,loss.item()))

# Save loss over time
output_file = f"{output_dir}/loss_overtime.csv"
pd.DataFrame(loss_overtime, columns=['Loss']).to_csv(output_file, index=False)

# Evaluate the model on the test data
model.eval()
with torch.no_grad():
    outputs = model(torch.tensor(X_test,dtype=torch.float).to(device))
    _, predicted = torch.max(outputs, 1)
    f1 = f1_score(y_test, predicted.cpu(), average='weighted')
    accuracy = accuracy_score(y_test, predicted.cpu())

print(f'Test F1 Score: {f1:.2f}')
print(f'Test Accuracy: {accuracy:.2f}')

# Save the trained model
torch.save(model.state_dict(), f'{output_dir}/trained_model.pth')
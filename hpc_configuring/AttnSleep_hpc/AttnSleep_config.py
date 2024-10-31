import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
from Re_AttnSleep import AttnSleep  # Assuming this is your model class

# Paths to the dataset and output directory
train_data_path = '/Volumes/ruoyu_hd/Projects/DScapstone/Sleep/Data/mni_sEEG/train_data_noCoordinates.npz'
test_data_path = '/Volumes/ruoyu_hd/Projects/DScapstone/Sleep/Data/mni_sEEG/test_data_noCoordinates.npz'
output_dir = '/Volumes/ruoyu_hd/Projects/DScapstone/Sleep/Data/mni_sEEG'

# Load the datasets
train_data = np.load(train_data_path)
test_data = np.load(test_data_path)

X_train = train_data['X']
y_train = train_data['y']
X_test = test_data['X']
y_test = test_data['y']

X_train = X_train.reshape(4576, 1, 6800)
X_test = X_test.reshape(1144, 1, 6800)

# Initialize the model
model = AttnSleep()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()  # Use appropriate loss function for your task

# Training loop (simplified)
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    print('start successfully')
    optimizer.zero_grad()
    outputs = model(torch.tensor(X_train,dtype=torch.float).to(device))
    print('gained output succ')
    loss = criterion(outputs, torch.tensor(y_train).to(device))
    loss.backward()
    print('back propagate succ')
    optimizer.step()

    # Optionally print progress
    if epoch % 1 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

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
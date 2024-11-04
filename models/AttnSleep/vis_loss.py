import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('models/AttnSleep/loss_overtime.csv')

# Plot the loss over time
plt.figure(figsize=(10, 6))
plt.plot(data.index[0:600], data['Loss'][0:600], marker='o', linestyle='-')
plt.title('Loss Over Time - AttnSleep')
plt.xlabel('Time Step')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

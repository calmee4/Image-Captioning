import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data from the training logs
epochs = np.arange(1, 21)
bleu_1 = [89.4929, 91.5893, 90.5615, 92.5640, 91.4698, 91.7916, 88.5642, 88.9963, 91.3246, 87.1121,
          87.4577, 89.4832, 89.7964, 87.6049, 29.9780, 35.5347, 34.0182, 34.9745, 34.3423, 35.1611]
bleu_2 = [25.3611, 56.7853, 64.1193, 77.1609, 77.6462, 78.7538, 76.0273, 76.6990, 79.9812, 75.4841,
          75.8795, 78.0005, 78.5978, 76.0879, 15.4508, 18.0405, 17.3797, 17.6079, 17.1368, 17.7560]
bleu_3 = [17.7608, 46.5854, 53.0855, 65.3780, 66.5715, 66.6943, 63.9968, 64.3400, 68.1833, 63.6489,
          64.3490, 66.2896, 67.1984, 64.6811, 9.1702, 10.0206, 9.8161, 9.7042, 9.3910, 9.8273]
bleu_4 = [12.8482, 38.9812, 44.7563, 56.2250, 57.6884, 57.0738, 54.6444, 54.9141, 58.8194, 54.3486,
          55.2632, 57.1625, 58.1617, 55.8004, 5.3953, 3.2311, 4.1657, 2.9129, 2.8828, 3.1071]

# Create a DataFrame for easy plotting
data = pd.DataFrame({
    'Epoch': epochs,
    'BLEU-1': bleu_1,
    'BLEU-2': bleu_2,
    'BLEU-3': bleu_3,
    'BLEU-4': bleu_4
})

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(data['Epoch'], data['BLEU-1'], label='BLEU-1', marker='o', linestyle='-', color='b', markersize=6)
plt.plot(data['Epoch'], data['BLEU-2'], label='BLEU-2', marker='s', linestyle='-', color='g', markersize=6)
plt.plot(data['Epoch'], data['BLEU-3'], label='BLEU-3', marker='^', linestyle='-', color='r', markersize=6)
plt.plot(data['Epoch'], data['BLEU-4'], label='BLEU-4', marker='d', linestyle='-', color='m', markersize=6)

# Enhancements for a cleaner plot
plt.title('BLEU Scores During Training', fontsize=16)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('BLEU Score', fontsize=12)
plt.xticks(np.arange(1, 21, 1))
plt.yticks(np.arange(0, 100, 10))
plt.grid(True, axis='both', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()

import matplotlib.pyplot as plt

# Data from your log (train_loss and val_loss for 20 epochs)
epochs = list(range(1, 21))  # 20 epochs
train_losses = [
    4.4187, 3.7212, 3.4677, 3.2871, 3.1358, 2.9990,  # Epochs 1-6
    21.2240, 9.1926, 6.6010, 5.5155, 4.8443, 4.3660,  # Epochs 7-12
    4.0631, 3.8116, 3.6199, 3.4566, 3.3111, 3.2016,  # Epochs 13-18
    3.1439, 3.2062  # Epochs 19-20
]
val_losses = [
    3.9487, 3.7494, 3.6663, 3.6385, 3.6322, 3.6486,  # Epochs 1-6
    9.0561, 6.1342, 5.1001, 4.8449, 4.5499, 4.4753,  # Epochs 7-12
    4.2283, 4.2319, 4.2087, 4.1533, 4.1565, 4.1207,  # Epochs 13-18
    4.1135, 4.2263  # Epochs 19-20
]

# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(epochs, train_losses, label='Training Loss', color='blue', linestyle='-', marker='o')
plt.plot(epochs, val_losses, label='Validation Loss', color='red', linestyle='--', marker='x')

# Adding labels and title
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()

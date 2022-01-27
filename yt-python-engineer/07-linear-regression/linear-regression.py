# 0) Prepare data
# 1) Design model (input size, output size, forward pass (operations, layers))
# 2) Construct loss and optimizer
# 3) Training loop
#       - forward pass: compute prediction
#       - backward pass: gradients
#       - update weights


import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


# Data
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

_x = torch.from_numpy(x_numpy.astype(np.float32))
_y = torch.from_numpy(y_numpy.astype(np.float32))
_y = _y.view(_y.shape[0], 1)                        # reshape _y into a column vector

n_samples, n_features = _x.shape


# Model
input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)


# Loss and Optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    # forward pass and loss
    y_predicted = model(_x)
    loss = criterion(y_predicted, _y)

    # backward pass
    loss.backward()

    # update
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f"epoch: {epoch + 1}, loss = {loss.item():.4}")


# Plot
predicted = y_predicted.detach().numpy()

plt.plot(x_numpy, y_numpy, 'ro')
plt.plot(x_numpy, predicted, 'b')
plt.show()
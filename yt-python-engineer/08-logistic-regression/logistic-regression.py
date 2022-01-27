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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Data
bc = datasets.load_breast_cancer()
_x, _y = bc.data, bc.target

n_samples, n_features = _x.shape

_x_train, _x_test, _y_train, _y_test = train_test_split(_x, _y, test_size=0.2, random_state=1234)

# scaling the features
sc = StandardScaler()
_x_train = sc.fit_transform(_x_train)
_x_test = sc.transform(_x_test)

_x_train = torch.from_numpy(_x_train.astype(np.float32))
_x_test = torch.from_numpy(_x_test.astype(np.float32))
_y_train = torch.from_numpy(_y_train.astype(np.float32))
_y_test = torch.from_numpy(_y_test.astype(np.float32))

_y_train = _y_train.view(_y_train.shape[0], 1)
_y_test = _y_test.view(_y_test.shape[0], 1)


# Model
# f = wx + b, sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features) -> None:
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)


# Loss and Optimizer
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# Training loop
n_epochs = 2000
for epoch in range(n_epochs):
    # forward pass and loss
    y_predicted = model(_x_train)
    loss = criterion(y_predicted, _y_train)

    # backward pass
    loss.backward()

    # updates
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 200 == 0:
        print(f"epoch: {epoch + 1}, loss = {loss.item():.4f}")

with torch.no_grad():
    y_predicted = model(_x_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(_y_test).sum() / float(_y_test.shape[0])
    print(f"accuracy = {acc:.4f}")

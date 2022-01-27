# 1) Design model (input size, output size, forward pass (operations, layers))
# 2) Construct loss and optimizer
# 3) Training loop
#       - forward pass: compute prediction
#       - backward pass: gradients
#       - update weights


import torch
import torch.nn as nn


# f = w * x

_x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)    # SHAPE CHANGE for model !!!
_y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = _x.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

# model = nn.Linear(input_size, output_size)    # Prebuilt model provided for us


class LinearRegression(nn.Module):              # Custom model

    def __init__(self, input_dim, output_dim) -> None:
        super(LinearRegression, self).__init__()
        # Define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


model = LinearRegression(input_size, output_size)


print(f"Prediction before training: f(5) = {model(X_test).item():.3f}")


# Training
learning_rate = 0.01
n_iters = 2500


loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)   # Stochastic Gradient Descent


for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(_x)

    # loss
    l = loss(_y, y_pred)

    # gradients = backward pass
    l.backward()    # dl/dw

    # update weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if epoch % 250 == 0:
        [w, b] = model.parameters()
        print(f"epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}")


print(f"Prediction after training: f(5) = {model(X_test).item():.3f}")
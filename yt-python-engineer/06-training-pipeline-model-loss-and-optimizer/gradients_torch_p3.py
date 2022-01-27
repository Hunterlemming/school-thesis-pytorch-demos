# 1) Design model (input size, output size, forward pass (operations, layers))
# 2) Construct loss and optimizer
# 3) Training loop
#       - forward pass: compute prediction
#       - backward pass: gradients
#       - update weights


import torch
import torch.nn as nn


# f = w * x

_x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
_y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

_w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


# model prediction
def forward(x):
    return _w * x

print(f"Prediction before training: f(5) = {forward(5):.3f}")


# Training
learning_rate = 0.01
n_iters = 100


loss = nn.MSELoss()
optimizer = torch.optim.SGD([_w], lr=learning_rate)   # Stochastic Gradient Descent


for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(_x)

    # loss
    l = loss(_y, y_pred)

    # gradients = backward pass
    l.backward()    # dl/dw

    # update weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f"epoch {epoch + 1}: w = {_w:.3f}, loss = {l:.8f}")


print(f"Prediction after training: f(5) = {forward(5):.3f}")
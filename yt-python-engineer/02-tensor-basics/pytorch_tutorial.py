import torch


# Creating tensors and adding tensors (sub, mul, div works the same if possible)

x = torch.tensor([[1, 2],[3, 4]])
y = torch.ones(2,2) * 2

z = torch.add(x, y)     # creating a new variable with the result
y.add_(x)               # overriding the initial value /wtr


# Slicing tensors

x = torch.rand(5, 3)

x_a0 = x[:, 0]          # first column in all the rows
x_1a = x[1, :]          # all columns in the second row
x_10 = x[1, 0].item()   # first column of the second row (element)


# Reshaping tensors

x = torch.rand(4, 4)
y = x.view(16)          # Setting the number of elements directly 
y = x.view(-1, 8)       # Reducing the number of dimensions in the first parameter


# Converting between numpy and pytorch tensors

import numpy as np

a = torch.ones(5)
b = a.numpy()

a.add_(1)               # if using cpu, both point to the same memory location -- both a and b are modified

a = np.ones(5)
b = torch.from_numpy(a)

a += 1


# Optimizing

# Whenever you have a variable in your model that you want to optimize, then you need the gradients => specify it as such 
x = torch.ones(5, requires_grad=True)
print(x)
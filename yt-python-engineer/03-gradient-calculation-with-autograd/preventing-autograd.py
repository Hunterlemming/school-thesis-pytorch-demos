# Preventing the tracing of history and calculating grad_fn-s
# ex.: updating weights

import torch


random_seed = 1997


torch.manual_seed(random_seed)
x = torch.randn(3, requires_grad=True)
print(x)

# x.requires_grad_(False)   # Overriding the requires_grad attribute
# print(x)

# y = x.detach()            # Creating a new vector with the same values, but it doesn't require the gradient
# print(y)

with torch.no_grad():       # Declaring a block, where we don't use it
    y = x + 2
    print(y)

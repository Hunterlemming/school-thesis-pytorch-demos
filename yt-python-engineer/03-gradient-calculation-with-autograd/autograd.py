import torch


random_seed = 1997


torch.manual_seed(random_seed)
x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2       # AddBackward gradient function
print(y)

z = y * y * 2   # MulBackward gradient function
print(z)

# z = z.mean()    # MeanBackward gradient function
# z.backward()    # dz/dx : gradient of z with respect to x - can only work on scalar outputs without parameter
# print(x.grad)

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v)   # dz/dx - needs a vector with the same sape as z
print(x.grad)



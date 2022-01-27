from operator import mod
import torch


manual_opt = True


weights = torch.ones(4, requires_grad=True)


if manual_opt:
    for epoch in range(3):
        model_output = (weights * 3).sum()

        model_output.backward()

        print(model_output)
        print(weights.grad)

        weights.grad.zero_()    # if we leave this out the weights' grad will not empty itself
else:
    optimizer = torch.optim.SGD(weights, lr=0.01)
    optimizer.step()
    optimizer.zero_grad()       # same as above

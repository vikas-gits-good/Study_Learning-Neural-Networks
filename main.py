from src.micrograd.engine import Value

from src.micrograd.nn import MLP
# from src.micrograd.nn_2 import MLP
# from src.micrograd.nn_3 import MLP

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
y_true = [1.0, -1.0, -1.0, 1.0]  # desired targets


nn = MLP(3, [4, 4, 1])
print(len(nn.parameters()))

for k in range(100):
    # forward pass
    y_pred = [nn(x) for x in xs]
    loss: Value = sum((yp - yt) ** 2 for yt, yp in zip(y_true, y_pred))

    # backward pass
    for p in nn.parameters():
        p.grad = 0.0
    loss.backward()

    # update params
    for p in nn.parameters():
        p.data += -0.025 * p.grad

    print(k, loss.data)

import random
from typing import List
from src.micrograd.engine import Value


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin: int = 5, nonlin: bool = True) -> None:
        super().__init__()
        self.w = [Value(data=random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(data=random.uniform(-1, 1))  # 0.0)  #
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # act = reduce(operator.add, (wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"Neuron(Activation={'ReLU' if self.nonlin else 'Linear'}, Weights={self.w}, Bias={self.b})"


class Layer(Module):
    def __init__(self, nin: int = 5, nout: int = 4, **kwargs) -> None:
        super().__init__()
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def repr(self):
        return f"NN Layer of [{', '.join([str(n) for n in self.neurons])}]"


class MLP(Module):
    def __init__(self, nin: int = 5, nouts: List[int] = [4, 4, 1]) -> None:
        super().__init__()
        sz = [nin] + nouts
        self.layers = [
            Layer(nin=sz[i], nout=sz[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join([str(layer) for layer in self.layers])}]"

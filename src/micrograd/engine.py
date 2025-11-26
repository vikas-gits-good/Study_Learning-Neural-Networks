import math


class Value:
    def __init__(
        self,
        data: float = 0.0,
        _children: tuple = (),
        _op: str = "",
    ) -> None:
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
        self.grad = 0.0

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        func = self.data + other.data
        out = Value(data=func, _children=(self, other), _op="+")

        def _derivative():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _derivative
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        func = self.data * other.data
        out = Value(data=func, _children=(self, other), _op="*")

        def _derivative():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _derivative
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), (
            "only supporting int/float powers for now"
        )
        func = self.data**other
        out = Value(data=func, _children=(self, other), _op=f"**{other}")

        def _derivative():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _derivative
        return out

    def exp(self):
        func = math.exp(self.data)
        out = Value(data=func, _children=(self,), _op="exp")

        def _derivative():
            self.grad += out.data * out.grad

        out._backward = _derivative
        return out

    def tanh(self):
        func = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Value(data=func, _children=(self,), _op="tanh")

        def _derivative():
            self.grad += (1 - func**2) * out.grad

        out._backward = _derivative
        return out

    def sigmoid(self):
        func = 1 / (1 + math.exp(-self.data))
        out = Value(data=func, _children=(self,), _op="sigmoid")

        def _derivative():
            self.grad += func * (1 - func) * out.grad

        out._backward = _derivative
        return out

    def relu(self):
        func = 0 if self.data <= 0 else self.data
        out = Value(data=func, _children=(self,), _op="relu")

        def _derivative():
            self.grad += 0 * out.grad if self.data < 0 else 1 * out.grad

        out._backward = _derivative
        return out

    def leaky_relu(self):
        func = 0.01 * self.data if self.data <= 0 else self.data
        out = Value(data=func, _children=(self,), _op="leaky_relu")

        def _derivative():
            self.grad += 0.01 * out.grad if self.data < 0 else 1 * out.grad

        out._backward = _derivative
        return out

    def elu(self, alpha: float = 0.02):
        func = alpha * (math.exp(self.data) - 1) if self.data <= 0 else self.data
        out = Value(data=func, _children=(self,), _op="elu")

        def _derivative():
            self.grad += (
                alpha * math.exp(self.data) * out.grad
                if self.data < 0
                else 1 * out.grad
            )

        out._backward = _derivative
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(val):
            if val not in visited:
                visited.add(val)
                for child in val._prev:
                    build_topo(child)
                topo.append(val)

        build_topo(self)

        self.grad = 1
        for node in reversed(topo):
            node._backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

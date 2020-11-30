import math


class Value:
    """Represents a scalar with a value and derivative (grad)."""
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0

        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, children=(self, other), op='+')
        def _add_backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _add_backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, children=(self, other), op='*')
        def _mul_backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _mul_backward
        return out

    def relu(self):
        out = Value(self.data if self.data >= 0 else 0, (self,), op='ReLU')
        def _relu_backward():
            self.grad += (self.data >= 0) * out.grad
        out._backward = _relu_backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # perform a topological sort on the children
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # apply chain rule
        self.grad = 1  # dz/dz = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self):
        return self * -1

    def __radd(self, other):  # what to do when clients calls 4.__add(Value)
        return self + other

    def __rmul(self, other):  # what to do when call 4.__mul__(Value)
        return self * other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"




class Value:
    def __init__(self, data, parents=(), operation=""):
        self.data = float(data)
        self.grad = 0.0
        self.parents = set(parents)
        self.operation = operation
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        result = Value(
            self.data + other.data,
            parents=(self, other),
            operation="+",
        )

        def backward():
            self.grad += result.grad
            other.grad += result.grad

        result._backward = backward
        return result

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        result = Value(
            self.data * other.data,
            parents=(self, other),
            operation="*",
        )

        def backward():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad

        result._backward = backward
        return result

    def __pow__(self, exponent):
        result = Value(
            self.data ** exponent,
            parents=(self,),
            operation=f"**{exponent}",
        )

        def backward():
            self.grad += (
                exponent
                * (self.data ** (exponent - 1))
                * result.grad
            )

        result._backward = backward
        return result

    def relu(self):
        result = Value(
            max(0.0, self.data),
            parents=(self,),
            operation="ReLU",
        )

        def backward():
            derivative = 1.0 if self.data > 0 else 0.0
            self.grad += derivative * result.grad

        result._backward = backward
        return result

    def backward(self):
        order = []
        visited = set()

        def visit(node):
            if node not in visited:
                visited.add(node)

                for parent in node.parents:
                    visit(parent)

                order.append(node)

        visit(self)

        # Reset gradients before this backward pass.
        for node in order:
            node.grad = 0.0

        # The output's gradient starts at 1.
        self.grad = 1.0

        # Visit nodes from output back toward the inputs.
        for node in reversed(order):
            node._backward()

    def sigmoid(self):
        import math

        output = 1.0 / (1.0 + math.exp(-self.data))
        result = Value(
            output,
            parents=(self,),
            operation="sigmoid",
        )

        def backward():
            self.grad += output * (1.0 - output) * result.grad

        result._backward = backward
        return result

    def log(self):
        import math

        result = Value(
            math.log(self.data),
            parents=(self,),
            operation="log",
        )

        def backward():
            self.grad += (1.0 / self.data) * result.grad

        result._backward = backward
        return result

def binary_cross_entropy(probability, target):
    target = Value(float(target))
    one = Value(1.0)

    loss = -(
        target * probability.log()
        + (one - target) * (one - probability).log()
    )
    return loss

if __name__ == "__main__":
    p = Value(0.8)
    loss = binary_cross_entropy(p, 1)
    loss.backward()

    print("BCE loss:", loss.data)  # about 0.2231
    print("Gradient:", p.grad)     # -1.25

    x = Value(3)
    y = Value(1)
    z = x - y
    z.backward()
    print(z.data, x.grad, y.grad)  # 2.0, 1.0, -1.0

    x_log = Value(2)
    y_log = x_log.log()
    y_log.backward()

    print("Log output:", y_log.data)      # about 0.6931
    print("Log gradient:", x_log.grad)    # 0.5

    x = Value(0)
    y = x.sigmoid()
    y.backward()

    print("Sigmoid output:", y.data)  # 0.5
    print("Sigmoid gradient:", x.grad)  # 0.25

    import torch
    import torch.nn.functional as F

    # Your autodiff engine
    a = Value(3)
    b = Value(2)
    c = Value(1)

    out = ((a * b + c) ** 2).relu()
    out.backward()

    # PyTorch comparison
    a_t = torch.tensor(3.0, requires_grad=True)
    b_t = torch.tensor(2.0, requires_grad=True)
    c_t = torch.tensor(1.0, requires_grad=True)

    out_t = F.relu((a_t * b_t + c_t) ** 2)
    out_t.backward()

    print("Your output:", out.data)
    print("PyTorch output:", out_t.item())

    print("Your gradients:", a.grad, b.grad, c.grad)
    print(
        "PyTorch gradients:",
        a_t.grad.item(),
        b_t.grad.item(),
        c_t.grad.item(),
    )

    # Numerical-gradient check using central differences
    def calculate_output(a_value, b_value, c_value):
        return max(0.0, (a_value * b_value + c_value) ** 2)

    epsilon = 1e-5

    numerical_a = (
        calculate_output(3 + epsilon, 2, 1)
        - calculate_output(3 - epsilon, 2, 1)
    ) / (2 * epsilon)

    numerical_b = (
        calculate_output(3, 2 + epsilon, 1)
        - calculate_output(3, 2 - epsilon, 1)
    ) / (2 * epsilon)

    numerical_c = (
        calculate_output(3, 2, 1 + epsilon)
        - calculate_output(3, 2, 1 - epsilon)
    ) / (2 * epsilon)

    differences = [
        abs(a.grad - numerical_a),
        abs(b.grad - numerical_b),
        abs(c.grad - numerical_c),
    ]

    print("Numerical gradients:", numerical_a, numerical_b, numerical_c)
    print("Absolute differences:", differences)
    print("Maximum difference:", max(differences))

    assert max(differences) < 1e-6, "Numerical gradient check failed"
    print("Numerical gradient check passed")
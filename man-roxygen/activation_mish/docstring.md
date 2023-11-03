Mish activation function.

It is defined as:

`mish(x) = x * tanh(softplus(x))`

where `softplus` is defined as:

`softplus(x) = log(exp(x) + 1)`

Args:
    x: Input tensor.

Reference:

- [Misra, 2019](https://arxiv.org/abs/1908.08681)

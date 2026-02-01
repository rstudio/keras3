# Hard SiLU activation function, also known as Hard Swish.

It is defined as:

- `0` if `if x < -3`

- `x` if `x > 3`

- `x * (x + 3) / 6` if `-3 <= x <= 3`

It's a faster, piecewise linear approximation of the silu activation.

## Usage

``` r
activation_hard_silu(x)

activation_hard_swish(x)
```

## Arguments

- x:

  Input tensor.

## Value

A tensor, the result from applying the activation to the input tensor
`x`.

## Reference

- [A Howard, 2019](https://arxiv.org/abs/1905.02244)

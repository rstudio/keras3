# Continuously Differentiable Exponential Linear Unit.

The CeLU activation function is defined as:

`celu(x) = alpha * (exp(x / alpha) - 1) for x < 0`,`celu(x) = x for x >= 0`.

where `alpha` is a scaling parameter that controls the activation's
shape.

## Usage

``` r
activation_celu(x, alpha = 1)
```

## Arguments

- x:

  Input tensor.

- alpha:

  The value for the CeLU formulation. Defaults to `1.0`.

## Value

A tensor, the result from applying the activation to the input tensor
`x`.

## Reference

- [Barron, J. T., 2017](https://arxiv.org/abs/1704.07483)

## See also

Other activations:  
[`activation_elu()`](https://keras3.posit.co/dev/reference/activation_elu.md)  
[`activation_exponential()`](https://keras3.posit.co/dev/reference/activation_exponential.md)  
[`activation_gelu()`](https://keras3.posit.co/dev/reference/activation_gelu.md)  
[`activation_glu()`](https://keras3.posit.co/dev/reference/activation_glu.md)  
[`activation_hard_shrink()`](https://keras3.posit.co/dev/reference/activation_hard_shrink.md)  
[`activation_hard_sigmoid()`](https://keras3.posit.co/dev/reference/activation_hard_sigmoid.md)  
[`activation_hard_tanh()`](https://keras3.posit.co/dev/reference/activation_hard_tanh.md)  
[`activation_leaky_relu()`](https://keras3.posit.co/dev/reference/activation_leaky_relu.md)  
[`activation_linear()`](https://keras3.posit.co/dev/reference/activation_linear.md)  
[`activation_log_sigmoid()`](https://keras3.posit.co/dev/reference/activation_log_sigmoid.md)  
[`activation_log_softmax()`](https://keras3.posit.co/dev/reference/activation_log_softmax.md)  
[`activation_mish()`](https://keras3.posit.co/dev/reference/activation_mish.md)  
[`activation_relu()`](https://keras3.posit.co/dev/reference/activation_relu.md)  
[`activation_relu6()`](https://keras3.posit.co/dev/reference/activation_relu6.md)  
[`activation_selu()`](https://keras3.posit.co/dev/reference/activation_selu.md)  
[`activation_sigmoid()`](https://keras3.posit.co/dev/reference/activation_sigmoid.md)  
[`activation_silu()`](https://keras3.posit.co/dev/reference/activation_silu.md)  
[`activation_soft_shrink()`](https://keras3.posit.co/dev/reference/activation_soft_shrink.md)  
[`activation_softmax()`](https://keras3.posit.co/dev/reference/activation_softmax.md)  
[`activation_softplus()`](https://keras3.posit.co/dev/reference/activation_softplus.md)  
[`activation_softsign()`](https://keras3.posit.co/dev/reference/activation_softsign.md)  
[`activation_sparse_plus()`](https://keras3.posit.co/dev/reference/activation_sparse_plus.md)  
[`activation_sparse_sigmoid()`](https://keras3.posit.co/dev/reference/activation_sparse_sigmoid.md)  
[`activation_sparsemax()`](https://keras3.posit.co/dev/reference/activation_sparsemax.md)  
[`activation_squareplus()`](https://keras3.posit.co/dev/reference/activation_squareplus.md)  
[`activation_tanh()`](https://keras3.posit.co/dev/reference/activation_tanh.md)  
[`activation_tanh_shrink()`](https://keras3.posit.co/dev/reference/activation_tanh_shrink.md)  
[`activation_threshold()`](https://keras3.posit.co/dev/reference/activation_threshold.md)  

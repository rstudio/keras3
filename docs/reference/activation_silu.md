# Swish (or Silu) activation function.

It is defined as: `swish(x) = x * sigmoid(x)`.

The Swish (or Silu) activation function is a smooth, non-monotonic
function that is unbounded above and bounded below.

## Usage

``` r
activation_silu(x)
```

## Arguments

- x:

  Input tensor.

## Value

A tensor, the result from applying the activation to the input tensor
`x`.

## Reference

- [Ramachandran et al., 2017](https://arxiv.org/abs/1710.05941)

## See also

- <https://keras.io/api/layers/activations#silu-function>

Other activations:  
[`activation_celu()`](https://keras3.posit.co/reference/activation_celu.md)  
[`activation_elu()`](https://keras3.posit.co/reference/activation_elu.md)  
[`activation_exponential()`](https://keras3.posit.co/reference/activation_exponential.md)  
[`activation_gelu()`](https://keras3.posit.co/reference/activation_gelu.md)  
[`activation_glu()`](https://keras3.posit.co/reference/activation_glu.md)  
[`activation_hard_shrink()`](https://keras3.posit.co/reference/activation_hard_shrink.md)  
[`activation_hard_sigmoid()`](https://keras3.posit.co/reference/activation_hard_sigmoid.md)  
[`activation_hard_tanh()`](https://keras3.posit.co/reference/activation_hard_tanh.md)  
[`activation_leaky_relu()`](https://keras3.posit.co/reference/activation_leaky_relu.md)  
[`activation_linear()`](https://keras3.posit.co/reference/activation_linear.md)  
[`activation_log_sigmoid()`](https://keras3.posit.co/reference/activation_log_sigmoid.md)  
[`activation_log_softmax()`](https://keras3.posit.co/reference/activation_log_softmax.md)  
[`activation_mish()`](https://keras3.posit.co/reference/activation_mish.md)  
[`activation_relu()`](https://keras3.posit.co/reference/activation_relu.md)  
[`activation_relu6()`](https://keras3.posit.co/reference/activation_relu6.md)  
[`activation_selu()`](https://keras3.posit.co/reference/activation_selu.md)  
[`activation_sigmoid()`](https://keras3.posit.co/reference/activation_sigmoid.md)  
[`activation_soft_shrink()`](https://keras3.posit.co/reference/activation_soft_shrink.md)  
[`activation_softmax()`](https://keras3.posit.co/reference/activation_softmax.md)  
[`activation_softplus()`](https://keras3.posit.co/reference/activation_softplus.md)  
[`activation_softsign()`](https://keras3.posit.co/reference/activation_softsign.md)  
[`activation_sparse_plus()`](https://keras3.posit.co/reference/activation_sparse_plus.md)  
[`activation_sparse_sigmoid()`](https://keras3.posit.co/reference/activation_sparse_sigmoid.md)  
[`activation_sparsemax()`](https://keras3.posit.co/reference/activation_sparsemax.md)  
[`activation_squareplus()`](https://keras3.posit.co/reference/activation_squareplus.md)  
[`activation_tanh()`](https://keras3.posit.co/reference/activation_tanh.md)  
[`activation_tanh_shrink()`](https://keras3.posit.co/reference/activation_tanh_shrink.md)  
[`activation_threshold()`](https://keras3.posit.co/reference/activation_threshold.md)  

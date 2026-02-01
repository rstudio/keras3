# Sparsemax activation function.

For each batch `i`, and class `j`, sparsemax activation function is
defined as:

`sparsemax(x)[i, j] = max(x[i, j] - (x[i, :]), 0).`

## Usage

``` r
activation_sparsemax(x, axis = -1L)
```

## Arguments

- x:

  Input tensor.

- axis:

  `int`, axis along which the sparsemax operation is applied. (1-based)

## Value

A tensor, output of sparsemax transformation. Has the same type and
shape as `x`.

## Reference

- [Martins et.al., 2016](https://arxiv.org/abs/1602.02068)

## See also

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
[`activation_silu()`](https://keras3.posit.co/reference/activation_silu.md)  
[`activation_soft_shrink()`](https://keras3.posit.co/reference/activation_soft_shrink.md)  
[`activation_softmax()`](https://keras3.posit.co/reference/activation_softmax.md)  
[`activation_softplus()`](https://keras3.posit.co/reference/activation_softplus.md)  
[`activation_softsign()`](https://keras3.posit.co/reference/activation_softsign.md)  
[`activation_sparse_plus()`](https://keras3.posit.co/reference/activation_sparse_plus.md)  
[`activation_sparse_sigmoid()`](https://keras3.posit.co/reference/activation_sparse_sigmoid.md)  
[`activation_squareplus()`](https://keras3.posit.co/reference/activation_squareplus.md)  
[`activation_tanh()`](https://keras3.posit.co/reference/activation_tanh.md)  
[`activation_tanh_shrink()`](https://keras3.posit.co/reference/activation_tanh_shrink.md)  
[`activation_threshold()`](https://keras3.posit.co/reference/activation_threshold.md)  

# Softmax converts a vector of values to a probability distribution.

The elements of the output vector are in range `[0, 1]` and sum to 1.

Each input vector is handled independently. The `axis` argument sets
which axis of the input the function is applied along.

Softmax is often used as the activation for the last layer of a
classification network because the result could be interpreted as a
probability distribution.

The softmax of each vector x is computed as `exp(x) / sum(exp(x))`.

The input values in are the log-odds of the resulting probability.

## Usage

``` r
activation_softmax(x, axis = -1L)
```

## Arguments

- x:

  Input tensor.

- axis:

  Integer, axis along which the softmax is applied.

## Value

A tensor, the result from applying the activation to the input tensor
`x`.

## See also

- <https://keras.io/api/layers/activations#softmax-function>

Other activations:  
[`activation_celu()`](https://keras3.posit.co/dev/reference/activation_celu.md)  
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
[`activation_softplus()`](https://keras3.posit.co/dev/reference/activation_softplus.md)  
[`activation_softsign()`](https://keras3.posit.co/dev/reference/activation_softsign.md)  
[`activation_sparse_plus()`](https://keras3.posit.co/dev/reference/activation_sparse_plus.md)  
[`activation_sparse_sigmoid()`](https://keras3.posit.co/dev/reference/activation_sparse_sigmoid.md)  
[`activation_sparsemax()`](https://keras3.posit.co/dev/reference/activation_sparsemax.md)  
[`activation_squareplus()`](https://keras3.posit.co/dev/reference/activation_squareplus.md)  
[`activation_tanh()`](https://keras3.posit.co/dev/reference/activation_tanh.md)  
[`activation_tanh_shrink()`](https://keras3.posit.co/dev/reference/activation_tanh_shrink.md)  
[`activation_threshold()`](https://keras3.posit.co/dev/reference/activation_threshold.md)  

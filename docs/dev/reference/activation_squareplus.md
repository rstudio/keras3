# Squareplus activation function.

The Squareplus activation function is defined as:

`f(x) = (x + sqrt(x^2 + b)) / 2`

Where `b` is a smoothness parameter.

## Usage

``` r
activation_squareplus(x, b = 4L)
```

## Arguments

- x:

  Input tensor.

- b:

  Smoothness parameter. Defaults to 4.

## Value

A tensor, the result from applying the activation to the input tensor
`x`.

## Reference

- [Ramachandran et al., 2021](https://arxiv.org/abs/2112.11687)

## See also

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
[`activation_softmax()`](https://keras3.posit.co/dev/reference/activation_softmax.md)  
[`activation_softplus()`](https://keras3.posit.co/dev/reference/activation_softplus.md)  
[`activation_softsign()`](https://keras3.posit.co/dev/reference/activation_softsign.md)  
[`activation_sparse_plus()`](https://keras3.posit.co/dev/reference/activation_sparse_plus.md)  
[`activation_sparse_sigmoid()`](https://keras3.posit.co/dev/reference/activation_sparse_sigmoid.md)  
[`activation_sparsemax()`](https://keras3.posit.co/dev/reference/activation_sparsemax.md)  
[`activation_tanh()`](https://keras3.posit.co/dev/reference/activation_tanh.md)  
[`activation_tanh_shrink()`](https://keras3.posit.co/dev/reference/activation_tanh_shrink.md)  
[`activation_threshold()`](https://keras3.posit.co/dev/reference/activation_threshold.md)  

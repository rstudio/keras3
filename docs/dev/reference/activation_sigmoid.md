# Sigmoid activation function.

It is defined as: `sigmoid(x) = 1 / (1 + exp(-x))`.

For small values (\<-5), `sigmoid` returns a value close to zero, and
for large values (\>5) the result of the function gets close to 1.

Sigmoid is equivalent to a 2-element softmax, where the second element
is assumed to be zero. The sigmoid function always returns a value
between 0 and 1.

## Usage

``` r
activation_sigmoid(x)
```

## Arguments

- x:

  Input tensor.

## Value

A tensor, the result from applying the activation to the input tensor
`x`.

## See also

- <https://keras.io/api/layers/activations#sigmoid-function>

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

# Scaled Exponential Linear Unit (SELU).

The Scaled Exponential Linear Unit (SELU) activation function is defined
as:

- `scale * x` if `x > 0`

- `scale * alpha * (exp(x) - 1)` if `x < 0`

where `alpha` and `scale` are pre-defined constants
(`alpha = 1.67326324` and `scale = 1.05070098`).

Basically, the SELU activation function multiplies `scale` (\> 1) with
the output of the `activation_elu` function to ensure a slope larger
than one for positive inputs.

The values of `alpha` and `scale` are chosen so that the mean and
variance of the inputs are preserved between two consecutive layers as
long as the weights are initialized correctly (see
[`initializer_lecun_normal()`](https://keras3.posit.co/reference/initializer_lecun_normal.md))
and the number of input units is "large enough" (see reference paper for
more information).

## Usage

``` r
activation_selu(x)
```

## Arguments

- x:

  Input tensor.

## Value

A tensor, the result from applying the activation to the input tensor
`x`.

## Notes

- To be used together with
  [`initializer_lecun_normal()`](https://keras3.posit.co/reference/initializer_lecun_normal.md).

- To be used together with the dropout variant
  [`layer_alpha_dropout()`](https://keras3.posit.co/reference/layer_alpha_dropout.md)
  (legacy, depracated).

## Reference

- [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)

## See also

- <https://keras.io/api/layers/activations#selu-function>

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
[`activation_sigmoid()`](https://keras3.posit.co/reference/activation_sigmoid.md)  
[`activation_silu()`](https://keras3.posit.co/reference/activation_silu.md)  
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

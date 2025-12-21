# Applies the rectified linear unit activation function.

With default values, this returns the standard ReLU activation:
`max(x, 0)`, the element-wise maximum of 0 and the input tensor.

Modifying default parameters allows you to use non-zero thresholds,
change the max value of the activation, and to use a non-zero multiple
of the input for values below the threshold.

## Usage

``` r
activation_relu(x, negative_slope = 0, max_value = NULL, threshold = 0)
```

## Arguments

- x:

  Input tensor.

- negative_slope:

  A `numeric` that controls the slope for values lower than the
  threshold.

- max_value:

  A `numeric` that sets the saturation threshold (the largest value the
  function will return).

- threshold:

  A `numeric` giving the threshold value of the activation function
  below which values will be damped or set to zero.

## Value

A tensor with the same shape and dtype as input `x`.

## Examples

    x <- c(-10, -5, 0, 5, 10)
    activation_relu(x)

    ## tf.Tensor([ 0.  0.  0.  5. 10.], shape=(5), dtype=float32)

    activation_relu(x, negative_slope = 0.5)

    ## tf.Tensor([-5.  -2.5  0.   5.  10. ], shape=(5), dtype=float32)

    activation_relu(x, max_value = 5)

    ## tf.Tensor([0. 0. 0. 5. 5.], shape=(5), dtype=float32)

    activation_relu(x, threshold = 5)

    ## tf.Tensor([-0. -0.  0.  0. 10.], shape=(5), dtype=float32)

## See also

- <https://keras.io/api/layers/activations#relu-function>

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
[`activation_sparsemax()`](https://keras3.posit.co/reference/activation_sparsemax.md)  
[`activation_squareplus()`](https://keras3.posit.co/reference/activation_squareplus.md)  
[`activation_tanh()`](https://keras3.posit.co/reference/activation_tanh.md)  
[`activation_tanh_shrink()`](https://keras3.posit.co/reference/activation_tanh_shrink.md)  
[`activation_threshold()`](https://keras3.posit.co/reference/activation_threshold.md)  

Scaled Exponential Linear Unit (SELU).

@description
The Scaled Exponential Linear Unit (SELU) activation function is defined as:

- `scale * x` if `x > 0`
- `scale * alpha * (exp(x) - 1)` if `x < 0`

where `alpha` and `scale` are pre-defined constants
(`alpha=1.67326324` and `scale=1.05070098`).

Basically, the SELU activation function multiplies `scale` (> 1) with the
output of the `keras.activations.elu` function to ensure a slope larger
than one for positive inputs.

The values of `alpha` and `scale` are
chosen so that the mean and variance of the inputs are preserved
between two consecutive layers as long as the weights are initialized
correctly (see `keras.initializers.LecunNormal` initializer)
and the number of input units is "large enough"
(see reference paper for more information).

# Notes
- To be used together with the
    `keras.initializers.LecunNormal` initializer.
- To be used together with the dropout variant
    `keras.layers.AlphaDropout` (rather than regular dropout).

# Reference
- [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)

@param x
Input tensor.

@export
@family activations
@seealso
+ <https:/keras.io/api/layers/activations#selu-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/activations/selu>

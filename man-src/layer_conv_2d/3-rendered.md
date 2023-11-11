2D convolution layer.

@description
This layer creates a convolution kernel that is convolved with the layer
input over a single spatial (or temporal) dimension to produce a tensor of
outputs. If `use_bias` is TRUE, a bias vector is created and added to the
outputs. Finally, if `activation` is not `NULL`, it is applied to the
outputs as well.

# Input Shape
- If `data_format="channels_last"`:
    A 4D tensor with shape: `(batch_size, height, width, channels)`
- If `data_format="channels_first"`:
    A 4D tensor with shape: `(batch_size, channels, height, width)`

# Output Shape
- If `data_format="channels_last"`:
    A 4D tensor with shape: `(batch_size, new_height, new_width, filters)`
- If `data_format="channels_first"`:
    A 4D tensor with shape: `(batch_size, filters, new_height, new_width)`

# Raises
ValueError: when both `strides > 1` and `dilation_rate > 1`.

# Examples

```r
x <- random_uniform(c(4, 10, 10, 128))
y <- x |> layer_conv_2d(32, 3, activation='relu')
y$shape
```

```
## TensorShape([4, 8, 8, 32])
```

@returns
A 4D tensor representing `activation(conv2d(inputs, kernel) + bias)`.

@param filters int, the dimension of the output space (the number of filters
    in the convolution).
@param kernel_size int or list of 2 integer, specifying the size of the
    convolution window.
@param strides int or list of 2 integer, specifying the stride length
    of the convolution. `strides > 1` is incompatible with
    `dilation_rate > 1`.
@param padding string, either `"valid"` or `"same"` (case-insensitive).
    `"valid"` means no padding. `"same"` results in padding evenly to
    the left/right or up/down of the input such that output has the same
    height/width dimension as the input.
@param data_format string, either `"channels_last"` or `"channels_first"`.
    The ordering of the dimensions in the inputs. `"channels_last"`
    corresponds to inputs with shape
    `(batch_size, channels, height, width)`
    while `"channels_first"` corresponds to inputs with shape
    `(batch_size, channels, height, width)`. It defaults to the
    `image_data_format` value found in your Keras config file at
    `~/.keras/keras.json`. If you never set it, then it will be
    `"channels_last"`.
@param dilation_rate int or list of 2 integers, specifying the dilation
    rate to use for dilated convolution.
@param groups A positive int specifying the number of groups in which the
    input is split along the channel axis. Each group is convolved
    separately with `filters // groups` filters. The output is the
    concatenation of all the `groups` results along the channel axis.
    Input channels and `filters` must both be divisible by `groups`.
@param activation Activation function. If `NULL`, no activation is applied.
@param use_bias bool, if `TRUE`, bias will be added to the output.
@param kernel_initializer Initializer for the convolution kernel. If `NULL`,
    the default initializer (`"glorot_uniform"`) will be used.
@param bias_initializer Initializer for the bias vector. If `NULL`, the
    default initializer (`"zeros"`) will be used.
@param kernel_regularizer Optional regularizer for the convolution kernel.
@param bias_regularizer Optional regularizer for the bias vector.
@param activity_regularizer Optional regularizer function for the output.
@param kernel_constraint Optional projection function to be applied to the
    kernel after being updated by an `Optimizer` (e.g. used to implement
    norm constraints or value constraints for layer weights). The
    function must take as input the unprojected variable and must return
    the projected variable (which must have the same shape). Constraints
    are not safe to use when doing asynchronous distributed training.
@param bias_constraint Optional projection function to be applied to the
    bias after being updated by an `Optimizer`.
@param object Object to compose the layer with. A tensor, array, or sequential model.
@param ... Passed on to the Python callable

@export
@family convolutional layers
@seealso
+ <https:/keras.io/api/layers/convolution_layers/convolution2d#conv2d-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D>

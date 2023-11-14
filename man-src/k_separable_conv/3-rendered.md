General N-D separable convolution.

@description
This ops supports 1D and 2D separable convolution. `separable_conv` is
a depthwise conv followed by a pointwise conv.

@returns
    A tensor of rank N+2, the result of the depthwise conv operation.

@param inputs
Tensor of rank N+2. `inputs` has shape
`(batch_size,) + inputs_spatial_shape + (num_channels,)` if
`data_format="channels_last"`, or
`(batch_size, num_channels) + inputs_spatial_shape` if
`data_format="channels_first"`.

@param depthwise_kernel
Tensor of rank N+2. `depthwise_kernel` has shape
`[kernel_spatial_shape, num_input_channels, num_channels_multiplier],`
`num_input_channels` should match the number of channels in
`inputs`.

@param pointwise_kernel
Tensor of rank N+2. `pointwise_kernel` has shape
`(*ones_like(kernel_spatial_shape),
num_input_channels * num_channels_multiplier, num_output_channels)`.

@param strides
int or int tuple/list of `len(inputs_spatial_shape)`,
specifying the strides of the convolution along each spatial
dimension. If `strides` is int, then every spatial dimension shares
the same `strides`.

@param padding
string, either `"valid"` or `"same"`. `"valid"` means no
padding is applied, and `"same"` results in padding evenly to the
left/right or up/down of the input such that output has the
same height/width dimension as the input when `strides=1`.

@param data_format
A string, either `"channels_last"` or `"channels_first"`.
`data_format` determines the ordering of the dimensions in the
inputs. If `data_format="channels_last"`, `inputs` is of shape
`(batch_size, ..., channels)` while if
`data_format="channels_first"`, `inputs` is of shape
`(batch_size, channels, ...)`.

@param dilation_rate
int or int tuple/list of `len(inputs_spatial_shape)`,
specifying the dilation rate to use for dilated convolution. If
`dilation_rate` is int, then every spatial dimension shares
the same `dilation_rate`.

@export
@family nn ops
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/nn#separableconv-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/separable_conv>

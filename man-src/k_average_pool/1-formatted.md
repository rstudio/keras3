Average pooling operation.

# Returns
    A tensor of rank N+2, the result of the average pooling operation.

@param inputs Tensor of rank N+2. `inputs` has shape
    `(batch_size,) + inputs_spatial_shape + (num_channels,)` if
    `data_format="channels_last"`, or
    `(batch_size, num_channels) + inputs_spatial_shape` if
    `data_format="channels_first"`. Pooling happens over the spatial
    dimensions only.
@param pool_size int or tuple/list of integers of size
    `len(inputs_spatial_shape)`, specifying the size of the pooling
    window for each spatial dimension of the input tensor. If
    `pool_size` is int, then every spatial dimension shares the same
    `pool_size`.
@param strides int or tuple/list of integers of size
    `len(inputs_spatial_shape)`. The stride of the sliding window for
    each spatial dimension of the input tensor. If `strides` is int,
    then every spatial dimension shares the same `strides`.
@param padding string, either `"valid"` or `"same"`. `"valid"` means no
    padding is applied, and `"same"` results in padding evenly to the
    left/right or up/down of the input such that output has the
    same height/width dimension as the input when `strides=1`.
@param data_format A string, either `"channels_last"` or `"channels_first"`.
    `data_format` determines the ordering of the dimensions in the
    inputs. If `data_format="channels_last"`, `inputs` is of shape
    `(batch_size, ..., channels)` while if
    `data_format="channels_first"`, `inputs` is of shape
    `(batch_size, channels, ...)`.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/average_pool>

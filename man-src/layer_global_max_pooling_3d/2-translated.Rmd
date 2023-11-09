Global max pooling operation for 3D data.

@description

# Input Shape
- If `data_format='channels_last'`:
    5D tensor with shape:
    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
- If `data_format='channels_first'`:
    5D tensor with shape:
    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

# Output Shape
- If `keepdims=False`:
    2D tensor with shape `(batch_size, channels)`.
- If `keepdims=True`:
    - If `data_format="channels_last"`:
        5D tensor with shape `(batch_size, 1, 1, 1, channels)`
    - If `data_format="channels_first"`:
        5D tensor with shape `(batch_size, channels, 1, 1, 1)`

# Examples
```python
x = np.random.rand(2, 4, 5, 4, 3)
y = keras.layers.GlobalMaxPooling3D()(x)
y.shape
# (2, 3)
```

@param data_format string, either `"channels_last"` or `"channels_first"`.
    The ordering of the dimensions in the inputs. `"channels_last"`
    corresponds to inputs with shape
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    while `"channels_first"` corresponds to inputs with shape
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
    It defaults to the `image_data_format` value found in your Keras
    config file at `~/.keras/keras.json`. If you never set it, then it
    will be `"channels_last"`.
@param keepdims A boolean, whether to keep the temporal dimension or not.
    If `keepdims` is `False` (default), the rank of the tensor is
    reduced for spatial dimensions. If `keepdims` is `True`, the
    spatial dimension are retained with length 1.
    The behavior is the same as for `tf.reduce_mean` or `np.mean`.
@param object Object to compose the layer with. A tensor, array, or sequential model.
@param ... Passed on to the Python callable

@export
@family pooling layers
@seealso
+ <https:/keras.io/api/layers/pooling_layers/global_max_pooling3d#globalmaxpooling3d-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalMaxPooling3D>

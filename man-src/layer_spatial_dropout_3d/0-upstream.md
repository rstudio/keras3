keras.layers.SpatialDropout3D
__signature__
(
  rate,
  data_format=None,
  seed=None,
  name=None,
  dtype=None
)
__doc__
Spatial 3D version of Dropout.

This version performs the same function as Dropout, however, it drops
entire 3D feature maps instead of individual elements. If adjacent voxels
within feature maps are strongly correlated (as is normally the case in
early convolution layers) then regular dropout will not regularize the
activations and will otherwise just result in an effective learning rate
decrease. In this case, SpatialDropout3D will help promote independence
between feature maps and should be used instead.

Args:
    rate: Float between 0 and 1. Fraction of the input units to drop.
    data_format: `"channels_first"` or `"channels_last"`.
        In `"channels_first"` mode, the channels dimension (the depth)
        is at index 1, in `"channels_last"` mode is it at index 4.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be `"channels_last"`.

Call arguments:
    inputs: A 5D tensor.
    training: Python boolean indicating whether the layer
            should behave in training mode (applying dropout)
            or in inference mode (pass-through).

Input shape:
    5D tensor with shape: `(samples, channels, dim1, dim2, dim3)` if
        data_format='channels_first'
    or 5D tensor with shape: `(samples, dim1, dim2, dim3, channels)` if
        data_format='channels_last'.

Output shape: Same as input.

Reference:

- [Tompson et al., 2014](https://arxiv.org/abs/1411.4280)

A preprocessing layer which randomly flips images during training.

This layer will flip the images horizontally and or vertically based on the
`mode` attribute. During inference time, the output will be identical to
input. Call the layer with `training=True` to flip the input.
Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
of integer or floating point dtype.
By default, the layer will output floats.

**Note:** This layer is safe to use inside a `tf.data` pipeline
(independently of which backend you're using).

Input shape:
    3D (unbatched) or 4D (batched) tensor with shape:
    `(..., height, width, channels)`, in `"channels_last"` format.

Output shape:
    3D (unbatched) or 4D (batched) tensor with shape:
    `(..., height, width, channels)`, in `"channels_last"` format.

Args:
    mode: String indicating which flip mode to use. Can be `"horizontal"`,
        `"vertical"`, or `"horizontal_and_vertical"`. `"horizontal"` is a
        left-right flip and `"vertical"` is a top-bottom flip. Defaults to
        `"horizontal_and_vertical"`
    seed: Integer. Used to create a random seed.
    **kwargs: Base layer keyword arguments, such as
        `name` and `dtype`.

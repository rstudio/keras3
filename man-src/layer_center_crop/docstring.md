A preprocessing layer which crops images.

This layers crops the central portion of the images to a target size. If an
image is smaller than the target size, it will be resized and cropped
so as to return the largest possible window in the image that matches
the target aspect ratio.

Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`).

Input shape:
    3D (unbatched) or 4D (batched) tensor with shape:
    `(..., height, width, channels)`, in `"channels_last"` format,
    or `(..., channels, height, width)`, in `"channels_first"` format.

Output shape:
    3D (unbatched) or 4D (batched) tensor with shape:
    `(..., target_height, target_width, channels)`,
    or `(..., channels, target_height, target_width)`,
    in `"channels_first"` format.

If the input height/width is even and the target height/width is odd (or
inversely), the input image is left-padded by 1 pixel.

**Note:** This layer is safe to use inside a `tf.data` pipeline
(independently of which backend you're using).

Args:
    height: Integer, the height of the output shape.
    width: Integer, the width of the output shape.
    data_format: string, either `"channels_last"` or `"channels_first"`.
        The ordering of the dimensions in the inputs. `"channels_last"`
        corresponds to inputs with shape `(batch, height, width, channels)`
        while `"channels_first"` corresponds to inputs with shape
        `(batch, channels, height, width)`. It defaults to the
        `image_data_format` value found in your Keras config file at
        `~/.keras/keras.json`. If you never set it, then it will be
        `"channels_last"`.

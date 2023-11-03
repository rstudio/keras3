Flattens the input. Does not affect the batch size.

Note: If inputs are shaped `(batch,)` without a feature axis, then
flattening adds an extra channel dimension and output shape is `(batch, 1)`.

Args:
    data_format: A string, one of `"channels_last"` (default) or
        `"channels_first"`. The ordering of the dimensions in the inputs.
        `"channels_last"` corresponds to inputs with shape
        `(batch, ..., channels)` while `"channels_first"` corresponds to
        inputs with shape `(batch, channels, ...)`.
        When unspecified, uses `image_data_format` value found in your Keras
        config file at `~/.keras/keras.json` (if exists). Defaults to
        `"channels_last"`.

Example:

>>> x = keras.Input(shape=(10, 64))
>>> y = keras.layers.Flatten()(x)
>>> y.shape
(None, 640)

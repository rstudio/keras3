keras.config.set_image_data_format
__signature__
(data_format)
__doc__
Set the value of the image data format convention.

Args:
    data_format: string. `'channels_first'` or `'channels_last'`.

Examples:

>>> keras.config.image_data_format()
'channels_last'

>>> keras.config.set_image_data_format('channels_first')
>>> keras.config.image_data_format()
'channels_first'

>>> # Set it back to `'channels_last'`
>>> keras.config.set_image_data_format('channels_last')

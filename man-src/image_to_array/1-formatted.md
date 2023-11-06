Converts a PIL Image instance to a NumPy array.

@description

# Usage
```python
from PIL import Image
img_data = np.random.random(size=(100, 100, 3))
img = keras.utils.array_to_img(img_data)
array = keras.utils.image.img_to_array(img)
```

@returns
    A 3D NumPy array.

@param img Input PIL Image instance.
@param data_format Image data format, can be either `"channels_first"` or
    `"channels_last"`. Defaults to `None`, in which case the global
    setting `keras.backend.image_data_format()` is used (unless you
    changed it, it defaults to `"channels_last"`).
@param dtype Dtype to use. `None` means the global setting
    `keras.backend.floatx()` is used (unless you changed it, it
    defaults to `"float32"`).

@export
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/utils/img_to_array>

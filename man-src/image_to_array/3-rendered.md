Converts a PIL Image instance to a matrix.

@description

# Usage

```r
img_data <- random_uniform(c(100, 100, 3))
img <- image_from_array(img_data)
array <- image_to_array(img)
str(array)
```

```
##  num [1:100, 1:100, 1:3] 179 41 69 100 108 173 14 64 200 163 ...
```

@returns
    A 3D array.

@param img Input PIL Image instance.
@param data_format Image data format, can be either `"channels_first"` or
    `"channels_last"`. Defaults to `NULL`, in which case the global
    setting `config_image_data_format()` is used (unless you
    changed it, it defaults to `"channels_last"`).
@param dtype Dtype to use. `NULL` means the global setting
    `config_floatx()` is used (unless you changed it, it
    defaults to `"float32"`).

@export
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/utils/img_to_array>

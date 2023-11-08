Converts a PIL Image instance to a matrix.

@description

# Usage

```r
image_path <- get_file(origin = "https://www.r-project.org/logo/Rlogo.png")
(img <- image_load(image_path))
```

```
## <PIL.Image.Image image mode=RGB size=724x561>
```

```r
array <- image_to_array(img)
str(array)
```

```
##  num [1:561, 1:724, 1:3] 0 0 0 0 0 0 0 0 0 0 ...
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
@family utils
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/utils/img_to_array>

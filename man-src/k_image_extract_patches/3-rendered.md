Extracts patches from the image(s).

@description

# Examples

```r
image <- random_uniform(c(2, 20, 20, 3), dtype = "float32") # batch of 2 RGB images
patches <- k_image_extract_patches(image, c(5, 5))
patches$shape
```

```
## TensorShape([2, 4, 4, 75])
```

```r
# (2, 4, 4, 75)
image <- random_uniform(c(20, 20, 3), dtype = "float32") # 1 RGB image
patches <- k_image_extract_patches(image, c(3, 3), c(1, 1))
patches$shape
```

```
## TensorShape([18, 18, 27])
```

```r
# (18, 18, 27)
```

@returns
Extracted patches 3D (if not batched) or 4D (if batched)

@param image
Input image or batch of images. Must be 3D or 4D.

@param size
Patch size int or list (patch_height, patch_width)

@param strides
strides along height and width. If not specified, or
if `NULL`, it defaults to the same value as `size`.

@param dilation_rate
This is the input stride, specifying how far two
consecutive patch samples are in the input. For value other than 1,
strides must be 1. NOTE: `strides > 1` is not supported in
conjunction with `dilation_rate > 1`

@param padding
The type of padding algorithm to use: `"same"` or `"valid"`.

@param data_format
string, either `"channels_last"` or `"channels_first"`.
The ordering of the dimensions in the inputs. `"channels_last"`
corresponds to inputs with shape `(batch, height, width, channels)`
while `"channels_first"` corresponds to inputs with shape
`(batch, channels, height, weight)`. It defaults to the
`image_data_format` value found in your Keras config file at
`~/.keras/keras.json`. If you never set it, then it will be
`"channels_last"`.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/image#extractpatches-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/image/extract_patches>


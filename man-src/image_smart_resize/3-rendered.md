Resize images to a target size without aspect ratio distortion.

@description
Image datasets typically yield images that have each a different
size. However, these images need to be batched before they can be
processed by Keras layers. To be batched, images need to share the same
height and width.

You could simply do, in TF (or JAX equivalent):


```r
size <- c(200, 200)
ds <- ds$map(\(img) tf$image$resize(img, size))
```

However, if you do this, you distort the aspect ratio of your images, since
in general they do not all have the same aspect ratio as `size`. This is
fine in many cases, but not always (e.g. for image generation models
this can be a problem).

Note that passing the argument `preserve_aspect_ratio=TRUE` to `tf$image$resize()`
will preserve the aspect ratio, but at the cost of no longer respecting the
provided target size.

This calls for:


```r
size <- c(200, 200)
ds <- ds$map(\(img) image_smart_resize(img, size))
```

Your output images will actually be `(200, 200)`, and will not be distorted.
Instead, the parts of the image that do not fit within the target size
get cropped out.

The resizing process is:

1. Take the largest centered crop of the image that has the same aspect
ratio as the target size. For instance, if `size=c(200, 200)` and the input
image has size `(340, 500)`, we take a crop of `(340, 340)` centered along
the width.
2. Resize the cropped image to the target size. In the example above,
we resize the `(340, 340)` crop to `(200, 200)`.

@returns
Array with shape `(size[1], size[2], channels)`.
If the input image was an array, the output is an array,
and if it was a backend-native tensor,
the output is a backend-native tensor.

@param x Input image or batch of images (as a tensor or array).
    Must be in format `(height, width, channels)`
    or `(batch_size, height, width, channels)`.
@param size Tuple of `(height, width)` integer. Target size.
@param interpolation String, interpolation to use for resizing.
    Defaults to `'bilinear'`.
    Supports `bilinear`, `nearest`, `bicubic`,
    `lanczos3`, `lanczos5`.
@param data_format `"channels_last"` or `"channels_first"`.
@param backend_module Backend module to use (if different from the default
    backend).

@export
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/smart_resize>

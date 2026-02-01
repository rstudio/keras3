# Resize images to a target size without aspect ratio distortion.

Image datasets typically yield images that have each a different size.
However, these images need to be batched before they can be processed by
Keras layers. To be batched, images need to share the same height and
width.

You could simply do, in TF (or JAX equivalent):

    size <- c(200, 200)
    ds <- ds$map(\(img) tf$image$resize(img, size))

However, if you do this, you distort the aspect ratio of your images,
since in general they do not all have the same aspect ratio as `size`.
This is fine in many cases, but not always (e.g. for image generation
models this can be a problem).

Note that passing the argument `preserve_aspect_ratio = TRUE` to
`tf$image$resize()` will preserve the aspect ratio, but at the cost of
no longer respecting the provided target size.

This calls for:

    size <- c(200, 200)
    ds <- ds$map(\(img) image_smart_resize(img, size))

Your output images will actually be `(200, 200)`, and will not be
distorted. Instead, the parts of the image that do not fit within the
target size get cropped out.

The resizing process is:

1.  Take the largest centered crop of the image that has the same aspect
    ratio as the target size. For instance, if `size = c(200, 200)` and
    the input image has size `(340, 500)`, we take a crop of
    `(340, 340)` centered along the width.

2.  Resize the cropped image to the target size. In the example above,
    we resize the `(340, 340)` crop to `(200, 200)`.

## Usage

``` r
image_smart_resize(
  x,
  size,
  interpolation = "bilinear",
  data_format = "channels_last",
  backend_module = NULL
)
```

## Arguments

- x:

  Input image or batch of images (as a tensor or array). Must be in
  format `(height, width, channels)` or
  `(batch_size, height, width, channels)`.

- size:

  Tuple of `(height, width)` integer. Target size.

- interpolation:

  String, interpolation to use for resizing. Supports `"bilinear"`,
  `"nearest"`, `"bicubic"`, `"lanczos3"`, `"lanczos5"`. Defaults to
  `'bilinear'`.

- data_format:

  `"channels_last"` or `"channels_first"`.

- backend_module:

  Backend module to use (if different from the default backend).

## Value

Array with shape `(size[1], size[2], channels)`. If the input image was
an array, the output is an array, and if it was a backend-native tensor,
the output is a backend-native tensor.

## See also

Other image utils:  
[`image_array_save()`](https://keras3.posit.co/reference/image_array_save.md)  
[`image_from_array()`](https://keras3.posit.co/reference/image_from_array.md)  
[`image_load()`](https://keras3.posit.co/reference/image_load.md)  
[`image_to_array()`](https://keras3.posit.co/reference/image_to_array.md)  
[`op_image_affine_transform()`](https://keras3.posit.co/reference/op_image_affine_transform.md)  
[`op_image_crop()`](https://keras3.posit.co/reference/op_image_crop.md)  
[`op_image_extract_patches()`](https://keras3.posit.co/reference/op_image_extract_patches.md)  
[`op_image_gaussian_blur()`](https://keras3.posit.co/reference/op_image_gaussian_blur.md)  
[`op_image_hsv_to_rgb()`](https://keras3.posit.co/reference/op_image_hsv_to_rgb.md)  
[`op_image_map_coordinates()`](https://keras3.posit.co/reference/op_image_map_coordinates.md)  
[`op_image_pad()`](https://keras3.posit.co/reference/op_image_pad.md)  
[`op_image_perspective_transform()`](https://keras3.posit.co/reference/op_image_perspective_transform.md)  
[`op_image_resize()`](https://keras3.posit.co/reference/op_image_resize.md)  
[`op_image_rgb_to_grayscale()`](https://keras3.posit.co/reference/op_image_rgb_to_grayscale.md)  
[`op_image_rgb_to_hsv()`](https://keras3.posit.co/reference/op_image_rgb_to_hsv.md)  

Other utils:  
[`audio_dataset_from_directory()`](https://keras3.posit.co/reference/audio_dataset_from_directory.md)  
[`clear_session()`](https://keras3.posit.co/reference/clear_session.md)  
[`config_disable_interactive_logging()`](https://keras3.posit.co/reference/config_disable_interactive_logging.md)  
[`config_disable_traceback_filtering()`](https://keras3.posit.co/reference/config_disable_traceback_filtering.md)  
[`config_enable_interactive_logging()`](https://keras3.posit.co/reference/config_enable_interactive_logging.md)  
[`config_enable_traceback_filtering()`](https://keras3.posit.co/reference/config_enable_traceback_filtering.md)  
[`config_is_interactive_logging_enabled()`](https://keras3.posit.co/reference/config_is_interactive_logging_enabled.md)  
[`config_is_traceback_filtering_enabled()`](https://keras3.posit.co/reference/config_is_traceback_filtering_enabled.md)  
[`get_file()`](https://keras3.posit.co/reference/get_file.md)  
[`get_source_inputs()`](https://keras3.posit.co/reference/get_source_inputs.md)  
[`image_array_save()`](https://keras3.posit.co/reference/image_array_save.md)  
[`image_dataset_from_directory()`](https://keras3.posit.co/reference/image_dataset_from_directory.md)  
[`image_from_array()`](https://keras3.posit.co/reference/image_from_array.md)  
[`image_load()`](https://keras3.posit.co/reference/image_load.md)  
[`image_to_array()`](https://keras3.posit.co/reference/image_to_array.md)  
[`layer_feature_space()`](https://keras3.posit.co/reference/layer_feature_space.md)  
[`normalize()`](https://keras3.posit.co/reference/normalize.md)  
[`pad_sequences()`](https://keras3.posit.co/reference/pad_sequences.md)  
[`set_random_seed()`](https://keras3.posit.co/reference/set_random_seed.md)  
[`split_dataset()`](https://keras3.posit.co/reference/split_dataset.md)  
[`text_dataset_from_directory()`](https://keras3.posit.co/reference/text_dataset_from_directory.md)  
[`timeseries_dataset_from_array()`](https://keras3.posit.co/reference/timeseries_dataset_from_array.md)  
[`to_categorical()`](https://keras3.posit.co/reference/to_categorical.md)  
[`zip_lists()`](https://keras3.posit.co/reference/zip_lists.md)  

Other preprocessing:  
[`image_dataset_from_directory()`](https://keras3.posit.co/reference/image_dataset_from_directory.md)  
[`text_dataset_from_directory()`](https://keras3.posit.co/reference/text_dataset_from_directory.md)  
[`timeseries_dataset_from_array()`](https://keras3.posit.co/reference/timeseries_dataset_from_array.md)  

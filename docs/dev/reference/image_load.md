# Loads an image into PIL format.

Loads an image into PIL format.

## Usage

``` r
image_load(
  path,
  color_mode = "rgb",
  target_size = NULL,
  interpolation = "nearest",
  keep_aspect_ratio = FALSE
)
```

## Arguments

- path:

  Path to image file.

- color_mode:

  One of `"grayscale"`, `"rgb"`, `"rgba"`. Default: `"rgb"`. The desired
  image format.

- target_size:

  Either `NULL` (default to original size) or tuple of ints
  `(img_height, img_width)`.

- interpolation:

  Interpolation method used to resample the image if the target size is
  different from that of the loaded image. Supported methods are
  `"nearest"`, `"bilinear"`, and `"bicubic"`. If PIL version 1.1.3 or
  newer is installed, `"lanczos"` is also supported. If PIL version
  3.4.0 or newer is installed, `"box"` and `"hamming"` are also
  supported. By default, `"nearest"` is used.

- keep_aspect_ratio:

  Boolean, whether to resize images to a target size without aspect
  ratio distortion. The image is cropped in the center with target
  aspect ratio before resizing.

## Value

A PIL Image instance.

## Example

    image_path <- get_file(origin = "https://www.r-project.org/logo/Rlogo.png")
    (image <- image_load(image_path))

    ## <PIL.Image.Image image mode=RGB size=724x561 at 0x0>

    input_arr <- image_to_array(image)
    str(input_arr)

    ##  num [1:561, 1:724, 1:3] 0 0 0 0 0 0 0 0 0 0 ...

    input_arr %<>% array_reshape(dim = c(1, dim(input_arr))) # Convert single image to a batch.

    model |> predict(input_arr)

## See also

- <https://keras.io/api/data_loading/image#loadimg-function>

Other image utils:  
[`image_array_save()`](https://keras3.posit.co/dev/reference/image_array_save.md)  
[`image_from_array()`](https://keras3.posit.co/dev/reference/image_from_array.md)  
[`image_smart_resize()`](https://keras3.posit.co/dev/reference/image_smart_resize.md)  
[`image_to_array()`](https://keras3.posit.co/dev/reference/image_to_array.md)  
[`op_image_affine_transform()`](https://keras3.posit.co/dev/reference/op_image_affine_transform.md)  
[`op_image_crop()`](https://keras3.posit.co/dev/reference/op_image_crop.md)  
[`op_image_extract_patches()`](https://keras3.posit.co/dev/reference/op_image_extract_patches.md)  
[`op_image_gaussian_blur()`](https://keras3.posit.co/dev/reference/op_image_gaussian_blur.md)  
[`op_image_hsv_to_rgb()`](https://keras3.posit.co/dev/reference/op_image_hsv_to_rgb.md)  
[`op_image_map_coordinates()`](https://keras3.posit.co/dev/reference/op_image_map_coordinates.md)  
[`op_image_pad()`](https://keras3.posit.co/dev/reference/op_image_pad.md)  
[`op_image_perspective_transform()`](https://keras3.posit.co/dev/reference/op_image_perspective_transform.md)  
[`op_image_resize()`](https://keras3.posit.co/dev/reference/op_image_resize.md)  
[`op_image_rgb_to_grayscale()`](https://keras3.posit.co/dev/reference/op_image_rgb_to_grayscale.md)  
[`op_image_rgb_to_hsv()`](https://keras3.posit.co/dev/reference/op_image_rgb_to_hsv.md)  

Other utils:  
[`audio_dataset_from_directory()`](https://keras3.posit.co/dev/reference/audio_dataset_from_directory.md)  
[`clear_session()`](https://keras3.posit.co/dev/reference/clear_session.md)  
[`config_disable_interactive_logging()`](https://keras3.posit.co/dev/reference/config_disable_interactive_logging.md)  
[`config_disable_traceback_filtering()`](https://keras3.posit.co/dev/reference/config_disable_traceback_filtering.md)  
[`config_enable_interactive_logging()`](https://keras3.posit.co/dev/reference/config_enable_interactive_logging.md)  
[`config_enable_traceback_filtering()`](https://keras3.posit.co/dev/reference/config_enable_traceback_filtering.md)  
[`config_is_interactive_logging_enabled()`](https://keras3.posit.co/dev/reference/config_is_interactive_logging_enabled.md)  
[`config_is_traceback_filtering_enabled()`](https://keras3.posit.co/dev/reference/config_is_traceback_filtering_enabled.md)  
[`get_file()`](https://keras3.posit.co/dev/reference/get_file.md)  
[`get_source_inputs()`](https://keras3.posit.co/dev/reference/get_source_inputs.md)  
[`image_array_save()`](https://keras3.posit.co/dev/reference/image_array_save.md)  
[`image_dataset_from_directory()`](https://keras3.posit.co/dev/reference/image_dataset_from_directory.md)  
[`image_from_array()`](https://keras3.posit.co/dev/reference/image_from_array.md)  
[`image_smart_resize()`](https://keras3.posit.co/dev/reference/image_smart_resize.md)  
[`image_to_array()`](https://keras3.posit.co/dev/reference/image_to_array.md)  
[`layer_feature_space()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)  
[`normalize()`](https://keras3.posit.co/dev/reference/normalize.md)  
[`pad_sequences()`](https://keras3.posit.co/dev/reference/pad_sequences.md)  
[`set_random_seed()`](https://keras3.posit.co/dev/reference/set_random_seed.md)  
[`split_dataset()`](https://keras3.posit.co/dev/reference/split_dataset.md)  
[`text_dataset_from_directory()`](https://keras3.posit.co/dev/reference/text_dataset_from_directory.md)  
[`timeseries_dataset_from_array()`](https://keras3.posit.co/dev/reference/timeseries_dataset_from_array.md)  
[`to_categorical()`](https://keras3.posit.co/dev/reference/to_categorical.md)  
[`zip_lists()`](https://keras3.posit.co/dev/reference/zip_lists.md)  

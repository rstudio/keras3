# Converts a PIL Image instance to a matrix.

Converts a PIL Image instance to a matrix.

## Usage

``` r
image_to_array(img, data_format = NULL, dtype = NULL)
```

## Arguments

- img:

  Input PIL Image instance.

- data_format:

  Image data format, can be either `"channels_first"` or
  `"channels_last"`. Defaults to `NULL`, in which case the global
  setting
  [`config_image_data_format()`](https://keras3.posit.co/dev/reference/config_image_data_format.md)
  is used (unless you changed it, it defaults to `"channels_last"`).

- dtype:

  Dtype to use. `NULL` and `"double"` return an R double array.
  `"integer"` returns an R integer array. All other values (e.g.,
  `"float32"`) return a NumPy array.

## Value

A 3D array.

## Example

    image_path <- get_file(origin = "https://www.r-project.org/logo/Rlogo.png")
    (img <- image_load(image_path))

    ## <PIL.Image.Image image mode=RGB size=724x561 at 0x0>

    array <- image_to_array(img)
    str(array)

    ##  num [1:561, 1:724, 1:3] 0 0 0 0 0 0 0 0 0 0 ...

## See also

- <https://keras.io/api/data_loading/image#imgtoarray-function>

Other image utils:  
[`image_array_save()`](https://keras3.posit.co/dev/reference/image_array_save.md)  
[`image_from_array()`](https://keras3.posit.co/dev/reference/image_from_array.md)  
[`image_load()`](https://keras3.posit.co/dev/reference/image_load.md)  
[`image_smart_resize()`](https://keras3.posit.co/dev/reference/image_smart_resize.md)  
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
[`image_load()`](https://keras3.posit.co/dev/reference/image_load.md)  
[`image_smart_resize()`](https://keras3.posit.co/dev/reference/image_smart_resize.md)  
[`layer_feature_space()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)  
[`normalize()`](https://keras3.posit.co/dev/reference/normalize.md)  
[`pad_sequences()`](https://keras3.posit.co/dev/reference/pad_sequences.md)  
[`set_random_seed()`](https://keras3.posit.co/dev/reference/set_random_seed.md)  
[`split_dataset()`](https://keras3.posit.co/dev/reference/split_dataset.md)  
[`text_dataset_from_directory()`](https://keras3.posit.co/dev/reference/text_dataset_from_directory.md)  
[`timeseries_dataset_from_array()`](https://keras3.posit.co/dev/reference/timeseries_dataset_from_array.md)  
[`to_categorical()`](https://keras3.posit.co/dev/reference/to_categorical.md)  
[`zip_lists()`](https://keras3.posit.co/dev/reference/zip_lists.md)  

# Normalizes an array.

If the input is an R array, an R array will be returned. If it's a
backend tensor, a backend tensor will be returned.

## Usage

``` r
normalize(x, axis = -1L, order = 2L)
```

## Arguments

- x:

  Array to normalize.

- axis:

  axis along which to normalize.

- order:

  Normalization order (e.g. `order=2` for L2 norm).

## Value

A normalized copy of the array.

## See also

- <https://keras.io/api/utils/python_utils#normalize-function>

Other numerical utils:  
[`to_categorical()`](https://keras3.posit.co/reference/to_categorical.md)  

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
[`image_smart_resize()`](https://keras3.posit.co/reference/image_smart_resize.md)  
[`image_to_array()`](https://keras3.posit.co/reference/image_to_array.md)  
[`layer_feature_space()`](https://keras3.posit.co/reference/layer_feature_space.md)  
[`pad_sequences()`](https://keras3.posit.co/reference/pad_sequences.md)  
[`set_random_seed()`](https://keras3.posit.co/reference/set_random_seed.md)  
[`split_dataset()`](https://keras3.posit.co/reference/split_dataset.md)  
[`text_dataset_from_directory()`](https://keras3.posit.co/reference/text_dataset_from_directory.md)  
[`timeseries_dataset_from_array()`](https://keras3.posit.co/reference/timeseries_dataset_from_array.md)  
[`to_categorical()`](https://keras3.posit.co/reference/to_categorical.md)  
[`zip_lists()`](https://keras3.posit.co/reference/zip_lists.md)  

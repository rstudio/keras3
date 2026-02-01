# Packs user-provided data into a list.

This is a convenience utility for packing data into the list formats
that [`fit()`](https://generics.r-lib.org/reference/fit.html) uses.

## Usage

``` r
pack_x_y_sample_weight(x, y = NULL, sample_weight = NULL)
```

## Arguments

- x:

  Features to pass to `Model`.

- y:

  Ground-truth targets to pass to `Model`.

- sample_weight:

  Sample weight for each element.

## Value

List in the format used in
[`fit()`](https://generics.r-lib.org/reference/fit.html).

## Example

    x <- op_ones(c(10, 1))
    data <- pack_x_y_sample_weight(x)


    y <- op_ones(c(10, 1))
    data <- pack_x_y_sample_weight(x, y)

## See also

Other data utils:  
[`unpack_x_y_sample_weight`](https://keras3.posit.co/reference/unpack_x_y_sample_weight.md)`()`  
[`zip_lists`](https://keras3.posit.co/reference/zip_lists.md)`()`  

Other utils:  
[`audio_dataset_from_directory`](https://keras3.posit.co/reference/audio_dataset_from_directory.md)`()`  
[`clear_session`](https://keras3.posit.co/reference/clear_session.md)`()`  
[`config_disable_interactive_logging`](https://keras3.posit.co/reference/config_disable_interactive_logging.md)`()`  
[`config_disable_traceback_filtering`](https://keras3.posit.co/reference/config_disable_traceback_filtering.md)`()`  
[`config_enable_interactive_logging`](https://keras3.posit.co/reference/config_enable_interactive_logging.md)`()`  
[`config_enable_traceback_filtering`](https://keras3.posit.co/reference/config_enable_traceback_filtering.md)`()`  
[`config_is_interactive_logging_enabled`](https://keras3.posit.co/reference/config_is_interactive_logging_enabled.md)`()`  
[`config_is_traceback_filtering_enabled`](https://keras3.posit.co/reference/config_is_traceback_filtering_enabled.md)`()`  
[`get_file`](https://keras3.posit.co/reference/get_file.md)`()`  
[`get_source_inputs`](https://keras3.posit.co/reference/get_source_inputs.md)`()`  
[`image_array_save`](https://keras3.posit.co/reference/image_array_save.md)`()`  
[`image_dataset_from_directory`](https://keras3.posit.co/reference/image_dataset_from_directory.md)`()`  
[`image_from_array`](https://keras3.posit.co/reference/image_from_array.md)`()`  
[`image_load`](https://keras3.posit.co/reference/image_load.md)`()`  
[`image_smart_resize`](https://keras3.posit.co/reference/image_smart_resize.md)`()`  
[`image_to_array`](https://keras3.posit.co/reference/image_to_array.md)`()`  
[`layer_feature_space`](https://keras3.posit.co/reference/layer_feature_space.md)`()`  
[`normalize`](https://keras3.posit.co/reference/normalize.md)`()`  
[`pad_sequences`](https://keras3.posit.co/reference/pad_sequences.md)`()`  
[`set_random_seed`](https://keras3.posit.co/reference/set_random_seed.md)`()`  
[`split_dataset`](https://keras3.posit.co/reference/split_dataset.md)`()`  
[`text_dataset_from_directory`](https://keras3.posit.co/reference/text_dataset_from_directory.md)`()`  
[`timeseries_dataset_from_array`](https://keras3.posit.co/reference/timeseries_dataset_from_array.md)`()`  
[`to_categorical`](https://keras3.posit.co/reference/to_categorical.md)`()`  
[`unpack_x_y_sample_weight`](https://keras3.posit.co/reference/unpack_x_y_sample_weight.md)`()`  
[`zip_lists`](https://keras3.posit.co/reference/zip_lists.md)`()`  

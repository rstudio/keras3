# Returns the list of input tensors necessary to compute `tensor`.

Output will always be a list of tensors (potentially with 1 element).

## Usage

``` r
get_source_inputs(tensor)
```

## Arguments

- tensor:

  The tensor to start from.

## Value

List of input tensors.

## Example

    input <- keras_input(c(3))
    output <- input |> layer_dense(4) |> op_multiply(5)
    reticulate::py_id(get_source_inputs(output)[[1]]) ==
    reticulate::py_id(input)

    ## [1] TRUE

## See also

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
[`image_array_save()`](https://keras3.posit.co/dev/reference/image_array_save.md)  
[`image_dataset_from_directory()`](https://keras3.posit.co/dev/reference/image_dataset_from_directory.md)  
[`image_from_array()`](https://keras3.posit.co/dev/reference/image_from_array.md)  
[`image_load()`](https://keras3.posit.co/dev/reference/image_load.md)  
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

# Splits a dataset into a left half and a right half (e.g. train / test).

Splits a dataset into a left half and a right half (e.g. train / test).

## Usage

``` r
split_dataset(
  dataset,
  left_size = NULL,
  right_size = NULL,
  shuffle = FALSE,
  seed = NULL
)
```

## Arguments

- dataset:

  A `tf$data$Dataset`, a `torch$utils$data$Dataset` object, or a list of
  arrays with the same length.

- left_size:

  If float (in the range `[0, 1]`), it signifies the fraction of the
  data to pack in the left dataset. If integer, it signifies the number
  of samples to pack in the left dataset. If `NULL`, defaults to the
  complement to `right_size`. Defaults to `NULL`.

- right_size:

  If float (in the range `[0, 1]`), it signifies the fraction of the
  data to pack in the right dataset. If integer, it signifies the number
  of samples to pack in the right dataset. If `NULL`, defaults to the
  complement to `left_size`. Defaults to `NULL`.

- shuffle:

  Boolean, whether to shuffle the data before splitting it.

- seed:

  A random seed for shuffling.

## Value

A list of two `tf$data$Dataset` objects: the left and right splits.

## Examples

    data <- random_uniform(c(1000, 4))
    c(left_ds, right_ds) %<-% split_dataset(list(data$numpy()), left_size = 0.8)
    left_ds$cardinality()

    ## tf.Tensor(800, shape=(), dtype=int64)

    right_ds$cardinality()

    ## tf.Tensor(200, shape=(), dtype=int64)

## See also

- <https://keras.io/api/utils/python_utils#splitdataset-function>

Other dataset utils:  
[`audio_dataset_from_directory()`](https://keras3.posit.co/reference/audio_dataset_from_directory.md)  
[`image_dataset_from_directory()`](https://keras3.posit.co/reference/image_dataset_from_directory.md)  
[`text_dataset_from_directory()`](https://keras3.posit.co/reference/text_dataset_from_directory.md)  
[`timeseries_dataset_from_array()`](https://keras3.posit.co/reference/timeseries_dataset_from_array.md)  

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
[`normalize()`](https://keras3.posit.co/reference/normalize.md)  
[`pad_sequences()`](https://keras3.posit.co/reference/pad_sequences.md)  
[`set_random_seed()`](https://keras3.posit.co/reference/set_random_seed.md)  
[`text_dataset_from_directory()`](https://keras3.posit.co/reference/text_dataset_from_directory.md)  
[`timeseries_dataset_from_array()`](https://keras3.posit.co/reference/timeseries_dataset_from_array.md)  
[`to_categorical()`](https://keras3.posit.co/reference/to_categorical.md)  
[`zip_lists()`](https://keras3.posit.co/reference/zip_lists.md)  

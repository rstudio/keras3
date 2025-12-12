# Pads sequences to the same length.

This function transforms a list (of length `num_samples`) of sequences
(lists of integers) into a 2D NumPy array of shape
`(num_samples, num_timesteps)`. `num_timesteps` is either the `maxlen`
argument if provided, or the length of the longest sequence in the list.

Sequences that are shorter than `num_timesteps` are padded with `value`
until they are `num_timesteps` long.

Sequences longer than `num_timesteps` are truncated so that they fit the
desired length.

The position where padding or truncation happens is determined by the
arguments `padding` and `truncating`, respectively. Pre-padding or
removing values from the beginning of the sequence is the default.

    sequence <- list(c(1), c(2, 3), c(4, 5, 6))
    pad_sequences(sequence)

    ##      [,1] [,2] [,3]
    ## [1,]    0    0    1
    ## [2,]    0    2    3
    ## [3,]    4    5    6

    pad_sequences(sequence, value=-1)

    ##      [,1] [,2] [,3]
    ## [1,]   -1   -1    1
    ## [2,]   -1    2    3
    ## [3,]    4    5    6

    pad_sequences(sequence, padding='post')

    ##      [,1] [,2] [,3]
    ## [1,]    1    0    0
    ## [2,]    2    3    0
    ## [3,]    4    5    6

    pad_sequences(sequence, maxlen=2)

    ##      [,1] [,2]
    ## [1,]    0    1
    ## [2,]    2    3
    ## [3,]    5    6

## Usage

``` r
pad_sequences(
  sequences,
  maxlen = NULL,
  dtype = "int32",
  padding = "pre",
  truncating = "pre",
  value = 0
)
```

## Arguments

- sequences:

  List of sequences (each sequence is a list of integers).

- maxlen:

  Optional Int, maximum length of all sequences. If not provided,
  sequences will be padded to the length of the longest individual
  sequence.

- dtype:

  (Optional, defaults to `"int32"`). Type of the output sequences. To
  pad sequences with variable length strings, you can use `object`.

- padding:

  String, "pre" or "post" (optional, defaults to `"pre"`): pad either
  before or after each sequence.

- truncating:

  String, "pre" or "post" (optional, defaults to `"pre"`): remove values
  from sequences larger than `maxlen`, either at the beginning or at the
  end of the sequences.

- value:

  Float or String, padding value. (Optional, defaults to `0`)

## Value

Array with shape `(len(sequences), maxlen)`

## See also

- <https://keras.io/api/data_loading/timeseries#padsequences-function>

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
[`image_to_array()`](https://keras3.posit.co/dev/reference/image_to_array.md)  
[`layer_feature_space()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)  
[`normalize()`](https://keras3.posit.co/dev/reference/normalize.md)  
[`set_random_seed()`](https://keras3.posit.co/dev/reference/set_random_seed.md)  
[`split_dataset()`](https://keras3.posit.co/dev/reference/split_dataset.md)  
[`text_dataset_from_directory()`](https://keras3.posit.co/dev/reference/text_dataset_from_directory.md)  
[`timeseries_dataset_from_array()`](https://keras3.posit.co/dev/reference/timeseries_dataset_from_array.md)  
[`to_categorical()`](https://keras3.posit.co/dev/reference/to_categorical.md)  
[`zip_lists()`](https://keras3.posit.co/dev/reference/zip_lists.md)  

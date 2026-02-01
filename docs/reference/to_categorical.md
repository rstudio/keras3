# Converts a class vector (integers) to binary class matrix.

E.g. for use with
[`loss_categorical_crossentropy()`](https://keras3.posit.co/reference/loss_categorical_crossentropy.md).

## Usage

``` r
to_categorical(x, num_classes = NULL)
```

## Arguments

- x:

  Array-like with class values to be converted into a matrix (integers
  from 0 to `num_classes - 1`). R factors are coerced to integer and
  offset to be 0-based, i.e., `as.integer(x) - 1L`.

- num_classes:

  Total number of classes. If `NULL`, this would be inferred as
  `max(x) + 1`. Defaults to `NULL`.

## Value

A binary matrix representation of the input as an R array. The class
axis is placed last.

## Examples

    a <- to_categorical(c(0, 1, 2, 3), num_classes=4)
    print(a)

    ##      [,1] [,2] [,3] [,4]
    ## [1,]    1    0    0    0
    ## [2,]    0    1    0    0
    ## [3,]    0    0    1    0
    ## [4,]    0    0    0    1

    b <- array(c(.9, .04, .03, .03,
                  .3, .45, .15, .13,
                  .04, .01, .94, .05,
                  .12, .21, .5, .17),
                  dim = c(4, 4))
    loss <- op_categorical_crossentropy(a, b)
    loss

    ## tf.Tensor([0.41284522 0.45601739 0.54430155 0.80437282], shape=(4), dtype=float64)

    loss <- op_categorical_crossentropy(a, a)
    loss

    ## tf.Tensor([1.00000005e-07 1.00000005e-07 1.00000005e-07 1.00000005e-07], shape=(4), dtype=float64)

## See also

- [`op_one_hot()`](https://keras3.posit.co/reference/op_one_hot.md),
  which does the same operation as `to_categorical()`, but operating on
  tensors.

- [`loss_sparse_categorical_crossentropy()`](https://keras3.posit.co/reference/loss_sparse_categorical_crossentropy.md),
  which can accept labels (`y_true`) as an integer vector, instead of as
  a dense one-hot matrix.

- <https://keras.io/api/utils/python_utils#tocategorical-function>

Other numerical utils:  
[`normalize()`](https://keras3.posit.co/reference/normalize.md)  

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
[`split_dataset()`](https://keras3.posit.co/reference/split_dataset.md)  
[`text_dataset_from_directory()`](https://keras3.posit.co/reference/text_dataset_from_directory.md)  
[`timeseries_dataset_from_array()`](https://keras3.posit.co/reference/timeseries_dataset_from_array.md)  
[`zip_lists()`](https://keras3.posit.co/reference/zip_lists.md)  

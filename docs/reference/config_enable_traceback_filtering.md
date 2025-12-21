# Turn on traceback filtering.

Raw Keras tracebacks (also known as stack traces) involve many internal
frames, which can be challenging to read through, while not being
actionable for end users. By default, Keras filters internal frames in
most exceptions that it raises, to keep traceback short, readable, and
focused on what's actionable for you (your own code).

See also
[`config_disable_traceback_filtering()`](https://keras3.posit.co/reference/config_disable_traceback_filtering.md)
and
[`config_is_traceback_filtering_enabled()`](https://keras3.posit.co/reference/config_is_traceback_filtering_enabled.md).

If you have previously disabled traceback filtering via
[`config_disable_traceback_filtering()`](https://keras3.posit.co/reference/config_disable_traceback_filtering.md),
you can re-enable it via `config_enable_traceback_filtering()`.

## Usage

``` r
config_enable_traceback_filtering()
```

## Value

No return value, called for side effects.

## See also

Other traceback utils:  
[`config_disable_traceback_filtering()`](https://keras3.posit.co/reference/config_disable_traceback_filtering.md)  
[`config_is_traceback_filtering_enabled()`](https://keras3.posit.co/reference/config_is_traceback_filtering_enabled.md)  

Other utils:  
[`audio_dataset_from_directory()`](https://keras3.posit.co/reference/audio_dataset_from_directory.md)  
[`clear_session()`](https://keras3.posit.co/reference/clear_session.md)  
[`config_disable_interactive_logging()`](https://keras3.posit.co/reference/config_disable_interactive_logging.md)  
[`config_disable_traceback_filtering()`](https://keras3.posit.co/reference/config_disable_traceback_filtering.md)  
[`config_enable_interactive_logging()`](https://keras3.posit.co/reference/config_enable_interactive_logging.md)  
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
[`to_categorical()`](https://keras3.posit.co/reference/to_categorical.md)  
[`zip_lists()`](https://keras3.posit.co/reference/zip_lists.md)  

Other config:  
[`config_backend()`](https://keras3.posit.co/reference/config_backend.md)  
[`config_disable_flash_attention()`](https://keras3.posit.co/reference/config_disable_flash_attention.md)  
[`config_disable_interactive_logging()`](https://keras3.posit.co/reference/config_disable_interactive_logging.md)  
[`config_disable_traceback_filtering()`](https://keras3.posit.co/reference/config_disable_traceback_filtering.md)  
[`config_dtype_policy()`](https://keras3.posit.co/reference/config_dtype_policy.md)  
[`config_enable_flash_attention()`](https://keras3.posit.co/reference/config_enable_flash_attention.md)  
[`config_enable_interactive_logging()`](https://keras3.posit.co/reference/config_enable_interactive_logging.md)  
[`config_enable_unsafe_deserialization()`](https://keras3.posit.co/reference/config_enable_unsafe_deserialization.md)  
[`config_epsilon()`](https://keras3.posit.co/reference/config_epsilon.md)  
[`config_floatx()`](https://keras3.posit.co/reference/config_floatx.md)  
[`config_image_data_format()`](https://keras3.posit.co/reference/config_image_data_format.md)  
[`config_is_interactive_logging_enabled()`](https://keras3.posit.co/reference/config_is_interactive_logging_enabled.md)  
[`config_is_nnx_enabled()`](https://keras3.posit.co/reference/config_is_nnx_enabled.md)  
[`config_is_traceback_filtering_enabled()`](https://keras3.posit.co/reference/config_is_traceback_filtering_enabled.md)  
[`config_max_epochs()`](https://keras3.posit.co/reference/config_max_epochs.md)  
[`config_set_backend()`](https://keras3.posit.co/reference/config_set_backend.md)  
[`config_set_dtype_policy()`](https://keras3.posit.co/reference/config_set_dtype_policy.md)  
[`config_set_epsilon()`](https://keras3.posit.co/reference/config_set_epsilon.md)  
[`config_set_floatx()`](https://keras3.posit.co/reference/config_set_floatx.md)  
[`config_set_image_data_format()`](https://keras3.posit.co/reference/config_set_image_data_format.md)  

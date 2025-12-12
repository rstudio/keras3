# Set the value of the image data format convention.

Set the value of the image data format convention.

## Usage

``` r
config_set_image_data_format(data_format)
```

## Arguments

- data_format:

  string. `'channels_first'` or `'channels_last'`.

## Value

No return value, called for side effects.

## Examples

    config_image_data_format()

    ## [1] "channels_last"

    # 'channels_last'

    keras3::config_set_image_data_format('channels_first')
    config_image_data_format()

    ## [1] "channels_first"

    # Set it back to `'channels_last'`
    keras3::config_set_image_data_format('channels_last')

## See also

- <https://keras.io/api/utils/config_utils#setimagedataformat-function>

Other config backend:  
[`config_backend()`](https://keras3.posit.co/dev/reference/config_backend.md)  
[`config_epsilon()`](https://keras3.posit.co/dev/reference/config_epsilon.md)  
[`config_floatx()`](https://keras3.posit.co/dev/reference/config_floatx.md)  
[`config_image_data_format()`](https://keras3.posit.co/dev/reference/config_image_data_format.md)  
[`config_set_epsilon()`](https://keras3.posit.co/dev/reference/config_set_epsilon.md)  
[`config_set_floatx()`](https://keras3.posit.co/dev/reference/config_set_floatx.md)  

Other backend:  
[`clear_session()`](https://keras3.posit.co/dev/reference/clear_session.md)  
[`config_backend()`](https://keras3.posit.co/dev/reference/config_backend.md)  
[`config_epsilon()`](https://keras3.posit.co/dev/reference/config_epsilon.md)  
[`config_floatx()`](https://keras3.posit.co/dev/reference/config_floatx.md)  
[`config_image_data_format()`](https://keras3.posit.co/dev/reference/config_image_data_format.md)  
[`config_set_epsilon()`](https://keras3.posit.co/dev/reference/config_set_epsilon.md)  
[`config_set_floatx()`](https://keras3.posit.co/dev/reference/config_set_floatx.md)  

Other config:  
[`config_backend()`](https://keras3.posit.co/dev/reference/config_backend.md)  
[`config_disable_flash_attention()`](https://keras3.posit.co/dev/reference/config_disable_flash_attention.md)  
[`config_disable_interactive_logging()`](https://keras3.posit.co/dev/reference/config_disable_interactive_logging.md)  
[`config_disable_traceback_filtering()`](https://keras3.posit.co/dev/reference/config_disable_traceback_filtering.md)  
[`config_dtype_policy()`](https://keras3.posit.co/dev/reference/config_dtype_policy.md)  
[`config_enable_flash_attention()`](https://keras3.posit.co/dev/reference/config_enable_flash_attention.md)  
[`config_enable_interactive_logging()`](https://keras3.posit.co/dev/reference/config_enable_interactive_logging.md)  
[`config_enable_traceback_filtering()`](https://keras3.posit.co/dev/reference/config_enable_traceback_filtering.md)  
[`config_enable_unsafe_deserialization()`](https://keras3.posit.co/dev/reference/config_enable_unsafe_deserialization.md)  
[`config_epsilon()`](https://keras3.posit.co/dev/reference/config_epsilon.md)  
[`config_floatx()`](https://keras3.posit.co/dev/reference/config_floatx.md)  
[`config_image_data_format()`](https://keras3.posit.co/dev/reference/config_image_data_format.md)  
[`config_is_interactive_logging_enabled()`](https://keras3.posit.co/dev/reference/config_is_interactive_logging_enabled.md)  
[`config_is_nnx_enabled()`](https://keras3.posit.co/dev/reference/config_is_nnx_enabled.md)  
[`config_is_traceback_filtering_enabled()`](https://keras3.posit.co/dev/reference/config_is_traceback_filtering_enabled.md)  
[`config_max_epochs()`](https://keras3.posit.co/dev/reference/config_max_epochs.md)  
[`config_set_backend()`](https://keras3.posit.co/dev/reference/config_set_backend.md)  
[`config_set_dtype_policy()`](https://keras3.posit.co/dev/reference/config_set_dtype_policy.md)  
[`config_set_epsilon()`](https://keras3.posit.co/dev/reference/config_set_epsilon.md)  
[`config_set_floatx()`](https://keras3.posit.co/dev/reference/config_set_floatx.md)  

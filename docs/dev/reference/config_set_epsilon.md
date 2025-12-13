# Set the value of the fuzz factor used in numeric expressions.

Set the value of the fuzz factor used in numeric expressions.

## Usage

``` r
config_set_epsilon(value)
```

## Arguments

- value:

  float. New value of epsilon.

## Value

No return value, called for side effects.

## Examples

    config_epsilon()

    ## [1] 1e-07

    config_set_epsilon(1e-5)
    config_epsilon()

    ## [1] 1e-05

    # Set it back to the default value.
    config_set_epsilon(1e-7)

## See also

- <https://keras.io/api/utils/config_utils#setepsilon-function>

Other config backend:  
[`config_backend()`](https://keras3.posit.co/dev/reference/config_backend.md)  
[`config_epsilon()`](https://keras3.posit.co/dev/reference/config_epsilon.md)  
[`config_floatx()`](https://keras3.posit.co/dev/reference/config_floatx.md)  
[`config_image_data_format()`](https://keras3.posit.co/dev/reference/config_image_data_format.md)  
[`config_set_floatx()`](https://keras3.posit.co/dev/reference/config_set_floatx.md)  
[`config_set_image_data_format()`](https://keras3.posit.co/dev/reference/config_set_image_data_format.md)  

Other backend:  
[`clear_session()`](https://keras3.posit.co/dev/reference/clear_session.md)  
[`config_backend()`](https://keras3.posit.co/dev/reference/config_backend.md)  
[`config_epsilon()`](https://keras3.posit.co/dev/reference/config_epsilon.md)  
[`config_floatx()`](https://keras3.posit.co/dev/reference/config_floatx.md)  
[`config_image_data_format()`](https://keras3.posit.co/dev/reference/config_image_data_format.md)  
[`config_set_floatx()`](https://keras3.posit.co/dev/reference/config_set_floatx.md)  
[`config_set_image_data_format()`](https://keras3.posit.co/dev/reference/config_set_image_data_format.md)  

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
[`config_set_floatx()`](https://keras3.posit.co/dev/reference/config_set_floatx.md)  
[`config_set_image_data_format()`](https://keras3.posit.co/dev/reference/config_set_image_data_format.md)  

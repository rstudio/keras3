# Set the default float dtype.

Set the default float dtype.

## Usage

``` r
config_set_floatx(value)
```

## Arguments

- value:

  String; `'bfloat16'`, `'float16'`, `'float32'`, or `'float64'`.

## Value

No return value, called for side effects.

## Note

It is not recommended to set this to `"float16"` for training, as this
will likely cause numeric stability issues. Instead, mixed precision,
which leverages a mix of `float16` and `float32`. It can be configured
by calling
`keras3::keras$mixed_precision$set_dtype_policy('mixed_float16')`.

## Examples

    config_floatx()

    ## [1] "float32"

    config_set_floatx('float64')
    config_floatx()

    ## [1] "float64"

    # Set it back to float32
    config_set_floatx('float32')

## Raises

ValueError: In case of invalid value.

## See also

- <https://keras.io/api/utils/config_utils#setfloatx-function>

Other config backend:  
[`config_backend()`](https://keras3.posit.co/reference/config_backend.md)  
[`config_epsilon()`](https://keras3.posit.co/reference/config_epsilon.md)  
[`config_floatx()`](https://keras3.posit.co/reference/config_floatx.md)  
[`config_image_data_format()`](https://keras3.posit.co/reference/config_image_data_format.md)  
[`config_set_epsilon()`](https://keras3.posit.co/reference/config_set_epsilon.md)  
[`config_set_image_data_format()`](https://keras3.posit.co/reference/config_set_image_data_format.md)  

Other backend:  
[`clear_session()`](https://keras3.posit.co/reference/clear_session.md)  
[`config_backend()`](https://keras3.posit.co/reference/config_backend.md)  
[`config_epsilon()`](https://keras3.posit.co/reference/config_epsilon.md)  
[`config_floatx()`](https://keras3.posit.co/reference/config_floatx.md)  
[`config_image_data_format()`](https://keras3.posit.co/reference/config_image_data_format.md)  
[`config_set_epsilon()`](https://keras3.posit.co/reference/config_set_epsilon.md)  
[`config_set_image_data_format()`](https://keras3.posit.co/reference/config_set_image_data_format.md)  

Other config:  
[`config_backend()`](https://keras3.posit.co/reference/config_backend.md)  
[`config_disable_flash_attention()`](https://keras3.posit.co/reference/config_disable_flash_attention.md)  
[`config_disable_interactive_logging()`](https://keras3.posit.co/reference/config_disable_interactive_logging.md)  
[`config_disable_traceback_filtering()`](https://keras3.posit.co/reference/config_disable_traceback_filtering.md)  
[`config_dtype_policy()`](https://keras3.posit.co/reference/config_dtype_policy.md)  
[`config_enable_flash_attention()`](https://keras3.posit.co/reference/config_enable_flash_attention.md)  
[`config_enable_interactive_logging()`](https://keras3.posit.co/reference/config_enable_interactive_logging.md)  
[`config_enable_traceback_filtering()`](https://keras3.posit.co/reference/config_enable_traceback_filtering.md)  
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
[`config_set_image_data_format()`](https://keras3.posit.co/reference/config_set_image_data_format.md)  

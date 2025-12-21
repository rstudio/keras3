# Configure the default training loop limits.

These helpers control the caps that built-in training loops obey when
running [`fit()`](https://generics.r-lib.org/reference/fit.html),
[`evaluate()`](https://rdrr.io/pkg/tensorflow/man/evaluate.html), or
[`predict()`](https://rdrr.io/r/stats/predict.html). The values can also
be provided via the `KERAS_MAX_EPOCHS` or `KERAS_MAX_STEPS_PER_EPOCH`
environment variables to quickly constrain a run without modifying
source code.

## Usage

``` r
config_max_epochs()

config_set_max_epochs(max_epochs)

config_max_steps_per_epoch()

config_set_max_steps_per_epoch(max_steps_per_epoch)
```

## Arguments

- max_epochs:

  Integer upper bound for epochs processed by built-in training loops.
  Use `NULL` to remove the cap.

- max_steps_per_epoch:

  Integer upper bound for steps processed per epoch by built-in training
  loops. Use `NULL` to remove the cap.

## Value

`config_max_epochs()` and `config_max_steps_per_epoch()` return the
current integer limits (or `NULL` if the cap is unset). The setter
variants return `NULL` invisibly and are called for side effects.

## See also

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
[`config_set_backend()`](https://keras3.posit.co/reference/config_set_backend.md)  
[`config_set_dtype_policy()`](https://keras3.posit.co/reference/config_set_dtype_policy.md)  
[`config_set_epsilon()`](https://keras3.posit.co/reference/config_set_epsilon.md)  
[`config_set_floatx()`](https://keras3.posit.co/reference/config_set_floatx.md)  
[`config_set_image_data_format()`](https://keras3.posit.co/reference/config_set_image_data_format.md)  

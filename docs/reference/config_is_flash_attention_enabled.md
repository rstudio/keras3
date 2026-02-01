# Checks whether flash attention is globally enabled in Keras.

Flash attention is a performance-optimized method for computing
attention in large models, such as transformers, allowing for faster and
more memory-efficient operations. This function checks the global Keras
configuration to determine if flash attention is enabled for compatible
layers (e.g., `MultiHeadAttention`).

Note that enabling flash attention does not guarantee it will always be
used. Typically, the inputs must be in `float16` or `bfloat16` dtype,
and input layout requirements may vary depending on the backend.

## Usage

``` r
config_is_flash_attention_enabled()
```

## Value

`FALSE` if disabled; otherwise, it indicates that it is enabled.

## See also

[`config_disable_flash_attention()`](https://keras3.posit.co/reference/config_disable_flash_attention.md)
[`config_enable_flash_attention()`](https://keras3.posit.co/reference/config_enable_flash_attention.md)

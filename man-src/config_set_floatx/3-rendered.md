Set the default float dtype.

@description

# Note
It is not recommended to set this to `"float16"` for training,
as this will likely cause numeric stability issues.
Instead, mixed precision, which leverages
a mix of `float16` and `float32`. It can be configured by calling
`keras::keras$mixed_precision$set_dtype_policy('mixed_float16')`.

# Examples

```r
config_floatx()
```

```
## [1] "float32"
```


```r
config_set_floatx('float64')
config_floatx()
```

```
## [1] "float64"
```


```r
# Set it back to float32
config_set_floatx('float32')
```

# Raises
    ValueError: In case of invalid value.

@param value String; `'float16'`, `'float32'`, or `'float64'`.

@export
@seealso
+ <https:/keras.io/keras_core/api/utils/config_utils#setfloatx-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/config/set_floatx>

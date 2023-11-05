Set the default float dtype.

@description

# Note
It is not recommended to set this to `"float16"` for training,
as this will likely cause numeric stability issues.
Instead, mixed precision, which leverages
a mix of `float16` and `float32`. It can be configured by calling
`keras.mixed_precision.set_dtype_policy('mixed_float16')`.

# Examples
```python
keras.config.floatx()
# 'float32'
```

```python
keras.config.set_floatx('float64')
keras.config.floatx()
# 'float64'
```

```python
# Set it back to float32
keras.config.set_floatx('float32')
```

# Raises
    ValueError: In case of invalid value.

@param value String; `'float16'`, `'float32'`, or `'float64'`.

@export
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/config/set_floatx>

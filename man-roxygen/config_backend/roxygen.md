Publicly accessible method for determining the current backend.

@description

# Returns
String, the name of the backend Keras is currently using. One of
`"tensorflow"`, `"torch"`, or `"jax"`.

# Examples
```python
keras.config.backend()
# 'tensorflow'
```

@export
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/config/backend>

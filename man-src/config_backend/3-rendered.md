Publicly accessible method for determining the current backend.

@description

# Examples

```r
keras::config_backend()
```

```
## [1] "tensorflow"
```

```r
# 'tensorflow'
```

@returns
String, the name of the backend Keras is currently using. One of
`"tensorflow"`, `"torch"`, or `"jax"`.

@export
@family config backend
@family backend
@family config
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/config/backend>

Set the value of the fuzz factor used in numeric expressions.

@description

# Examples

```r
config_epsilon()
```

```
## [1] 1e-07
```


```r
config_set_epsilon(1e-5)
config_epsilon()
```

```
## [1] 1e-05
```


```r
# Set it back to the default value.
config_set_epsilon(1e-7)
```

@param value
float. New value of epsilon.

@export
@seealso
+ <https:/keras.io/keras_core/api/utils/config_utils#setepsilon-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/config/set_epsilon>

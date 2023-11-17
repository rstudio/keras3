Compute the (weighted) mean of the given values.

@description
For example, if values is `c(1, 3, 5, 7)` then the mean is 4.
If `sample_weight` was specified as `c(1, 1, 0, 0)` then the mean would be 2.

This metric creates two variables, `total` and `count`.
The mean value returned is simply `total` divided by `count`.

# Examples

```r
m <- metric_mean()
m$update_state(c(1, 3, 5, 7))
m$result()
```

```
## tf.Tensor(4.0, shape=(), dtype=float32)
```


```r
m$reset_state()
m$update_state(c(1, 3, 5, 7), sample_weight = c(1, 1, 0, 0))
m$result()
```

```
## tf.Tensor(2.0, shape=(), dtype=float32)
```
```

@param name
(Optional) string name of the metric instance.

@param dtype
(Optional) data type of the metric result.

@param ...
Passed on to the Python callable

@export
@family reduction metrics
@family metrics
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Mean>


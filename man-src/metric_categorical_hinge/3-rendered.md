Computes the categorical hinge metric between `y_true` and `y_pred`.

@description
Formula:


```r
loss <- maximum(neg - pos + 1, 0)
```

where `neg=maximum((1-y_true)*y_pred)` and `pos=sum(y_true*y_pred)`

# Usage
Standalone usage:

```r
m <- metric_categorical_hinge()
m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(0.6, 0.4), c(0.4, 0.6)))
m$result()
```

```
## tf.Tensor(1.4000001, shape=(), dtype=float32)
```

```r
m$reset_state()
m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(0.6, 0.4), c(0.4, 0.6)),
               sample_weight = c(1, 0))
m$result()
```

```
## tf.Tensor(1.2, shape=(), dtype=float32)
```

@returns
Categorical hinge loss values with shape = `[batch_size, d0, .. dN-1]`.

@param name
(Optional) string name of the metric instance.

@param dtype
(Optional) data type of the metric result.

@param y_true
The ground truth values. `y_true` values are expected to be
either `{-1, +1}` or `{0, 1}` (i.e. a one-hot-encoded tensor) with
shape = `[batch_size, d0, .. dN]`.

@param y_pred
The predicted values with shape = `[batch_size, d0, .. dN]`.

@param ...
Passed on to the Python callable

@export
@family losses
@family metrics
@family hinge metrics
@seealso
+ <https:/keras.io/api/metrics/hinge_metrics#categoricalhinge-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CategoricalHinge>


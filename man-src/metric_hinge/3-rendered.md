Computes the hinge metric between `y_true` and `y_pred`.

@description
Formula:


```r
loss <- mean(maximum(1 - y_true * y_pred, 0), axis=-1)
```

`y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
provided we will convert them to -1 or 1.

# Usage
Standalone usage:


```r
m <- metric_hinge()
m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(0.6, 0.4), c(0.4, 0.6)))
m$result()
```

```
## tf.Tensor(1.3, shape=(), dtype=float32)
```

```r
m$reset_state()
m$update_state(rbind(c(0, 1), c(0, 0)), rbind(c(0.6, 0.4), c(0.4, 0.6)),
               sample_weight = c(1, 0))
m$result()
```

```
## tf.Tensor(1.1, shape=(), dtype=float32)
```

@param name
(Optional) string name of the metric instance.

@param dtype
(Optional) data type of the metric result.

@param y_true
The ground truth values. `y_true` values are expected to be -1
or 1. If binary (0 or 1) labels are provided they will be converted
to -1 or 1 with shape = `[batch_size, d0, .. dN]`.

@param y_pred
The predicted values with shape = `[batch_size, d0, .. dN]`.

@param ...
Passed on to the Python callable

@export
@family metrics
@seealso
+ <https:/keras.io/api/metrics/hinge_metrics#hinge-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Hinge>


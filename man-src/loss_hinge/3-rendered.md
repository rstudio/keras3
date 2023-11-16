Computes the hinge loss between `y_true` & `y_pred`.

@description
Formula:


```r
loss <- mean(maximum(1 - y_true * y_pred, 0), axis=-1)
```

`y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
provided we will convert them to -1 or 1.

# Examples

```r
y_true <- array(sample(c(-1,1), 6, replace = TRUE), dim = c(2, 3))
y_pred <- random_uniform(c(2, 3))
loss <- loss_hinge(y_true, y_pred)
loss
```

```
## tf.Tensor([1.0610152  0.93285507], shape=(2), dtype=float32)
```

@returns
Hinge loss values with shape = `[batch_size, d0, .. dN-1]`.

@param reduction
Type of reduction to apply to the loss. In almost all cases
this should be `"sum_over_batch_size"`.
Supported options are `"sum"`, `"sum_over_batch_size"` or `NULL`.

@param name
Optional name for the loss instance.

@param y_true
The ground truth values. `y_true` values are expected to be -1
or 1. If binary (0 or 1) labels are provided they will be converted
to -1 or 1 with shape = `[batch_size, d0, .. dN]`.

@param y_pred
The predicted values with shape = `[batch_size, d0, .. dN]`.

@param ...
Passed on to the Python callable

@export
@family losses
@seealso
+ <https:/keras.io/api/losses/hinge_losses#hinge-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/losses/Hinge>


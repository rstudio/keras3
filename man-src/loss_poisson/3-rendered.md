Computes the Poisson loss between `y_true` & `y_pred`.

@description
Formula:


```r
loss <- y_pred - y_true * log(y_pred)
```

# Examples

```r
y_true <- random_uniform(c(2, 3), 0, 2)
y_pred <- random_uniform(c(2, 3))
loss <- loss_poisson(y_true, y_pred)
loss
```

```
## tf.Tensor([2.5907533  0.66836613], shape=(2), dtype=float32)
```

@returns
Poisson loss values with shape = `[batch_size, d0, .. dN-1]`.

@param reduction
Type of reduction to apply to the loss. In almost all cases
this should be `"sum_over_batch_size"`.
Supported options are `"sum"`, `"sum_over_batch_size"` or `NULL`.

@param name
Optional name for the loss instance.

@param y_true
Ground truth values. shape = `[batch_size, d0, .. dN]`.

@param y_pred
The predicted values. shape = `[batch_size, d0, .. dN]`.

@param ...
Passed on to the Python callable

@export
@family losses
@seealso
+ <https:/keras.io/api/losses/probabilistic_losses#poisson-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/losses/Poisson>


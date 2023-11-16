Computes the mean squared logarithmic error between `y_true` & `y_pred`.

@description
Note that `y_pred` and `y_true` cannot be less or equal to 0. Negative
values and 0 values will be replaced with `keras.backend.epsilon()`
(default to `1e-7`).

Formula:


```r
loss <- mean(square(log(y_true + 1) - log(y_pred + 1)))
```

# Examples

```r
y_true <- random_uniform(c(2, 3), 0, 2)
y_pred <- random_uniform(c(2, 3))
loss <- loss_mean_squared_logarithmic_error(y_true, y_pred)
```

@returns
Mean squared logarithmic error values with shape = `[batch_size, d0, .. dN-1]`.

@param reduction
Type of reduction to apply to the loss. In almost all cases
this should be `"sum_over_batch_size"`.
Supported options are `"sum"`, `"sum_over_batch_size"` or `NULL`.

@param name
Optional name for the loss instance.

@param y_true
Ground truth values with shape = `[batch_size, d0, .. dN]`.

@param y_pred
The predicted values with shape = `[batch_size, d0, .. dN]`.

@param ...
Passed on to the Python callable

@export
@family losses
@seealso
+ <https:/keras.io/api/losses/regression_losses#meansquaredlogarithmicerror-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredLogarithmicError>


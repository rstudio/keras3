Computes the cosine similarity between `y_true` & `y_pred`.

@description
Formula:

```r
loss <- -sum(l2_norm(y_true) * l2_norm(y_pred))
```

Note that it is a number between -1 and 1. When it is a negative number
between -1 and 0, 0 indicates orthogonality and values closer to -1
indicate greater similarity. This makes it usable as a loss function in a
setting where you try to maximize the proximity between predictions and
targets. If either `y_true` or `y_pred` is a zero vector, cosine
similarity will be 0 regardless of the proximity between predictions
and targets.

# Examples

```r
y_true <- rbind(c(0., 1.), c(1., 1.), c(1., 1.))
y_pred <- rbind(c(1., 0.), c(1., 1.), c(-1., -1.))
loss <- loss_cosine_similarity(y_true, y_pred, axis=-1)
loss
```

```
## tf.Tensor([-0. -1.  1.], shape=(3), dtype=float64)
```

@returns
Cosine similarity tensor.

@param axis
The axis along which the cosine similarity is computed
(the features axis). Defaults to `-1`.

@param reduction
Type of reduction to apply to the loss. In almost all cases
this should be `"sum_over_batch_size"`.
Supported options are `"sum"`, `"sum_over_batch_size"` or `NULL`.

@param name
Optional name for the loss instance.

@param y_true
Tensor of true targets.

@param y_pred
Tensor of predicted targets.

@param ...
Passed on to the Python callable

@export
@family losses
@seealso
+ <https:/keras.io/api/losses/regression_losses#cosinesimilarity-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/losses/CosineSimilarity>


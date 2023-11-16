Computes the crossentropy loss between the labels and predictions.

@description
Use this crossentropy loss function when there are two or more label
classes.  We expect labels to be provided as integers. If you want to
provide labels using `one-hot` representation, please use
`CategoricalCrossentropy` loss.  There should be `# classes` floating point
values per feature for `y_pred` and a single floating point value per
feature for `y_true`.

In the snippet below, there is a single floating point value per example for
`y_true` and `num_classes` floating pointing values per example for
`y_pred`. The shape of `y_true` is `[batch_size]` and the shape of `y_pred`
is `[batch_size, num_classes]`.

# Examples

```r
y_true <- c(1, 2)
y_pred <- rbind(c(0.05, 0.95, 0), c(0.1, 0.8, 0.1))
loss <- loss_sparse_categorical_crossentropy(y_true, y_pred)
loss
```

```
## tf.Tensor([0.05129339 2.30258509], shape=(2), dtype=float64)
```

```r
y_true <- c(1, 2)
y_pred <- rbind(c(0.05, 0.95, 0), c(0.1, 0.8, 0.1))
# Using 'auto'/'sum_over_batch_size' reduction type.
scce <- loss_sparse_categorical_crossentropy()
scce(k_array(y_true), k_array(y_pred))
```

```
## tf.Tensor(1.1769392, shape=(), dtype=float32)
```

```r
# 1.177
```


```r
# Calling with 'sample_weight'.
scce(k_array(y_true), k_array(y_pred), sample_weight = k_array(c(0.3, 0.7)))
```

```
## tf.Tensor(0.8135988, shape=(), dtype=float32)
```


```r
# Using 'sum' reduction type.
scce <- loss_sparse_categorical_crossentropy(reduction="sum")
scce(k_array(y_true), k_array(y_pred))
```

```
## tf.Tensor(2.3538785, shape=(), dtype=float32)
```

```r
# 2.354
```


```r
# Using 'none' reduction type.
scce <- loss_sparse_categorical_crossentropy(reduction=NULL)
scce(k_array(y_true), k_array(y_pred))
```

```
## tf.Tensor([0.05129344 2.3025851 ], shape=(2), dtype=float32)
```

```r
# array([0.0513, 2.303], dtype=float32)
```

Usage with the `compile()` API:


```r
model %>% compile(optimizer = 'sgd',
                  loss = loss_sparse_categorical_crossentropy())
```

@returns
Sparse categorical crossentropy loss value.

@param from_logits
Whether `y_pred` is expected to be a logits tensor. By
default, we assume that `y_pred` encodes a probability distribution.

@param reduction
Type of reduction to apply to the loss. In almost all cases
this should be `"sum_over_batch_size"`.
Supported options are `"sum"`, `"sum_over_batch_size"` or `NULL`.

@param name
Optional name for the loss instance.

@param y_true
Ground truth values.

@param y_pred
The predicted values.

@param ignore_class
Optional integer. The ID of a class to be ignored during
loss computation. This is useful, for example, in segmentation
problems featuring a "void" class (commonly -1 or 255) in
segmentation maps. By default (`ignore_class=NULL`), all classes are
considered.

@param axis
Defaults to `-1`. The dimension along which the entropy is
computed.

@param ...
Passed on to the Python callable

@export
@family losses
@seealso
+ <https:/keras.io/api/losses/probabilistic_losses#sparsecategoricalcrossentropy-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy>


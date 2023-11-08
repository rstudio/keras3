keras.losses.SparseCategoricalCrossentropy
__signature__
(
  from_logits=False,
  ignore_class=None,
  reduction='sum_over_batch_size',
  name='sparse_categorical_crossentropy'
)
__doc__
Computes the crossentropy loss between the labels and predictions.

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

Args:
    from_logits: Whether `y_pred` is expected to be a logits tensor. By
        default, we assume that `y_pred` encodes a probability distribution.
    reduction: Type of reduction to apply to the loss. In almost all cases
        this should be `"sum_over_batch_size"`.
        Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
    name: Optional name for the loss instance.

Examples:

>>> y_true = [1, 2]
>>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
>>> # Using 'auto'/'sum_over_batch_size' reduction type.
>>> scce = keras.losses.SparseCategoricalCrossentropy()
>>> scce(y_true, y_pred)
1.177

>>> # Calling with 'sample_weight'.
>>> scce(y_true, y_pred, sample_weight=np.array([0.3, 0.7]))
0.814

>>> # Using 'sum' reduction type.
>>> scce = keras.losses.SparseCategoricalCrossentropy(
...     reduction="sum")
>>> scce(y_true, y_pred)
2.354

>>> # Using 'none' reduction type.
>>> scce = keras.losses.SparseCategoricalCrossentropy(
...     reduction=None)
>>> scce(y_true, y_pred)
array([0.0513, 2.303], dtype=float32)

Usage with the `compile()` API:

```python
model.compile(optimizer='sgd',
              loss=keras.losses.SparseCategoricalCrossentropy())
```

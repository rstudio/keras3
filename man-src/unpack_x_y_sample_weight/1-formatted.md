Unpacks user-provided data tuple.

@description
This is a convenience utility to be used when overriding
`Model.train_step`, `Model.test_step`, or `Model.predict_step`.
This utility makes it easy to support data of the form `(x,)`,
`(x, y)`, or `(x, y, sample_weight)`.

# Usage
Standalone usage:

```python
features_batch = ops.ones((10, 5))
labels_batch = ops.zeros((10, 5))
data = (features_batch, labels_batch)
# `y` and `sample_weight` will default to `None` if not provided.
x, y, sample_weight = unpack_x_y_sample_weight(data)
sample_weight is None
# True
```

@returns
The unpacked tuple, with `None`s for `y` and `sample_weight` if they are
not provided.

@param data A tuple of the form `(x,)`, `(x, y)`, or `(x, y, sample_weight)`.

@export
@family utils
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/utils/unpack_x_y_sample_weight>

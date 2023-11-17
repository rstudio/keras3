Packs user-provided data into a tuple.

@description
This is a convenience utility for packing data into the tuple formats
that `Model.fit()` uses.

# Usage
Standalone usage:

```python
x = ops.ones((10, 1))
data = pack_x_y_sample_weight(x)
isinstance(data, ops.Tensor)
# True
y = ops.ones((10, 1))
data = pack_x_y_sample_weight(x, y)
isinstance(data, tuple)
# True
x, y = data
```

@returns
    Tuple in the format used in `Model.fit()`.

@param x
Features to pass to `Model`.

@param y
Ground-truth targets to pass to `Model`.

@param sample_weight
Sample weight for each element.

@export
@family datum util adapter trainers
@family datum adapter trainers
@family trainers
@family utils
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/utils/pack_x_y_sample_weight>

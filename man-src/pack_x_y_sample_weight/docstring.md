Packs user-provided data into a tuple.

This is a convenience utility for packing data into the tuple formats
that `Model.fit()` uses.

Usage:

Standalone usage:

>>> x = ops.ones((10, 1))
>>> data = pack_x_y_sample_weight(x)
>>> isinstance(data, ops.Tensor)
True
>>> y = ops.ones((10, 1))
>>> data = pack_x_y_sample_weight(x, y)
>>> isinstance(data, tuple)
True
>>> x, y = data

Args:
    x: Features to pass to `Model`.
    y: Ground-truth targets to pass to `Model`.
    sample_weight: Sample weight for each element.

Returns:
    Tuple in the format used in `Model.fit()`.

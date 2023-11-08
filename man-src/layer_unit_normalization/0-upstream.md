keras.layers.UnitNormalization
__signature__
(axis=-1, **kwargs)
__doc__
Unit normalization layer.

Normalize a batch of inputs so that each input in the batch has a L2 norm
equal to 1 (across the axes specified in `axis`).

Example:

>>> data = np.arange(6).reshape(2, 3)
>>> normalized_data = keras.layers.UnitNormalization()(data)
>>> print(np.sum(normalized_data[0, :] ** 2)
1.0

Args:
    axis: Integer or list/tuple. The axis or axes to normalize across.
        Typically, this is the features axis or axes. The left-out axes are
        typically the batch axis or axes. `-1` is the last dimension
        in the input. Defaults to `-1`.

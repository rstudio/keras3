Splits a dataset into a left half and a right half (e.g. train / test).

Args:
    dataset:
        A `tf.data.Dataset`, a `torch.utils.data.Dataset` object,
        or a list/tuple of arrays with the same length.
    left_size: If float (in the range `[0, 1]`), it signifies
        the fraction of the data to pack in the left dataset. If integer, it
        signifies the number of samples to pack in the left dataset. If
        `None`, defaults to the complement to `right_size`.
        Defaults to `None`.
    right_size: If float (in the range `[0, 1]`), it signifies
        the fraction of the data to pack in the right dataset.
        If integer, it signifies the number of samples to pack
        in the right dataset.
        If `None`, defaults to the complement to `left_size`.
        Defaults to `None`.
    shuffle: Boolean, whether to shuffle the data before splitting it.
    seed: A random seed for shuffling.

Returns:
    A tuple of two `tf.data.Dataset` objects:
    the left and right splits.

Example:

>>> data = np.random.random(size=(1000, 4))
>>> left_ds, right_ds = keras.utils.split_dataset(data, left_size=0.8)
>>> int(left_ds.cardinality())
800
>>> int(right_ds.cardinality())
200

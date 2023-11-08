keras.ops.split
__signature__
(x, indices_or_sections, axis=0)
__doc__
Split a tensor into chunks.

Args:
    x: Input tensor.
    indices_or_sections: Either an integer indicating the number of
        sections along `axis` or a list of integers indicating the indices
        along `axis` at which the tensor is split.
    indices_or_sections: If an integer, N, the tensor will be split into N
        equal sections along `axis`. If a 1-D array of sorted integers,
        the entries indicate indices at which the tensor will be split
        along `axis`.
    axis: Axis along which to split. Defaults to `0`.

Note:
    A split does not have to result in equal division when using
    Torch backend.

Returns:
    A list of tensors.

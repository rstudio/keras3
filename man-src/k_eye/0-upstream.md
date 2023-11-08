keras.ops.eye
__signature__
(
  N,
  M=None,
  k=0,
  dtype=None
)
__doc__
Return a 2-D tensor with ones on the diagonal and zeros elsewhere.

Args:
    N: Number of rows in the output.
    M: Number of columns in the output. If `None`, defaults to `N`.
    k: Index of the diagonal: 0 (the default) refers to the main
        diagonal, a positive value refers to an upper diagonal,
        and a negative value to a lower diagonal.
    dtype: Data type of the returned tensor.

Returns:
    Tensor with ones on the k-th diagonal and zeros elsewhere.

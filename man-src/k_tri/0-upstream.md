keras.ops.tri
__signature__
(
  N,
  M=None,
  k=0,
  dtype=None
)
__doc__
Return a tensor with ones at and below a diagonal and zeros elsewhere.

Args:
    N: Number of rows in the tensor.
    M: Number of columns in the tensor.
    k: The sub-diagonal at and below which the array is filled.
        `k = 0` is the main diagonal, while `k < 0` is below it, and
        `k > 0` is above. The default is 0.
    dtype: Data type of the returned tensor. The default is "float32".

Returns:
    Tensor with its lower triangle filled with ones and zeros elsewhere.
    `T[i, j] == 1` for `j <= i + k`, 0 otherwise.

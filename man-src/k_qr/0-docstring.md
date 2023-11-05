Computes the QR decomposition of a tensor.

Args:
    x: Input tensor.
    mode: A string specifying the mode of the QR decomposition.
        - 'reduced': Returns the reduced QR decomposition. (default)
        - 'complete': Returns the complete QR decomposition.

Returns:
    A tuple containing two tensors. The first tensor represents the
    orthogonal matrix Q, and the second tensor represents the upper
    triangular matrix R.

Example:

>>> x = keras.ops.convert_to_tensor([[1., 2.], [3., 4.], [5., 6.]])
>>> q, r = qr(x)
>>> print(q)
array([[-0.16903079  0.897085]
       [-0.5070925   0.2760267 ]
       [-0.8451542  -0.34503305]], shape=(3, 2), dtype=float32)

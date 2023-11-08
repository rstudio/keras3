keras.ops.fft2
__signature__
(x)
__doc__
Computes the 2D Fast Fourier Transform along the last two axes of input.

Args:
    x: Tuple of the real and imaginary parts of the input tensor. Both
        tensors in the tuple should be of floating type.

Returns:
    A tuple containing two tensors - the real and imaginary parts of the
    output.

Example:

>>> x = (
...     keras.ops.convert_to_tensor([[1., 2.], [2., 1.]]),
...     keras.ops.convert_to_tensor([[0., 1.], [1., 0.]]),
... )
>>> fft2(x)
(array([[ 6.,  0.],
    [ 0., -2.]], dtype=float32), array([[ 2.,  0.],
    [ 0., -2.]], dtype=float32))

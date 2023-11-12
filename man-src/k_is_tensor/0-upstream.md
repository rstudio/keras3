keras.ops.is_tensor
__signature__
(x)
__doc__
Check whether the given object is a tensor.

Note: This checks for backend specific tensors so passing a TensorFlow
tensor would return `False` if your backend is PyTorch or JAX.

Args:
    x: A variable.

Returns:
    `True` if `x` is a tensor, otherwise `False`.

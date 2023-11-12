Check whether the given object is a tensor.

@description

# Note
This checks for backend specific tensors so passing a TensorFlow
tensor would return `False` if your backend is PyTorch or JAX.

@returns
    `True` if `x` is a tensor, otherwise `False`.

@param x A variable.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/is_tensor>

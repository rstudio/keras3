Reverse the order of elements in the tensor along the given axis.

@description
The shape of the tensor is preserved, but the elements are reordered.

@returns
    Output tensor with entries of `axis` reversed.

@param x Input tensor.
@param axis Axis or axes along which to flip the tensor. The default,
    `axis=None`, will flip over all of the axes of the input tensor.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/flip>

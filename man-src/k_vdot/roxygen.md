Return the dot product of two vectors.

@description
If the first argument is complex, the complex conjugate of the first
argument is used for the calculation of the dot product.

Multidimensional tensors are flattened before the dot product is taken.

# Returns
    Output tensor.

@param x1 First input tensor. If complex, its complex conjugate is taken
    before calculation of the dot product.
@param x2 Second input tensor.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/vdot>

Return the dot product of two vectors.

If the first argument is complex, the complex conjugate of the first
argument is used for the calculation of the dot product.

Multidimensional tensors are flattened before the dot product is taken.

Args:
    x1: First input tensor. If complex, its complex conjugate is taken
        before calculation of the dot product.
    x2: Second input tensor.

Returns:
    Output tensor.

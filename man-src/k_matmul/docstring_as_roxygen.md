Matrix product of two tensors.

@description
- If both tensors are 1-dimensional, the dot product (scalar) is returned.
- If either tensor is N-D, N > 2, it is treated as a stack of matrices
  residing in the last two indexes and broadcast accordingly.
- If the first tensor is 1-D, it is promoted to a matrix by prepending
  a 1 to its dimensions. After matrix multiplication the prepended
  1 is removed.
- If the second tensor is 1-D, it is promoted to a matrix by appending a 1
  to its dimensions. After matrix multiplication the appended 1 is removed.

# Returns
    Output tensor, matrix product of the inputs.

@param x1 First tensor.
@param x2 Second tensor.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/matmul>

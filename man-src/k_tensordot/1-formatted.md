Compute the tensor dot product along specified axes.

@description

# Returns
    The tensor dot product of the inputs.

@param x1 First tensor.
@param x2 Second tensor.
@param axes
- If an integer, N, sum over the last N axes of `x1` and the
        first N axes of `x2` in order. The sizes of the corresponding
        axes must match.
      - Or, a list of axes to be summed over, first sequence applying
        to `x1`, second to `x2`. Both sequences must be of the
        same length.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/tensordot>

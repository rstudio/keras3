keras.layers.ZeroPadding1D
__signature__
(padding=1, **kwargs)
__doc__
Zero-padding layer for 1D input (e.g. temporal sequence).

Examples:

>>> input_shape = (2, 2, 3)
>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
>>> x
[[[ 0  1  2]
  [ 3  4  5]]
 [[ 6  7  8]
  [ 9 10 11]]]
>>> y = keras.layers.ZeroPadding1D(padding=2)(x)
>>> y
[[[ 0  0  0]
  [ 0  0  0]
  [ 0  1  2]
  [ 3  4  5]
  [ 0  0  0]
  [ 0  0  0]]
 [[ 0  0  0]
  [ 0  0  0]
  [ 6  7  8]
  [ 9 10 11]
  [ 0  0  0]
  [ 0  0  0]]]

Args:
    padding: Int, or tuple of int (length 2), or dictionary.
        - If int: how many zeros to add at the beginning and end of
          the padding dimension (axis 1).
        - If tuple of 2 ints: how many zeros to add at the beginning and the
          end of the padding dimension (`(left_pad, right_pad)`).

Input shape:
    3D tensor with shape `(batch_size, axis_to_pad, features)`

Output shape:
    3D tensor with shape `(batch_size, padded_axis, features)`

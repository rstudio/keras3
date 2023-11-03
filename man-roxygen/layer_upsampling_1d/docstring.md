Upsampling layer for 1D inputs.

Repeats each temporal step `size` times along the time axis.

Examples:

>>> input_shape = (2, 2, 3)
>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
>>> x
[[[ 0  1  2]
  [ 3  4  5]]
 [[ 6  7  8]
  [ 9 10 11]]]
>>> y = keras.layers.UpSampling1D(size=2)(x)
>>> y
[[[ 0.  1.  2.]
  [ 0.  1.  2.]
  [ 3.  4.  5.]
  [ 3.  4.  5.]]

 [[ 6.  7.  8.]
  [ 6.  7.  8.]
  [ 9. 10. 11.]
  [ 9. 10. 11.]]]

Args:
    size: Integer. Upsampling factor.

Input shape:
    3D tensor with shape: `(batch_size, steps, features)`.

Output shape:
    3D tensor with shape: `(batch_size, upsampled_steps, features)`.

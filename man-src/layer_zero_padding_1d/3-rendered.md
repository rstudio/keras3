Zero-padding layer for 1D input (e.g. temporal sequence).

@description

# Examples

```r
input_shape <- c(2, 2, 3)
x <- k_reshape(seq_len(prod(input_shape)), input_shape)
x
```

```
## tf.Tensor(
## [[[ 1  2  3]
##   [ 4  5  6]]
##
##  [[ 7  8  9]
##   [10 11 12]]], shape=(2, 2, 3), dtype=int64)
```

```r
y <- layer_zero_padding_1d(x, padding = 2)
y
```

```
## tf.Tensor(
## [[[ 0  0  0]
##   [ 0  0  0]
##   [ 1  2  3]
##   [ 4  5  6]
##   [ 0  0  0]
##   [ 0  0  0]]
##
##  [[ 0  0  0]
##   [ 0  0  0]
##   [ 7  8  9]
##   [10 11 12]
##   [ 0  0  0]
##   [ 0  0  0]]], shape=(2, 6, 3), dtype=int64)
```

# Input Shape
3D tensor with shape `(batch_size, axis_to_pad, features)`

# Output Shape
    3D tensor with shape `(batch_size, padded_axis, features)`

@param padding
Int, or list of int (length 2), or named listionary.
- If int: how many zeros to add at the beginning and end of
  the padding dimension (axis 1).
- If list of 2 ints: how many zeros to add at the beginning and the
  end of the padding dimension (`(left_pad, right_pad)`).

@param object
Object to compose the layer with. A tensor, array, or sequential model.

@param ...
Passed on to the Python callable

@export
@family reshaping layers
@family layers
@seealso
+ <https:/keras.io/api/layers/reshaping_layers/zero_padding1d#zeropadding1d-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding1D>


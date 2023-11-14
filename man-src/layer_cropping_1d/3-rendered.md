Cropping layer for 1D input (e.g. temporal sequence).

@description
It crops along the time dimension (axis 1).

# Examples

```r
input_shape <- c(2, 3, 2)
x <- k_arange(prod(input_shape)) |> k_reshape(input_shape)
x
```

```
## tf.Tensor(
## [[[ 0.  1.]
##   [ 2.  3.]
##   [ 4.  5.]]
##
##  [[ 6.  7.]
##   [ 8.  9.]
##   [10. 11.]]], shape=(2, 3, 2), dtype=float64)
```

```r
y <- x |> layer_cropping_1d(cropping = 1)
y
```

```
## tf.Tensor(
## [[[2. 3.]]
##
##  [[8. 9.]]], shape=(2, 1, 2), dtype=float32)
```

# Input Shape
3D tensor with shape `(batch_size, axis_to_crop, features)`

# Output Shape
    3D tensor with shape `(batch_size, cropped_axis, features)`

@param cropping
Int, or list of int (length 2).
- If int: how many units should be trimmed off at the beginning and
  end of the cropping dimension (axis 1).
- If list of 2 ints: how many units should be trimmed off at the
  beginning and end of the cropping dimension
  (`(left_crop, right_crop)`).

@param object
Object to compose the layer with. A tensor, array, or sequential model.

@param ...
Passed on to the Python callable

@export
@family cropping1d reshaping layers
@family reshaping layers
@family layers
@seealso
+ <https:/keras.io/api/layers/reshaping_layers/cropping1d#cropping1d-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Cropping1D>

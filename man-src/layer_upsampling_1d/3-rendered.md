Upsampling layer for 1D inputs.

@description
Repeats each temporal step `size` times along the time axis.

# Examples

```r
input_shape <- c(2, 2, 3)
x <- seq_len(prod(input_shape)) %>% k_reshape(input_shape)
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
y <- layer_upsampling_1d(x, size = 2)
y
```

```
## tf.Tensor(
## [[[ 1  2  3]
##   [ 1  2  3]
##   [ 4  5  6]
##   [ 4  5  6]]
##
##  [[ 7  8  9]
##   [ 7  8  9]
##   [10 11 12]
##   [10 11 12]]], shape=(2, 4, 3), dtype=int64)
```

 `[[ 6.  7.  8.]`
  `[ 6.  7.  8.]`
  `[ 9. 10. 11.]`
  `[ 9. 10. 11.]]]`

# Input Shape
3D tensor with shape: `(batch_size, steps, features)`.

# Output Shape
    3D tensor with shape: `(batch_size, upsampled_steps, features)`.

@param size
Integer. Upsampling factor.

@param object
Object to compose the layer with. A tensor, array, or sequential model.

@param ...
Passed on to the Python callable

@export
@family reshaping layers
@family layers
@seealso
+ <https:/keras.io/api/layers/reshaping_layers/up_sampling1d#upsampling1d-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/UpSampling1D>


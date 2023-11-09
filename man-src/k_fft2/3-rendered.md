Computes the 2D Fast Fourier Transform along the last two axes of input.

@description

# Examples

```r
x <- c(k_array(rbind(c(1, 2),
                     c(2, 1))),
       k_array(rbind(c(0, 1),
                     c(1, 0))))
k_fft2(x)
```

```
## [[1]]
## tf.Tensor(
## [[ 6.  0.]
##  [ 0. -2.]], shape=(2, 2), dtype=float64)
## 
## [[2]]
## tf.Tensor(
## [[ 2.  0.]
##  [ 0. -2.]], shape=(2, 2), dtype=float64)
```

@returns
A list containing two tensors - the real and imaginary parts of the
output.

@param x list of the real and imaginary parts of the input tensor. Both
tensors provided should be of floating type.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/fft#fft2-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/fft2>

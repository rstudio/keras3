Inverse real-valued Fast Fourier transform along the last axis.

@description
Computes the inverse 1D Discrete Fourier Transform of a real-valued signal
over the inner-most dimension of input.

The inner-most dimension of the input is assumed to be the result of RFFT:
the `fft_length / 2 + 1` unique components of the DFT of a real-valued
signal. If `fft_length` is not provided, it is computed from the size of the
inner-most dimension of the input `(fft_length = 2 * (inner - 1))`. If the
FFT length used to compute is odd, it should be provided since it cannot
be inferred properly.

Along the axis IRFFT is computed on, if `fft_length / 2 + 1` is smaller than
the corresponding dimension of the input, the dimension is cropped. If it is
larger, the dimension is padded with zeros.

# Examples

```r
real <- k_array(c(0, 1, 2, 3, 4))
imag <- k_array(c(0, 1, 2, 3, 4))
k_irfft(c(real, imag))
```

```
## tf.Tensor(
## [ 2.         -2.06066017  0.5        -0.35355339  0.          0.06066017
##  -0.5         0.35355339], shape=(8), dtype=float64)
```

```r
# array([0.66666667, -0.9106836, 0.24401694))
```


```r
k_irfft(k_rfft(real, 5), 5)
```

```
## tf.Tensor([0. 1. 2. 3. 4.], shape=(5), dtype=float64)
```

@returns
A tensor containing the inverse real-valued Fast Fourier Transform
along the last axis of `x`.

@param x List of the real and imaginary parts of the input tensor. Both
    tensors in the list should be of floating type.
@param fft_length An integer representing the number of the fft length. If not
    specified, it is inferred from the length of the last axis of `x`.
    Defaults to `NULL`.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/fft#irfft-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/irfft>

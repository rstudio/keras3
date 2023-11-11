Short-Time Fourier Transform along the last axis of the input.

@description
The STFT computes the Fourier transform of short overlapping windows of the
input. This giving frequency components of the signal as they change over
time.

# Examples

```r
x <- k_array(c(0, 1, 2, 3, 4))
k_stft(x, 3, 2, 3)
```

```
## [[1]]
## tf.Tensor(
## [[ 0.  0.]
##  [ 2. -1.]
##  [ 4. -2.]], shape=(3, 2), dtype=float64)
##
## [[2]]
## tf.Tensor(
## [[ 0.          0.        ]
##  [ 0.         -1.73205081]
##  [ 0.         -3.46410162]], shape=(3, 2), dtype=float64)
```

@returns
A list containing two tensors - the real and imaginary parts of the
STFT output.

@param x Input tensor.
@param sequence_length An integer representing the sequence length.
@param sequence_stride An integer representing the sequence hop size.
@param fft_length An integer representing the size of the FFT to apply. If not
    specified, uses the smallest power of 2 enclosing `sequence_length`.
@param window A string, a tensor of the window or `NULL`. If `window` is a
    string, available values are `"hann"` and `"hamming"`. If `window`
    is a tensor, it will be used directly as the window and its length
    must be `sequence_length`. If `window` is `NULL`, no windowing is
    used. Defaults to `"hann"`.
@param center Whether to pad `x` on both sides so that the t-th sequence is
    centered at time `t * sequence_stride`. Otherwise, the t-th sequence
    begins at time `t * sequence_stride`. Defaults to `TRUE`.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/fft#stft-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/stft>


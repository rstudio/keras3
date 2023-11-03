Real-valued Fast Fourier Transform along the last axis of the input.

@description
Computes the 1D Discrete Fourier Transform of a real-valued signal over the
inner-most dimension of input.

Since the Discrete Fourier Transform of a real-valued signal is
Hermitian-symmetric, RFFT only returns the `fft_length / 2 + 1` unique
components of the FFT: the zero-frequency term, followed by the
`fft_length / 2` positive-frequency terms.

Along the axis RFFT is computed on, if `fft_length` is smaller than the
corresponding dimension of the input, the dimension is cropped. If it is
larger, the dimension is padded with zeros.

# Returns
A tuple containing two tensors - the real and imaginary parts of the
output.

# Examples
```python
x = keras.ops.convert_to_tensor([0.0, 1.0, 2.0, 3.0, 4.0])
rfft(x)
# (array([10.0, -2.5, -2.5]), array([0.0, 3.4409548, 0.81229924]))
```

```python
rfft(x, 3)
# (array([3.0, -1.5]), array([0.0, 0.8660254]))
```

@param x Input tensor.
@param fft_length An integer representing the number of the fft length. If not
    specified, it is inferred from the length of the last axis of `x`.
    Defaults to `None`.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/rfft>

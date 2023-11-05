Inverse real-valued Fast Fourier transform along the last axis.

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

Args:
    x: Tuple of the real and imaginary parts of the input tensor. Both
        tensors in the tuple should be of floating type.
    fft_length: An integer representing the number of the fft length. If not
        specified, it is inferred from the length of the last axis of `x`.
        Defaults to `None`.

Returns:
    A tensor containing the inverse real-valued Fast Fourier Transform
    along the last axis of `x`.

Examples:

>>> real = keras.ops.convert_to_tensor([0.0, 1.0, 2.0, 3.0, 4.0])
>>> imag = keras.ops.convert_to_tensor([0.0, 1.0, 2.0, 3.0, 4.0])
>>> irfft((real, imag))
array([0.66666667, -0.9106836, 0.24401694])

>>> irfft(rfft(real, 5), 5)
array([0.0, 1.0, 2.0, 3.0, 4.0])

k_stft <-
function (x, sequence_length, sequence_stride, fft_length, window = "hann", 
    center = TRUE) 
{
    args <- capture_args2(list(sequence_length = as_integer, 
        sequence_stride = as_integer, fft_length = as_integer))
    do.call(keras$ops$stft, args)
}

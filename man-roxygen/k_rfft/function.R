k_rfft <-
function (x, fft_length = NULL) 
{
    args <- capture_args2(list(fft_length = as_integer))
    do.call(keras$ops$rfft, args)
}

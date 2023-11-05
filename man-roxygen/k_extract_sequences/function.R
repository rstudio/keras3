k_extract_sequences <-
function (x, sequence_length, sequence_stride) 
{
    args <- capture_args2(list(sequence_length = as_integer, 
        sequence_stride = as_integer))
    do.call(keras$ops$extract_sequences, args)
}

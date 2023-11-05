pad_sequences <-
function (sequences, maxlen = NULL, dtype = "int32", padding = "pre", 
    truncating = "pre", value = 0) 
{
    args <- capture_args2(list(maxlen = as_integer))
    do.call(keras$utils$pad_sequences, args)
}

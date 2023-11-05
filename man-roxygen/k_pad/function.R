k_pad <-
function (x, pad_width, mode = "constant") 
{
    args <- capture_args2(list(pad_width = as_integer))
    do.call(keras$ops$pad, args)
}

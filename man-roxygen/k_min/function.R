k_min <-
function (x, axis = NULL, keepdims = FALSE, initial = NULL) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$min, args)
}

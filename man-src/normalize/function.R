normalize <-
function (x, axis = -1L, order = 2L) 
{
    args <- capture_args2(list(axis = as_axis, order = as_integer))
    do.call(keras$utils$normalize, args)
}

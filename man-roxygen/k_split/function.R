k_split <-
function (x, indices_or_sections, axis = 0L) 
{
    args <- capture_args2(list(indices_or_sections = as_integer, 
        axis = as_axis))
    do.call(keras$ops$split, args)
}

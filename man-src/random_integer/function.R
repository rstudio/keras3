random_integer <-
function (shape, minval, maxval, dtype = "int32", seed = NULL) 
{
    args <- capture_args2(list(shape = normalize_shape, seed = as_integer, 
        maxval = function (x) 
        as_integer(ceiling(x)), minval = as_integer))
    do.call(keras$random$randint, args)
}

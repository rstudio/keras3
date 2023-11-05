random_shuffle <-
function (x, axis = 0L, seed = NULL) 
{
    args <- capture_args2(list(axis = as_axis, seed = as_integer))
    do.call(keras$random$shuffle, args)
}

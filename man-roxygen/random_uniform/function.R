random_uniform <-
function (shape, minval = 0, maxval = 1, dtype = NULL, seed = NULL) 
{
    args <- capture_args2(list(shape = normalize_shape, seed = as_integer))
    do.call(keras$random$uniform, args)
}

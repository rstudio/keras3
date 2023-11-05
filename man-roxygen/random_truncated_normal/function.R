random_truncated_normal <-
function (shape, mean = 0, stddev = 1, dtype = NULL, seed = NULL) 
{
    args <- capture_args2(list(shape = normalize_shape, seed = as_integer))
    do.call(keras$random$truncated_normal, args)
}

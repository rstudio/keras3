random_dropout <-
function (inputs, rate, noise_shape = NULL, seed = NULL) 
{
    args <- capture_args2(list(seed = as_integer, noise_shape = normalize_shape))
    do.call(keras$random$dropout, args)
}

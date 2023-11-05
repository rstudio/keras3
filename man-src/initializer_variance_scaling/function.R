initializer_variance_scaling <-
function (scale = 1, mode = "fan_in", distribution = "truncated_normal", 
    seed = NULL) 
{
    args <- capture_args2(list(seed = as_integer))
    do.call(keras$initializers$VarianceScaling, args)
}

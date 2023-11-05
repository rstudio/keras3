activation_leaky_relu <-
structure(function (x, negative_slope = 0.2) 
{
    args <- capture_args2(NULL)
    do.call(keras$activations$leaky_relu, args)
}, py_function_name = "leaky_relu")

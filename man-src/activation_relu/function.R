activation_relu <-
structure(function (x, negative_slope = 0, max_value = NULL, 
    threshold = 0) 
{
    args <- capture_args2(NULL)
    do.call(keras$activations$relu, args)
}, py_function_name = "relu")

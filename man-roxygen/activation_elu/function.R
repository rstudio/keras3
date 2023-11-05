activation_elu <-
structure(function (x, alpha = 1) 
{
    args <- capture_args2(NULL)
    do.call(keras$activations$elu, args)
}, py_function_name = "elu")

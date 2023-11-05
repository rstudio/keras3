activation_selu <-
structure(function (x) 
{
    args <- capture_args2(NULL)
    do.call(keras$activations$selu, args)
}, py_function_name = "selu")

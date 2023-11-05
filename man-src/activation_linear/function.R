activation_linear <-
structure(function (x) 
{
    args <- capture_args2(NULL)
    do.call(keras$activations$linear, args)
}, py_function_name = "linear")

activation_mish <-
structure(function (x) 
{
    args <- capture_args2(NULL)
    do.call(keras$activations$mish, args)
}, py_function_name = "mish")

activation_sigmoid <-
structure(function (x) 
{
    args <- capture_args2(NULL)
    do.call(keras$activations$sigmoid, args)
}, py_function_name = "sigmoid")

activation_softplus <-
structure(function (x) 
{
    args <- capture_args2(NULL)
    do.call(keras$activations$softplus, args)
}, py_function_name = "softplus")

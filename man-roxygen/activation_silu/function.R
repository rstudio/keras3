activation_silu <-
structure(function (x) 
{
    args <- capture_args2(NULL)
    do.call(keras$activations$silu, args)
}, py_function_name = "silu")

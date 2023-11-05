activation_tanh <-
structure(function (x) 
{
    args <- capture_args2(NULL)
    do.call(keras$activations$tanh, args)
}, py_function_name = "tanh")

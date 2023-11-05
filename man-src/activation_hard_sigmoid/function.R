activation_hard_sigmoid <-
structure(function (x) 
{
    args <- capture_args2(NULL)
    do.call(keras$activations$hard_sigmoid, args)
}, py_function_name = "hard_sigmoid")

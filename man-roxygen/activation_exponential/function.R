activation_exponential <-
structure(function (x) 
{
    args <- capture_args2(NULL)
    do.call(keras$activations$exponential, args)
}, py_function_name = "exponential")

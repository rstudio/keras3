activation_gelu <-
structure(function (x, approximate = FALSE) 
{
    args <- capture_args2(NULL)
    do.call(keras$activations$gelu, args)
}, py_function_name = "gelu")

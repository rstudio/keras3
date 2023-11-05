activation_relu6 <-
structure(function (x) 
{
    args <- capture_args2(NULL)
    do.call(keras$activations$relu6, args)
}, py_function_name = "relu6")

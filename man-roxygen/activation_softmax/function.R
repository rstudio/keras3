activation_softmax <-
structure(function (x, axis = -1L) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$activations$softmax, args)
}, py_function_name = "softmax")

activation_log_softmax <-
structure(function (x, axis = -1L) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$activations$log_softmax, args)
}, py_function_name = "log_softmax")

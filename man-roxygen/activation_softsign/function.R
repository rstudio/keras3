activation_softsign <-
structure(function (x) 
{
    args <- capture_args2(NULL)
    do.call(keras$activations$softsign, args)
}, py_function_name = "softsign")

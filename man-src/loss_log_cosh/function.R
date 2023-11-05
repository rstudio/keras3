loss_log_cosh <-
structure(function (y_true, y_pred, ..., reduction = "sum_over_batch_size", 
    name = "log_cosh") 
{
    args <- capture_args2(NULL)
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$losses$LogCosh
    else keras$losses$log_cosh
    do.call(callable, args)
}, py_function_name = "log_cosh")

loss_mean_absolute_error <-
structure(function (y_true, y_pred, ..., reduction = "sum_over_batch_size", 
    name = "mean_absolute_error") 
{
    args <- capture_args2(NULL)
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$losses$MeanAbsoluteError
    else keras$losses$mean_absolute_error
    do.call(callable, args)
}, py_function_name = "mean_absolute_error")

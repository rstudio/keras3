metric_mean_absolute_error <-
structure(function (y_true, y_pred, ..., name = "mean_absolute_error", 
    dtype = NULL) 
{
    args <- capture_args2(NULL)
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$metrics$MeanAbsoluteError
    else keras$metrics$mean_absolute_error
    do.call(callable, args)
}, py_function_name = "mean_absolute_error")

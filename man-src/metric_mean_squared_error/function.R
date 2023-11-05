metric_mean_squared_error <-
structure(function (y_true, y_pred, ..., name = "mean_squared_error", 
    dtype = NULL) 
{
    args <- capture_args2(NULL)
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$metrics$MeanSquaredError
    else keras$metrics$mean_squared_error
    do.call(callable, args)
}, py_function_name = "mean_squared_error")

loss_mean_squared_logarithmic_error <-
structure(function (y_true, y_pred, ..., reduction = "sum_over_batch_size", 
    name = "mean_squared_logarithmic_error") 
{
    args <- capture_args2(NULL)
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$losses$MeanSquaredLogarithmicError
    else keras$losses$mean_squared_logarithmic_error
    do.call(callable, args)
}, py_function_name = "mean_squared_logarithmic_error")

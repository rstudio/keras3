metric_poisson <-
structure(function (y_true, y_pred, ..., name = "poisson", dtype = NULL) 
{
    args <- capture_args2(NULL)
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$metrics$Poisson
    else keras$metrics$poisson
    do.call(callable, args)
}, py_function_name = "poisson")

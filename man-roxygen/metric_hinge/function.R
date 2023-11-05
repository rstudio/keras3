metric_hinge <-
structure(function (y_true, y_pred, ..., name = "hinge", dtype = NULL) 
{
    args <- capture_args2(NULL)
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$metrics$Hinge
    else keras$metrics$hinge
    do.call(callable, args)
}, py_function_name = "hinge")

metric_squared_hinge <-
structure(function (y_true, y_pred, ..., name = "squared_hinge", 
    dtype = NULL) 
{
    args <- capture_args2(NULL)
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$metrics$SquaredHinge
    else keras$metrics$squared_hinge
    do.call(callable, args)
}, py_function_name = "squared_hinge")

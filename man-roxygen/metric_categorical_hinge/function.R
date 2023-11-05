metric_categorical_hinge <-
structure(function (y_true, y_pred, ..., name = "categorical_hinge", 
    dtype = NULL) 
{
    args <- capture_args2(NULL)
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$metrics$CategoricalHinge
    else keras$metrics$categorical_hinge
    do.call(callable, args)
}, py_function_name = "categorical_hinge")

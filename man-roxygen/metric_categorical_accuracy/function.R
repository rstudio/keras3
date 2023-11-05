metric_categorical_accuracy <-
structure(function (y_true, y_pred, ..., name = "categorical_accuracy", 
    dtype = NULL) 
{
    args <- capture_args2(NULL)
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$metrics$CategoricalAccuracy
    else keras$metrics$categorical_accuracy
    do.call(callable, args)
}, py_function_name = "categorical_accuracy")

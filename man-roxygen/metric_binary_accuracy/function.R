metric_binary_accuracy <-
structure(function (y_true, y_pred, threshold = 0.5, ..., name = "binary_accuracy", 
    dtype = NULL) 
{
    args <- capture_args2(NULL)
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$metrics$BinaryAccuracy
    else keras$metrics$binary_accuracy
    do.call(callable, args)
}, py_function_name = "binary_accuracy")

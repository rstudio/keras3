metric_top_k_categorical_accuracy <-
structure(function (y_true, y_pred, k = 5L, ..., name = "top_k_categorical_accuracy", 
    dtype = NULL) 
{
    args <- capture_args2(list(k = as_integer))
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$metrics$TopKCategoricalAccuracy
    else keras$metrics$top_k_categorical_accuracy
    do.call(callable, args)
}, py_function_name = "top_k_categorical_accuracy")

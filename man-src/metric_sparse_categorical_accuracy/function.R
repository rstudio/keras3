metric_sparse_categorical_accuracy <-
structure(function (y_true, y_pred, ..., name = "sparse_categorical_accuracy", 
    dtype = NULL) 
{
    args <- capture_args2(NULL)
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$metrics$SparseCategoricalAccuracy
    else keras$metrics$sparse_categorical_accuracy
    do.call(callable, args)
}, py_function_name = "sparse_categorical_accuracy")

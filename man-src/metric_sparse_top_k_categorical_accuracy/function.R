metric_sparse_top_k_categorical_accuracy <-
structure(function (y_true, y_pred, k = 5L, ..., name = "sparse_top_k_categorical_accuracy", 
    dtype = NULL) 
{
    args <- capture_args2(list(k = as_integer))
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$metrics$SparseTopKCategoricalAccuracy
    else keras$metrics$sparse_top_k_categorical_accuracy
    do.call(callable, args)
}, py_function_name = "sparse_top_k_categorical_accuracy")

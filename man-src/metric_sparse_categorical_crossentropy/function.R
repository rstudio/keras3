metric_sparse_categorical_crossentropy <-
structure(function (y_true, y_pred, from_logits = FALSE, ignore_class = NULL, 
    axis = -1L, ..., name = "sparse_categorical_crossentropy", 
    dtype = NULL) 
{
    args <- capture_args2(list(axis = as_axis, ignore_class = as_integer))
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$metrics$SparseCategoricalCrossentropy
    else keras$metrics$sparse_categorical_crossentropy
    do.call(callable, args)
}, py_function_name = "sparse_categorical_crossentropy")

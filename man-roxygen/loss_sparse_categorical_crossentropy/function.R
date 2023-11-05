loss_sparse_categorical_crossentropy <-
structure(function (y_true, y_pred, from_logits = FALSE, ignore_class = NULL, 
    axis = -1L, ..., reduction = "sum_over_batch_size", name = "sparse_categorical_crossentropy") 
{
    args <- capture_args2(list(ignore_class = as_integer, axis = as_axis))
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$losses$SparseCategoricalCrossentropy
    else keras$losses$sparse_categorical_crossentropy
    do.call(callable, args)
}, py_function_name = "sparse_categorical_crossentropy")

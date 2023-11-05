loss_binary_crossentropy <-
structure(function (y_true, y_pred, from_logits = FALSE, label_smoothing = 0, 
    axis = -1L, ..., reduction = "sum_over_batch_size", name = "binary_crossentropy") 
{
    args <- capture_args2(list(axis = as_axis))
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$losses$BinaryCrossentropy
    else keras$losses$binary_crossentropy
    do.call(callable, args)
}, py_function_name = "binary_crossentropy")

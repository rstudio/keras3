metric_binary_crossentropy <-
structure(function (y_true, y_pred, from_logits = FALSE, label_smoothing = 0, 
    axis = -1L, ..., name = "binary_crossentropy", dtype = NULL) 
{
    args <- capture_args2(list(label_smoothing = as_integer, 
        axis = as_axis))
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$metrics$BinaryCrossentropy
    else keras$metrics$binary_crossentropy
    do.call(callable, args)
}, py_function_name = "binary_crossentropy")

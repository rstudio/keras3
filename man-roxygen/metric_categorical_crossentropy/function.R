metric_categorical_crossentropy <-
structure(function (y_true, y_pred, from_logits = FALSE, label_smoothing = 0, 
    axis = -1L, ..., name = "categorical_crossentropy", dtype = NULL) 
{
    args <- capture_args2(list(label_smoothing = as_integer, 
        axis = as_axis))
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$metrics$CategoricalCrossentropy
    else keras$metrics$categorical_crossentropy
    do.call(callable, args)
}, py_function_name = "categorical_crossentropy")

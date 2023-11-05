loss_binary_focal_crossentropy <-
structure(function (y_true, y_pred, apply_class_balancing = FALSE, 
    alpha = 0.25, gamma = 2, from_logits = FALSE, label_smoothing = 0, 
    axis = -1L, ..., reduction = "sum_over_batch_size", name = "binary_focal_crossentropy") 
{
    args <- capture_args2(list(axis = as_axis))
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$losses$BinaryFocalCrossentropy
    else keras$losses$binary_focal_crossentropy
    do.call(callable, args)
}, py_function_name = "binary_focal_crossentropy")

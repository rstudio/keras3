metric_binary_focal_crossentropy <-
function (y_true, y_pred, apply_class_balancing = FALSE, alpha = 0.25, 
    gamma = 2, from_logits = FALSE, label_smoothing = 0, axis = -1L) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$metrics$binary_focal_crossentropy, args)
}

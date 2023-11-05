metric_categorical_focal_crossentropy <-
function (y_true, y_pred, alpha = 0.25, gamma = 2, from_logits = FALSE, 
    label_smoothing = 0, axis = -1L) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$metrics$categorical_focal_crossentropy, args)
}

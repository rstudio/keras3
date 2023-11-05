metric_iou <-
function (..., num_classes, target_class_ids, name = NULL, dtype = NULL, 
    ignore_class = NULL, sparse_y_true = TRUE, sparse_y_pred = TRUE, 
    axis = -1L) 
{
    args <- capture_args2(list(ignore_class = as_integer, axis = as_axis))
    do.call(keras$metrics$IoU, args)
}

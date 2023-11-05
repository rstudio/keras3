metric_one_hot_iou <-
function (..., num_classes, target_class_ids, name = NULL, dtype = NULL, 
    ignore_class = NULL, sparse_y_pred = FALSE, axis = -1L) 
{
    args <- capture_args2(list(ignore_class = as_integer, axis = as_axis))
    do.call(keras$metrics$OneHotIoU, args)
}

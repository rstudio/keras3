metric_binary_iou <-
function (..., target_class_ids = list(0L, 1L), threshold = 0.5, 
    name = NULL, dtype = NULL) 
{
    args <- capture_args2(NULL)
    do.call(keras$metrics$BinaryIoU, args)
}

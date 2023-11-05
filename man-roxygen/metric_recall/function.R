metric_recall <-
function (..., thresholds = NULL, top_k = NULL, class_id = NULL, 
    name = NULL, dtype = NULL) 
{
    args <- capture_args2(list(top_k = as_integer, class_id = as_integer))
    do.call(keras$metrics$Recall, args)
}

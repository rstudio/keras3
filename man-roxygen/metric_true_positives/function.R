metric_true_positives <-
function (..., thresholds = NULL, name = NULL, dtype = NULL) 
{
    args <- capture_args2(NULL)
    do.call(keras$metrics$TruePositives, args)
}

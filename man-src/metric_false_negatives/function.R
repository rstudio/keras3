metric_false_negatives <-
function (..., thresholds = NULL, name = NULL, dtype = NULL) 
{
    args <- capture_args2(NULL)
    do.call(keras$metrics$FalseNegatives, args)
}

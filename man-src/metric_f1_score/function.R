metric_f1_score <-
function (..., average = NULL, threshold = NULL, name = "f1_score", 
    dtype = NULL) 
{
    args <- capture_args2(NULL)
    do.call(keras$metrics$F1Score, args)
}

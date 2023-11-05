metric_fbeta_score <-
function (..., average = NULL, beta = 1, threshold = NULL, name = "fbeta_score", 
    dtype = NULL) 
{
    args <- capture_args2(NULL)
    do.call(keras$metrics$FBetaScore, args)
}

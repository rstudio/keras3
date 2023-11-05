metric_mean <-
function (..., name = "mean", dtype = NULL) 
{
    args <- capture_args2(NULL)
    do.call(keras$metrics$Mean, args)
}

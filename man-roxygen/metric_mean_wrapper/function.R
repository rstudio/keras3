metric_mean_wrapper <-
function (..., fn, name = NULL, dtype = NULL) 
{
    args <- capture_args2(NULL)
    do.call(keras$metrics$MeanMetricWrapper, args)
}

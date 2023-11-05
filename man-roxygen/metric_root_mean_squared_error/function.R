metric_root_mean_squared_error <-
function (..., name = "root_mean_squared_error", dtype = NULL) 
{
    args <- capture_args2(NULL)
    do.call(keras$metrics$RootMeanSquaredError, args)
}

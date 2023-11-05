metric_log_cosh_error <-
function (..., name = "logcosh", dtype = NULL) 
{
    args <- capture_args2(NULL)
    do.call(keras$metrics$LogCoshError, args)
}

metric_log_cosh <-
function (y_true, y_pred) 
{
    args <- capture_args2(NULL)
    do.call(keras$metrics$log_cosh, args)
}

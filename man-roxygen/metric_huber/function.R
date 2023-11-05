metric_huber <-
function (y_true, y_pred, delta = 1) 
{
    args <- capture_args2(NULL)
    do.call(keras$metrics$huber, args)
}

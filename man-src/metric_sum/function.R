metric_sum <-
function (..., name = "sum", dtype = NULL) 
{
    args <- capture_args2(NULL)
    do.call(keras$metrics$Sum, args)
}

metric_r2_score <-
function (..., class_aggregation = "uniform_average", num_regressors = 0L, 
    name = "r2_score", dtype = NULL) 
{
    args <- capture_args2(list(num_regressors = as_integer))
    do.call(keras$metrics$R2Score, args)
}

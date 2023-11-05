metric_kl_divergence <-
structure(function (y_true, y_pred, ..., name = "kl_divergence", 
    dtype = NULL) 
{
    args <- capture_args2(NULL)
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$metrics$KLDivergence
    else keras$metrics$kl_divergence
    do.call(callable, args)
}, py_function_name = "kl_divergence")

loss_kl_divergence <-
structure(function (y_true, y_pred, ..., reduction = "sum_over_batch_size", 
    name = "kl_divergence") 
{
    args <- capture_args2(NULL)
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$losses$KLDivergence
    else keras$losses$kl_divergence
    do.call(callable, args)
}, py_function_name = "kl_divergence")

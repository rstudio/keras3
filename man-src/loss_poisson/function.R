loss_poisson <-
structure(function (y_true, y_pred, ..., reduction = "sum_over_batch_size", 
    name = "poisson") 
{
    args <- capture_args2(NULL)
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$losses$Poisson
    else keras$losses$poisson
    do.call(callable, args)
}, py_function_name = "poisson")

loss_huber <-
structure(function (y_true, y_pred, delta = 1, ..., reduction = "sum_over_batch_size", 
    name = "huber_loss") 
{
    args <- capture_args2(NULL)
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$losses$Huber
    else keras$losses$huber
    do.call(callable, args)
}, py_function_name = "huber")

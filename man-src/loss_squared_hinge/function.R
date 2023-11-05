loss_squared_hinge <-
structure(function (y_true, y_pred, ..., reduction = "sum_over_batch_size", 
    name = "squared_hinge") 
{
    args <- capture_args2(NULL)
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$losses$SquaredHinge
    else keras$losses$squared_hinge
    do.call(callable, args)
}, py_function_name = "squared_hinge")

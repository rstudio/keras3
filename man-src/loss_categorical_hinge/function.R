loss_categorical_hinge <-
structure(function (y_true, y_pred, ..., reduction = "sum_over_batch_size", 
    name = "categorical_hinge") 
{
    args <- capture_args2(NULL)
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$losses$CategoricalHinge
    else keras$losses$categorical_hinge
    do.call(callable, args)
}, py_function_name = "categorical_hinge")

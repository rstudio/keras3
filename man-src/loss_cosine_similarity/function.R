loss_cosine_similarity <-
structure(function (y_true, y_pred, axis = -1L, ..., reduction = "sum_over_batch_size", 
    name = "cosine_similarity") 
{
    args <- capture_args2(list(axis = as_axis))
    callable <- if (missing(y_true) && missing(y_pred)) 
        keras$losses$CosineSimilarity
    else keras$losses$cosine_similarity
    do.call(callable, args)
}, py_function_name = "cosine_similarity")

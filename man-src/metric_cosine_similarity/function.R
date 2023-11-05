metric_cosine_similarity <-
function (..., name = "cosine_similarity", dtype = NULL, axis = -1L) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$metrics$CosineSimilarity, args)
}

random_categorical <-
function (logits, num_samples, dtype = "int32", seed = NULL) 
{
    args <- capture_args2(list(num_samples = as_integer, seed = as_integer))
    do.call(keras$random$categorical, args)
}

set_random_seed <-
function (seed) 
{
    args <- capture_args2(list(seed = as_integer))
    do.call(keras$utils$set_random_seed, args)
}

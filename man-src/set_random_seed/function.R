set_random_seed <-
function (seed) 
{
    args <- capture_args2(list(seed = as_integer))
    set.seed(args$seed)
    do.call(keras$utils$set_random_seed, args)
}

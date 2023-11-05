k_round <-
function (x, decimals = 0L) 
{
    args <- capture_args2(list(decimals = as_integer))
    do.call(keras$ops$round, args)
}

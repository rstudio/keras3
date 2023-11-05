split_dataset <-
function (dataset, left_size = NULL, right_size = NULL, shuffle = FALSE, 
    seed = NULL) 
{
    args <- capture_args2(list(left_size = as_integer, right_size = as_integer, 
        seed = as_integer))
    do.call(keras$utils$split_dataset, args)
}

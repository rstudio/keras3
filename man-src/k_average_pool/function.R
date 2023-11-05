k_average_pool <-
function (inputs, pool_size, strides = NULL, padding = "valid", 
    data_format = NULL) 
{
    args <- capture_args2(list(pool_size = as_integer, strides = as_integer))
    do.call(keras$ops$average_pool, args)
}

unpack_x_y_sample_weight <-
function (data) 
{
    args <- capture_args2(NULL)
    do.call(keras$utils$unpack_x_y_sample_weight, args)
}

pack_x_y_sample_weight <-
function (x, y = NULL, sample_weight = NULL) 
{
    args <- capture_args2(NULL)
    do.call(keras$utils$pack_x_y_sample_weight, args)
}

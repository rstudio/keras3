config_set_floatx <-
function (value) 
{
    args <- capture_args2(NULL)
    do.call(keras$config$set_floatx, args)
}

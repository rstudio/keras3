config_floatx <-
function () 
{
    args <- capture_args2(NULL)
    do.call(keras$config$floatx, args)
}

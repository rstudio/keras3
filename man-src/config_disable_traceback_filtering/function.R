config_disable_traceback_filtering <-
function () 
{
    args <- capture_args2(NULL)
    do.call(keras$config$disable_traceback_filtering, args)
}

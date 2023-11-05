config_enable_traceback_filtering <-
function () 
{
    args <- capture_args2(NULL)
    do.call(keras$config$enable_traceback_filtering, args)
}

config_is_traceback_filtering_enabled <-
function () 
{
    args <- capture_args2(NULL)
    do.call(keras$config$is_traceback_filtering_enabled, args)
}

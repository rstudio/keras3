config_is_interactive_logging_enabled <-
function () 
{
    args <- capture_args2(NULL)
    do.call(keras$config$is_interactive_logging_enabled, args)
}

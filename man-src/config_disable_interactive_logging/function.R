config_disable_interactive_logging <-
function () 
{
    args <- capture_args2(NULL)
    do.call(keras$config$disable_interactive_logging, args)
}

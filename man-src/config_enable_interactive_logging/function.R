config_enable_interactive_logging <-
function () 
{
    args <- capture_args2(NULL)
    do.call(keras$config$enable_interactive_logging, args)
}

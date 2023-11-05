config_backend <-
function () 
{
    args <- capture_args2(NULL)
    do.call(keras$config$backend, args)
}

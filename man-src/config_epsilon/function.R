config_epsilon <-
function () 
{
    args <- capture_args2(NULL)
    do.call(keras$config$epsilon, args)
}

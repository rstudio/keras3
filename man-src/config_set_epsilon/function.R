config_set_epsilon <-
function (value) 
{
    args <- capture_args2(NULL)
    do.call(keras$config$set_epsilon, args)
}

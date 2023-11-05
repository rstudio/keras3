callback_terminate_on_nan <-
function () 
{
    args <- capture_args2(NULL)
    do.call(keras$callbacks$TerminateOnNaN, args)
}

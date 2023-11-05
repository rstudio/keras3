callback_progbar_logger <-
function (count_mode = NULL) 
{
    args <- capture_args2(NULL)
    do.call(keras$callbacks$ProgbarLogger, args)
}

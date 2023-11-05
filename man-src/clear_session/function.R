clear_session <-
function () 
{
    args <- capture_args2(NULL)
    do.call(keras$utils$clear_session, args)
}

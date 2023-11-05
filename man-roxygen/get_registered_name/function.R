get_registered_name <-
function (obj) 
{
    args <- capture_args2(NULL)
    do.call(keras$utils$get_registered_name, args)
}
